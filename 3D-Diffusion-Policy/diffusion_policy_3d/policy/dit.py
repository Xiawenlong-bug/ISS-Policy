from typing import Dict, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from termcolor import cprint
import copy
import time
import pytorch3d.ops as torch3d_ops

from diffusion_policy_3d.model.common.normalizer import LinearNormalizer
from diffusion_policy_3d.policy.base_policy import BasePolicy
from diffusion_policy_3d.model.diffusion.dit1d import DiT1D
from diffusion_policy_3d.model.diffusion.mask_generator import LowdimMaskGenerator
from diffusion_policy_3d.common.pytorch_util import dict_apply
from diffusion_policy_3d.common.model_util import print_params
from diffusion_policy_3d.model.vision.pointnet_extractor import PointCloudEncoder, StateEncoder
from diffusion_policy_3d.model.diffusion.iss import ISS

class DIT(BasePolicy):
    """Diffusion Policy with Transformer architecture and Implicit State Space model"""
    
    def __init__(self, 
            shape_meta: dict,
            noise_scheduler: DDPMScheduler,
            horizon: int, 
            n_action_steps: int, 
            n_obs_steps: int,
            num_inference_steps: Optional[int] = None,
            obs_as_global_cond: bool = True,
            diffusion_step_embed_dim: int = 256,
            # Transformer parameters
            hidden_size_dit: int = 768,
            hidden_size_iss: int = 512,
            depth: int = 12,
            num_heads: int = 12,
            mlp_ratio: float = 4.0,
            enable_cross_attention: bool = True,
            num_action_tokens: int = 1,
            encoder_output_dim: int = 256,
            crop_shape: Optional[tuple] = None,
            use_pc_color: bool = False,
            pointnet_type: str = "pointnet",
            pointcloud_encoder_cfg: Optional[dict] = None,
            next_pc_embed_dim: int = 64,
            iss_loss_weight: float = 0.1,
            gt_input_prob: float = 0.4,
            n_skip_steps: int = 2,
            **kwargs):
        super().__init__()

        self.enable_cross_attention = enable_cross_attention
        self.n_skip_steps = n_skip_steps
        self.iss_loss_weight = iss_loss_weight
        self.gt_input_prob = gt_input_prob

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2:  # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")
            
        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])
        point_cloud_encoder = PointCloudEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )
        
        state_encoder = StateEncoder(
            observation_space=obs_dict,
            img_crop_shape=crop_shape,
            out_channel=encoder_output_dim,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=use_pc_color,
            pointnet_type=pointnet_type,
        )

        # create diffusion model
        obs_feature_dim = point_cloud_encoder.output_shape() + state_encoder.output_shape()
        input_dim = action_dim + obs_feature_dim
        
        if obs_as_global_cond:
            # When using cross-attention, don't concatenate observation features in input
            input_dim = action_dim

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        cprint(f"[DIT] use_pc_color: {self.use_pc_color}", "yellow")
        cprint(f"[DIT] pointnet_type: {self.pointnet_type}", "yellow")
        model = DiT1D(
            input_dim=input_dim,
            hidden_size=hidden_size_dit,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            horizon=horizon,
        )
        
        iss_model = ISS(
            input_dim=input_dim,
            hidden_size=hidden_size_iss,
            output_dim=next_pc_embed_dim,
            n_skip_steps=n_skip_steps,
        )
        
        self.point_cloud_encoder = point_cloud_encoder
        self.state_encoder = state_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        self.iss = iss_model
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
                
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.kwargs = kwargs
        
        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        print_params(self)
        
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            condition_data_pc=None, condition_mask_pc=None,
            local_cond=None, global_cond=None,
            pc_cond=None, state_cond=None,
            generator=None,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        model = self.model
        scheduler = self.noise_scheduler


        trajectory = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)


        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]


            model_output = model(sample=trajectory,
                                timestep=t, 
                                local_cond=local_cond,
                                pc_cond=pc_cond,
                                state_cond=state_cond,
                                global_cond=global_cond)
            
            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, ).prev_sample
            
                
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]   


        return trajectory


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        this_n_point_cloud = nobs['point_cloud']
        
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        pc_cond = None
        state_cond_inf = None
        if self.obs_as_global_cond:
            # condition through global feature
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            # 预测时也构造 pc/state 条件 (B, D) 或 (B, L*D)
            pn_feat = self.point_cloud_encoder(this_nobs)
            state_feat = self.state_encoder(this_nobs)
            # pc_cond = pn_feat.reshape(B, -1)
            # state_cond_inf = state_feat.reshape(B, -1)
            pc_cond = rearrange(pn_feat, '(b t) d -> b (t d)', t=self.n_obs_steps)
            state_cond_inf = rearrange(state_feat, '(b t) d -> b (t d)', t=self.n_obs_steps)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
            pn_feat = self.point_cloud_encoder(this_nobs)
            state_feat = self.state_encoder(this_nobs)
            nobs_features = torch.cat([pn_feat, state_feat], dim=-1)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da+Do), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:,:To,Da:] = nobs_features
            cond_mask[:,:To,Da:] = True

        # run sampling
        nsample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            pc_cond=pc_cond if self.obs_as_global_cond else None,
            state_cond=state_cond_inf if self.obs_as_global_cond else None,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction


        result = {
            'action': action,
            'action_pred': action_pred,
        }
        
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input

        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action'])

        if not self.use_pc_color:
            nobs['point_cloud'] = nobs['point_cloud'][..., :3]
        
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions
        cond_data = trajectory
        
        pn_feat = None
        state_feat = None
        skip_pn_feat = None
        if self.obs_as_global_cond:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, 
                lambda x: x[:,:self.n_obs_steps,...].reshape(-1,*x.shape[2:]))
            # Predict point cloud features using ISS model
            skip_frame_idx = self.n_obs_steps - 1 + self.n_skip_steps
            skip_cloud_points = nobs['point_cloud'][:, skip_frame_idx]
            skip_pn_feat = self.point_cloud_encoder(skip_cloud_points)
            pn_feat = self.point_cloud_encoder(this_nobs)
            state_feat = self.state_encoder(this_nobs) 
            nobs_features = torch.cat([pn_feat, state_feat], dim=-1)
            pn_feat = rearrange(pn_feat, '(b t) d -> b (t d)', t=self.n_obs_steps)
            state_feat = rearrange(state_feat, '(b t) d -> b (t d)', t=self.n_obs_steps)
            if self.enable_cross_attention:
                # In cross-attention mode, keep sequence shape (B, n_obs_steps, obs_feature_dim)
                global_cond = nobs_features.reshape(batch_size, self.n_obs_steps, -1)
            else:
                # In other modes, flatten to (B, obs_feature_dim * n_obs_steps)
                global_cond = nobs_features.reshape(batch_size, -1)
            this_n_point_cloud = this_nobs['point_cloud'].reshape(batch_size,-1, *this_nobs['point_cloud'].shape[1:])
            this_n_point_cloud = this_n_point_cloud[..., :3]
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            pn_feat = self.point_cloud_encoder(this_nobs)
            state_feat = self.state_encoder(this_nobs)
            nobs_features = torch.cat([pn_feat, state_feat], dim=-1)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()


        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)

        
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        


        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        
        pred = self.model(sample=noisy_trajectory, 
                        timestep=timesteps, 
                            local_cond=local_cond, 
                            pc_cond = pn_feat,
                            state_cond = state_feat)

        start_idx = self.n_obs_steps - 1
        end_idx = start_idx + self.n_skip_steps
        # 以一定概率使用pred或GT作为输入（teacher forcing）
        if self.training and self.gt_input_prob > 0:
            # 为每个样本独立生成随机数，决定该样本使用GT还是pred
            batch_size = pred.shape[0]
            use_gt_mask = torch.rand(batch_size, device=pred.device) < self.gt_input_prob
            # 样本级混合：根据mask选择每个样本使用GT还是pred
            gt_input = trajectory[:, start_idx:end_idx]  # GT输入
            pred_input = pred[:, start_idx:end_idx]  # 模型预测输入
            # 扩展mask维度以匹配输入张量的形状 (batch_size, n_skip_steps, feature_dim)
            use_gt_mask = use_gt_mask.view(batch_size, 1, 1)
            iss_action_input = torch.where(
                use_gt_mask, 
                gt_input, 
                pred_input
            )
        else:
            # 推理时或gt_input_prob=0时，只用pred
            iss_action_input = pred[:, start_idx:end_idx]
        
        # Predict next point cloud features using ISS model
        skip_pc_pred = self.iss(iss_action_input, pn_feat.detach())

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        elif pred_type == 'v_prediction':
            # https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # https://github.com/huggingface/diffusers/blob/v0.11.1-patch/src/diffusers/schedulers/scheduling_dpmsolver_multistep.py
            # sigma = self.noise_scheduler.sigmas[timesteps]
            # alpha_t, sigma_t = self.noise_scheduler._sigma_to_alpha_sigma_t(sigma)
            self.noise_scheduler.alpha_t = self.noise_scheduler.alpha_t.to(self.device)
            self.noise_scheduler.sigma_t = self.noise_scheduler.sigma_t.to(self.device)
            alpha_t, sigma_t = self.noise_scheduler.alpha_t[timesteps], self.noise_scheduler.sigma_t[timesteps]
            alpha_t = alpha_t.unsqueeze(-1).unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1).unsqueeze(-1)
            v_t = alpha_t * noise - sigma_t * trajectory
            target = v_t
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss_bc = F.mse_loss(pred, target, reduction='none')
        loss_bc = loss_bc * loss_mask.type(loss_bc.dtype)
        loss_bc = reduce(loss_bc, 'b ... -> b', 'mean') 
        loss_bc = loss_bc.mean()  # scalar

        # ISS loss for point cloud prediction
        loss_iss = F.mse_loss(skip_pc_pred, skip_pn_feat, reduction='none')
        loss_iss = reduce(loss_iss, 'b ... -> b', 'mean')
        loss_iss = loss_iss.mean()  # scalar
        
        # Total loss
        loss = loss_bc + loss_iss * self.iss_loss_weight
        

        loss_dict = {
            'bc_loss': loss_bc.item(),
            'iss_loss': loss_iss.item(),
            'total_loss': loss.item(),
        }

        return loss, loss_dict