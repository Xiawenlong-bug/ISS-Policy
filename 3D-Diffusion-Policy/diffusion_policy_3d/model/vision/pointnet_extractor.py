import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules




class PointNetEncoderXYZRGB(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256, 512]
        cprint("pointnet use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("pointnet use_final_norm: {}".format(final_norm), 'cyan')
        
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[2], block_channel[3]),
        )
        
       
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


from .pointnet2_utils import PointNetSetAbstraction

class PointNetPlusEncoderXYZ(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 1024,
                 use_layernorm: bool = False,
                 final_norm: str = 'none',
                 use_projection: bool = True,
                 # 与 yanx27 默认一致的 SSG 参数
                 sa1_npoint: int = 1024, sa1_radius: float = 0.1,  sa1_nsample: int = 32,  sa1_mlp: List[int] = (64, 64, 128),
                 sa2_npoint: int = 256,  sa2_radius: float = 0.2,  sa2_nsample: int = 32,  sa2_mlp: List[int] = (128, 128, 256),
                 sa3_npoint: int = 64,   sa3_radius: float = 0.4,  sa3_nsample: int = 32,  sa3_mlp: List[int] = (256, 256, 512),
                 post_mlp: List[int] = (),
                 **kwargs):
        super().__init__()
        self.use_projection = use_projection
        self.final_norm = final_norm
        self.sa1 = PointNetSetAbstraction(sa1_npoint, sa1_radius, sa1_nsample, in_channel=3, mlp=sa1_mlp, group_all=False)
        self.sa2 = PointNetSetAbstraction(sa2_npoint, sa2_radius, sa2_nsample, in_channel=sa1_mlp[-1] + 3, mlp=sa2_mlp, group_all=False)
        self.sa3 = PointNetSetAbstraction(sa3_npoint, sa3_radius, sa3_nsample, in_channel=sa2_mlp[-1] + 3, mlp=sa3_mlp, group_all=False)

        # 全局池化 + 头
        head = []
        in_dim = sa3_mlp[-1]
        for h in post_mlp:
            head += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            in_dim = h
        head += [nn.Linear(in_dim, out_channels)]
        if final_norm == 'layernorm':
            head += [nn.LayerNorm(out_channels)]
        elif final_norm == 'none':
            pass
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
        self.final_projection = nn.Sequential(*head) 
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetPlusEncoderXYZ] not use projection", "yellow")
        self.n_output_channels = out_channels     

    def forward(self, x):
        xyz, points = x.transpose(2, 1), None  # (B,3,N)
        xyz, points = self.sa1(xyz, points)
        xyz, points = self.sa2(xyz, points)
        xyz, points = self.sa3(xyz, points)      # points: (B, C3, S3)
        feat = torch.max(points, 2)[0]           # (B, C3)
        out = self.final_projection(feat)        # (B, out_channels)
        return out


    def output_shape(self):
        return self.n_output_channels


class PointNetEncoderXYZ(nn.Module):
    """Encoder for Pointcloud
    """

    def __init__(self,
                 in_channels: int=3,
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 use_projection: bool=True,
                 **kwargs
                 ):
        """_summary_

        Args:
            in_channels (int): feature size of input (3 or 6)
            input_transform (bool, optional): whether to use transformation for coordinates. Defaults to True.
            feature_transform (bool, optional): whether to use transformation for features. Defaults to True.
            is_seg (bool, optional): for segmentation or classification. Defaults to False.
        """
        super().__init__()
        block_channel = [64, 128, 256]
        # block_channel = [128, 256, 512,512]
        cprint("[PointNetEncoderXYZ] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[PointNetEncoderXYZ] use_final_norm: {}".format(final_norm), 'cyan')
        
        assert in_channels == 3, cprint(f"PointNetEncoderXYZ only supports 3 channels, but got {in_channels}", "red")
       
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.SiLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.SiLU(),
            # nn.Linear(block_channel[2], block_channel[3]),
            # nn.LayerNorm(block_channel[3]) if use_layernorm else nn.Identity(),
            # nn.SiLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")

        self.use_projection = use_projection
        if not use_projection:
            self.final_projection = nn.Identity()
            cprint("[PointNetEncoderXYZ] not use projection", "yellow")
            
        VIS_WITH_GRAD_CAM = False
        if VIS_WITH_GRAD_CAM:
            self.gradient = None
            self.feature = None
            self.input_pointcloud = None
            self.mlp[0].register_forward_hook(self.save_input)
            self.mlp[6].register_forward_hook(self.save_feature)
            self.mlp[6].register_backward_hook(self.save_gradient)
         
         
    def forward(self, x):
        x = self.mlp(x)
        x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x
    
    def save_gradient(self, module, grad_input, grad_output):
        """
        for grad-cam
        """
        self.gradient = grad_output[0]

    def save_feature(self, module, input, output):
        """
        for grad-cam
        """
        if isinstance(output, tuple):
            self.feature = output[0].detach()
        else:
            self.feature = output.detach()
    
    def save_input(self, module, input, output):
        """
        for grad-cam
        """
        self.input_pointcloud = input[0].detach()


class PointCloudEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg) #this
        elif pointnet_type == "pointnet_plus":
            self.extractor = PointNetPlusEncoderXYZ(**pointcloud_encoder_cfg)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

    def forward(self, observations: Union[Dict, torch.Tensor]) -> torch.Tensor:
        # 判断输入类型：如果是tensor，直接使用；如果是字典，从字典中取出point_cloud
        if isinstance(observations, torch.Tensor):
            points = observations
        elif isinstance(observations, dict):
            points = observations[self.point_cloud_key]
        else:
            raise TypeError(f"observations must be either Dict or torch.Tensor, but got {type(observations)}")
        
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        
        # 只有当输入是字典且use_imagined_robot为True时，才添加imagined robot的点云
        if isinstance(observations, dict) and self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        pn_feat = self.extractor(points)    # B * out_channel
            
        return pn_feat


    def output_shape(self):
        return self.n_output_channels


class StateEncoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.rgb_image_key = 'image'
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        # StateEncoder 只输出 state_feat，维度是 output_dim
        self.n_output_channels = output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

    def forward(self, observations: Dict) -> torch.Tensor:
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * output_dim
        return state_feat


    def output_shape(self):
        return self.n_output_channels



class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 img_crop_shape=None,
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 pointcloud_encoder_cfg=None,
                 use_pc_color=False,
                 pointnet_type='pointnet',
                 ):
        super().__init__()
        self.imagination_key = 'imagin_robot'
        self.state_key = 'agent_pos'
        self.point_cloud_key = 'point_cloud'
        self.rgb_image_key = 'image'
        self.n_output_channels = out_channel
        
        self.use_imagined_robot = self.imagination_key in observation_space.keys()
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]
        if self.use_imagined_robot:
            self.imagination_shape = observation_space[self.imagination_key]
        else:
            self.imagination_shape = None
            
        
        
        cprint(f"[DP3Encoder] point cloud shape: {self.point_cloud_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] imagination point shape: {self.imagination_shape}", "yellow")
        

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type
        if pointnet_type == "pointnet":
            if use_pc_color:
                pointcloud_encoder_cfg.in_channels = 6
                self.extractor = PointNetEncoderXYZRGB(**pointcloud_encoder_cfg)
            else:
                pointcloud_encoder_cfg.in_channels = 3
                self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg) #this
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")


        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
        if self.use_imagined_robot:
            img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
            points = torch.concat([points, img_points], dim=1)
        
        #points : (B*T N 3)
        pn_feat = self.extractor(points)    # B * out_channel
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        # final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return pn_feat, state_feat
    # def forward(self, observations: Dict) -> torch.Tensor:
    #     points = observations[self.point_cloud_key]
    #     assert len(points.shape) == 3, cprint(f"point cloud shape: {points.shape}, length should be 3", "red")
    #     if self.use_imagined_robot:
    #         img_points = observations[self.imagination_key][..., :points.shape[-1]] # align the last dim
    #         points = torch.concat([points, img_points], dim=1)
        
    #     # points = torch.transpose(points, 1, 2)   # B * 3 * N
    #     # points: B * 3 * (N + sum(Ni))
    #     pn_feat = self.extractor(points)    # B * out_channel
            
    #     state = observations[self.state_key]
    #     state_feat = self.state_mlp(state)  # B * 64
    #     final_feat = torch.cat([pn_feat, state_feat], dim=-1)
    #     return final_feat

    def output_shape(self):
        return self.n_output_channels