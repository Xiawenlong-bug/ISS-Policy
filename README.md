<div align="center">   

# ISS Policy : Scalable Diffusion Policy with Implicit Scene Supervision
</div>

<div align="center">


[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2512.15020)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)
</div>




# 💻 Installation

See [INSTALL.md](INSTALL.md) for installation instructions. 

See [ERROR_CATCH.md](ERROR_CATCH.md) for error catching I personally encountered during installation.


**Algorithms**. We provide the implementation of the following algorithms: 
- ISS Policy : `dit-iss.yaml`
- DP3: `dp3.yaml`
- Simple DP3: `simple_dp3.yaml`

`dit-iss.yaml` is the proposed algorithm in our paper, showing a significant improvement over the baselines.

# 📊 Benchmark of ISS Policy

**Simulation environments.** We provide dexterous manipulation environments and expert policies for `Adroit`, `DexArt`, and `MetaWorld` in this codebase (3+4+50=57 tasks in total). 



# 🛠️ Usage
Follow [DP3  codebase](https://github.com/YanjieZe/3D-Diffusion-Policy), scripts for generating demonstrations, training, and evaluation are all provided in the `scripts/` folder. 

The results are logged by `wandb`, so you need to `wandb login` first to see the results and videos.
For more detailed arguments, please refer to the scripts and the code. We here provide a simple instruction for using the codebase.

1. Generate demonstrations by `gen_demonstration_adroit.sh` and `gen_demonstration_dexart.sh`. See the scripts for details. For example:
    ```bash
    bash scripts/gen_demonstration_adroit.sh hammer
    ```
    This will generate demonstrations for the `hammer` task in Adroit environment. The data will be saved in `3D-Diffusion-Policy/data/` folder automatically.


2. Train and evaluate a policy with behavior cloning. For example:
    ```bash
    bash scripts/train_policy.sh dit-iss adroit_hammer 0112 0 0
    ```
    This will train ISS Policy on the `hammer` task in Adroit environment using point cloud modality. By default we **save** the ckpt (optional in the script).


3. Evaluate a saved policy or use it for inference. Please set  For example:
    ```bash
    bash scripts/eval_policy.sh dit-iss adroit_hammer 0112 0 0
    ```
    This will evaluate the saved ISS Policy you just trained. 
    
    **Note: the evaluation script is only provided for deployment/inference. For benchmarking, please use the results logged in wandb during training.**

# 😺 Acknowledgement
Our code is generally built upon: [DP3](https://github.com/YanjieZe/3D-Diffusion-Policy), [Diffusion Policy](https://github.com/real-stanford/diffusion_policy), [VRL3](https://github.com/microsoft/VRL3), [MetaWorld](https://github.com/Farama-Foundation/Metaworld). We thank all these authors for their nicely open sourced code and their great contributions to the community.

Contact `Wenlong Xia`(wenlongxia1@gmail.com) if you have any questions or suggestions.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation <a name="citation"></a>

```bibtex
@article{xia2025isspolicyscalable,
  title   = {ISS Policy : Scalable Diffusion Policy with Implicit Scene Supervision},
  author  = {Xia, Wenlong and Zhang, Jinhao and Zhang, Ce and Wang, Yaojia and Li, Huizhe and Gong, Youmin and Mei, Jie},
  journal = {arXiv preprint arXiv:2512.15020},
  year    = {2025}
}
```
