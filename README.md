# Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI
<div align="center">
  <a href="https://arxiv.org/abs/2511.20620" target="_blank">
      <img alt="arXiv" src="https://img.shields.io/badge/arXiv-Wanderland-red?logo=arxiv" height="25" />
  </a>
  <a href="https://ai4ce.github.io/wanderland/" target="_blank">
      <img alt="Website" src="https://img.shields.io/badge/🔮_Website-ai4ce.github.io-blue" height="25" />
  </a>
  <a href="https://huggingface.co/datasets/ai4ce/wanderland" target="_blank">
      <img alt="HF Dataset: Wanderland" src="https://img.shields.io/badge/%F0%9F%A4%97%20_Data-Wanderland-ffc107?color=ffc107&logoColor=white" height="25" />
  </a>

  <div style="font-family: charter;">
      <a href="https://gaaaavin.github.io/">Xinhao Liu</a>*,
      <a href="https://songard.github.io/">Jiaqi Li</a>*,
      <a href="https://denghilbert.github.io/">Youming Deng</a>,
      <a href="https://www.linkedin.com/in/rue-chen-174a63370/">Ruxin Chen</a>,
      <a href="#">Yingjia Zhang</a>,
      <a href="#">Yifei Ma</a>,
      <a href="https://cs.shanghai.nyu.edu/faculty/li-guo-guoli">Li Guo</a>,
      <a href="https://yimingli-page.github.io/">Yiming Li</a>,
      <a href="https://jingz6676.github.io">Jing Zhang</a>,
      <a href="https://engineering.nyu.edu/faculty/chen-feng">Chen Feng</a>
  </div>

  <br>

  ![](./assets/wanderland.gif)
</div>

# Release TODO
- [ ] Simulation Code Release
- [x] [Jan 17, 2026] 3D Reconstruction Benchmark Release
- [x] [Jan 17, 2026] Data Release

# Getting Started
Wanderland is a comprehensive framework that consists of many different components. 
Each component is relatively independent so please refer to the README.md file in each subfolder for more details.
Note that all components use [uv](https://docs.astral.sh/uv/) for dependency management. Please install uv first if you haven't done so.
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```
* To download data from Hugging Face, check [data_processing](data_processing/README.md)
* To benchmark 3D reconstruction methods, check [3d_recon_benchmark](3d_recon_benchmark/README.md)
* To evaluate navigation performance, check [navigation](navigation/README.md) [TODO]
* To reproduce our reconstruction pipeline, check [reconstruction](reconstruction/README.md)

Each component have its own virtual environment. Remember to `deactivate` and reactivate the venv when switching between different components.

# Citation
```
@article{liu2025wanderland,
  title={Wanderland: Geometrically Grounded Simulation for Open-World Embodied AI},
  author={Liu, Xinhao and Li, Jiaqi and Deng, Youming and Chen, Ruxin and Zhang, Yingjia and Ma, Yifei and Guo, Li and Li, Yiming and Zhang, Jing and Feng, Chen},
  journal={arXiv preprint arXiv:2511.20620},
  year={2025}
}
```

# Related Projects
**Real-to-Sim**:
* [CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos](https://github.com/ai4ce/CityWalker), CVPR 2025
* [Vid2Sim: Realistic and Interactive Simulation from Video for Urban Navigation](https://github.com/Vid2Sim/Vid2Sim), CVPR 2025
* [Gauss Gym: A Geometrically Grounded Simulation Environment for Embodied AI](https://github.com/escontra/gauss_gym), arXiv 2025
* [BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation](https://github.com/StanfordVL/BEHAVIOR-1K), arXiv 2024

**3D Reconstruction**:
* [VGGT: Visual Geometry Grounded Transformer](https://github.com/facebookresearch/vggt), CVPR 2025
* [π3 : Permutation-Equivariant Visual Geometry Learning](https://github.com/yyfz/Pi3), arXiv 2025
* [MapAnything: Universal Feed-Forward Metric 3D Reconstruction](https://github.com/facebookresearch/map-anything), arXiv 2025
* [Depth Anything 3: Recovering the Visual Space from Any Views](https://github.com/ByteDance-Seed/Depth-Anything-3), arXiv 2025

**Navigation**:
* [DeepExplorer: Metric-Free Exploration for Topological Mapping by Task and Motion Imitation in Feature Space](https://github.com/ai4ce/DeepExplorer), RSS 2023
* [CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos](https://github.com/ai4ce/CityWalker), CVPR 2025
* [NaVILA: Legged Robot Vision-Language-Action Model for Navigation](https://navila-bot.github.io/), RSS 2025
* [Learning to Drive Anywhere with Model-Based Reannotation](https://model-base-reannotation.github.io/), RA-L 2025
* [From Seeing to Experiencing: Scaling Navigation Foundation Models with Reinforcement Learning](https://metadriverse.github.io/s2e/), arXiv 2025

<a href="https://star-history.com/#ai4ce/wanderland&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=ai4ce/wanderland&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=ai4ce/wanderland&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=ai4ce/wanderland&type=Date" />
 </picture>
</a>
