# Multiple Thinking Achieving Meta-Ability Decoupling for Object Navigation (ICML 2023)
Ronghao Dang, Lu Chen, Liuyi Wang, Zongtao He, Chengju Liu, Qijun Chen

## Abstract
We propose a meta-ability decoupling (MAD) paradigm, which brings together various object navigation methods in an architecture system, allowing them to mutually enhance each other and evolve together. Based on the MAD paradigm, we design a multiple thinking (MT) model that leverages distinct thinking to abstract various metaabilities. Our method decouples meta-abilities from three aspects: input, encoding, and reward while employing the multiple thinking collaboration (MTC) module to promote mutual cooperation between thinking. MAD introduces a novel qualitative and quantitative interpretability system for object navigation. Through extensive experiments on AI2-Thor and RoboTHOR, we demonstrate that our method outperforms state-of-the-art (SOTA) methods on both typical and zero-shot object navigation tasks.

[Arxiv Paper](https://arxiv.org/abs/2302.01520)

<p align="center"><img src="images\architecture.jpg" width="800
" /></p>

## Setup
- Clone the repository and move into the top level directory
  ```shell
  git clone https://github.com/izilu/MT.git
  cd MT
  ```
- Create pretrain conda enviroment 
  ```shell
  conda create -n MT_Pretrain python=3.6
  pip install requirements_pretrain.txt
  ```
- Create main conda environment
  ```shell
  conda env create -f environment.yml
  ```
- Download the [dataset](https://drive.google.com/file/d/1kvYvutjqc6SLEO65yQjo8AuU85voT5sC/view), which refers to [ECCV-VN](https://github.com/xiaobaishu0097/ECCV-VN). The offline data is discretized from [AI2-Thor](https://ai2thor.allenai.org/) simulator.
- Download the [pretrain dataset](https://drive.google.com/file/d/1dFQV10i4IixaSUxN2Dtc6EGEayr661ce/view), which refers to [VTNet](https://github.com/xiaobaishu0097/ICLR_VTNet).
- You can also use the [DETR object detection features](https://drive.google.com/file/d/1d761VxrwctupzOat4qxsLCm5ndC4wA-M/view?usp=sharing).
  The `data` folder should look like this: 
  ```python
  data/ 
      └── Scene_Data/
          ├── FloorPlan1/
          │   ├── resnet18_featuremap.hdf5
          │   ├── graph.json
          │   ├── visible_object_map_1.5.json
          │   ├── detr_features_22cls.hdf5
          │   ├── grid.json
          │   └── optimal_action.json
          ├── FloorPlan2/
          └── ...
      └── AI2Thor_VisTrans_Pretrain_Data/
          ├── data/
          ├── annotation_train.json
          ├── annotation_val.json
          └── annotation_test.json
  ``` 

## Training and Evaluation
### Pre-train the search thinking network of our MT model
`python main_pretraining.py --title ST_Pretrain --model ST_Pretrain --workers 9 --gpu-ids 0 --epochs 20 --log-dir runs/pretrain --save-model-dir trained_models/pretrain`

### Train our MT model
`python main.py --title Multi_Thinking_4T --model Multi_Thinking_4T --workers 18 --gpu-ids 0 1 --max-ep 5000000 --save-model-dir trained_models/Multi_Thinking/multi_thinking_4T --log-dir runs/Multi_Thinking/multi_thinking_4T  --pretrained-trans trained_models/pretrain/checkpoint0003.pth` 

### Evaluate our MT model
`python full_eval.py --title Multi_Thinking_4T --model Multi_Thinking_4T --results-json eval_best_results/Multi_Thinking/multi_thinking_4T --gpu-ids  0 --workers 4 --save-model-dir trained_models/multi_thinking_4T --log-dir runs/multi_thinking_4T`

## Citing
If you find this project useful in your research, please consider citing:
```
@article{dang2023search,
  title={Multiple Thinking Achieving Meta-Ability Decoupling for Object Navigation},
  author={Ronghao Dang, Lu Chen, Liuyi Wang, Zongtao He, Chengju Liu, Qijun Chen},
  journal={arXiv preprint arXiv:2302.01520},
  year={2023}
}
```
