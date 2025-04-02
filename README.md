# Rose
## Official implementation of the paper "Integrating remote sensing with OpenStreetMap data for comprehensive scene understanding through multi-modal self-supervised learning", a multi-modal self-supervised learning approach to fuse RS images and OSM data.

## Main Requirements
torch==1.10.0+cu111
torchvision==0.11.0+cu111
torch-cluster==1.6.0
torch_geometric==2.4.0
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-spline-conv==1.2.1
timm==0.6.12

## Pretraining Dataset
We collect RS images (Source: Bing Map and Arcgis Map) and OSM data in Shenzhen (China), Singapore, Chicago (U.S.). 
The pretraining dataset can be found at: 

## pretraining
```
python train_rose.py \
   data_path=path_to_your_pretraining_dataset \
   batch_size=64 \
   total_epoch=120 \
   schedule=[90,110]\
   save_root_dir=path_to_your_save_root
   save_dir=rose-path_to_your_save_dir
```
To check if the pre-training is working properly, one can refer to the train logs in loss_record.txt.
## downstream tasks
```
# semantic segmentation task
python evaluate_seg_fusion.py
```
```
# pop and CO2 tasks
python evaluate_pop_fusion.py
```
Downstream evaluation dataset will come soon

```bash
@article{bai2025integrating,
  title={Integrating remote sensing with OpenStreetMap data for comprehensive scene understanding through multi-modal self-supervised learning},
  author={Bai, Lubin and Zhang, Xiuyuan and Wang, Haoyu and Du, Shihong},
  journal={Remote Sensing of Environment},
  volume={318},
  pages={114573},
  year={2025},
  publisher={Elsevier}
}
```
