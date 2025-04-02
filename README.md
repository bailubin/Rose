# Rose
## Official implementation of the paper "Integrating remote sensing with OpenStreetMap data for comprehensive scene understanding through multi-modal self-supervised learning", a multi-modal self-supervised learning approach to fuse RS images and OSM data.
> **Abstract.**
> OpenStreetMap (OSM) contains valuable geographic knowledge for remote sensing (RS) interpretation. They can provide correlated and complementary descriptions of a given region. Integrating RS images with OSM data can lead to a more comprehensive understanding of a geographic scene. But due to the significant differences between them, little progress has been made in data fusion for RS and OSM data, and how to extract, interact, and collaborate the information from multiple geographic data sources remains largely unexplored. In this work, we focus on designing a multi-modal self-supervised learning (SSL) approach to fuse RS images and OSM data, which can extract meaningful features from the two complementary data sources in an unsupervised manner, resulting in comprehensive scene understanding. We harmonize the parts of information extraction, interaction, and collaboration for RS and OSM data into a unified SSL framework, named Rose. For information extraction, we start from the complementarity between the two modalities, designing an OSM encoder to harmoniously align with the ViT image encoder. For information interaction, we leverage the spatial correlation between RS and OSM data to guide the cross-attention module, thereby enhancing the information transfer. For information collaboration, we design the joint mask-reconstruction learning strategy to achieve cooperation between the two modalities, which reconstructs the original inputs by referring to information from both sources. The three parts are interlinked and blending seamlessly into a unified framework. Finally, Rose can generate three kinds of representations, i.e., RS feature, OSM feature, and RS-OSM fusion feature, which can be used for multiple downstream tasks. Extensive experiments on land use semantic segmentation, population estimation, and carbon emission estimation tasks demonstrate the multitasking capability, label efficiency, and robustness to noise of Rose. Rose can associate RS images and OSM data at a fine level of granularity, enhancing its effectiveness on fine-grained tasks like land use semantic segmentation.


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
To check if the pretraining is working properly, one can refer to the train logs in loss_record.txt.
## downstream tasks
```
# semantic segmentation task
python evaluate_seg_fusion.py
```
```
# pop and CO2 tasks
python evaluate_pop_fusion.py
```
Downstream evaluation dataset will come soon.

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
