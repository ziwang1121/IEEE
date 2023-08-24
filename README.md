# Interact, Embed, and EnlargE (IEEE): Boosting Modality-specific Representations for Multi-Modal Person Re-identification

We provide the codes for reproducing result of our paper **Interact, Embed, and EnlargE (IEEE): Boosting Modality-specific Representations for Multi-Modal Person Re-identification**.



## Installation

1. Basic environments: `python3.6`, `pytorch1.8.0`, `cuda11.1`.

2. Our codes structure is based on `Torchreid`. (More details can be found in link: https://github.com/KaiyangZhou/deep-person-reid , you can download the packages according to `Torchreid` requirements.)

```python
# create environment
cd AAAI2022_IEEE/
conda create --name ieeeReid python=3.6
conda activate ieeeReid

# install dependencies
# make sure `which python` and `which pip` point to the correct path
pip install -r requirements.txt

# install torch and torchvision (select the proper cuda version to suit your machine)
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# install torchreid (don't need to re-build it if you modify the source code)
python setup.py develop
```



## Get start

1. You can use the setting in `im_r50_softmax_256x128_amsgrad_RGBNT_ieee_part_margin.yaml` to get the results of full IEEE.

   ```python
   python ./scripts/mainMultiModal.py --config-file ./configs/RGBNT_ieee_part_margin.yaml --seed 40
   ```

## Details

1. The details of our  **Cross-modal Interacting Module (CIM)** and **Relation-based Embedding Module (REM)** can be found in `.\torchreid\models\ieee3modalPart.py`. The design of  **Multi-modal Margin Loss(3M loss)** can be found in `.\torchreid\losses\multi_modal_margin_loss_new.py`.

2. Ablation study settings.

   You can control these two modules and the loss by change the corresponding codes.

   1) **Cross-modal Interacting Module (CIM)** and **Relation-based Embedding Module (REM)**
   
   ```python
   # change the code in .\torchreid\models\ieee3modalPart.py
   
   class IEEE3modalPart(nn.Module):
       def __init__(···
       ):
           modal_number = 3
           fc_dims = [128]
           pooling_dims = 768
           super(IEEE3modalPart, self).__init__()
           self.loss = loss
           self.parts = 6
           
           self.backbone = nn.ModuleList(···
           )

   		  # using Cross-modal Interacting Module (CIM)
           self.interaction = True
           # using channel attention in CIM
           self.attention = True
           
           # using Relation-based Embedding Module (REM)
           self.using_REM = True
           
           ···
   ```
   
   2) **Multi-modal Margin Loss(3M loss)**
   
   ```python
   # change the code in .\configs\your_config_file.yaml
   
   # using Multi-modal Margin Loss(3M loss), you can change the margin by modify the parameter of "ieee_margin".
   ···
   loss:
     name: 'margin'
     softmax:
       label_smooth: True
     ieee_margin: 1
     weight_m: 1.0
     weight_x: 1.0
   ···
   
   # using only CE loss
   ···
   loss:
     name: 'softmax'
     softmax:
       label_smooth: True
     weight_x: 1.0
   ···
   ```
   
 ## Dataset
   #### RGBNT201:
      Google Drive Link: https://drive.google.com/drive/folders/1EscBadX-wMAT56_It5lXY-S3-b5nK1wH?usp=sharing
   #### RGBNT201 cross-modal dataset:
      Google Drive Link: https://drive.google.com/file/d/1PQ78O0Pxi4pGEfRN2NkQ0ItF6EIoBxjq/view?usp=sharing
   Please contact with Zi Wang (email address: ziwang1121@foxmail.com).
 
