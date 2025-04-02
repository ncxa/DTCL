# DTCL
This repo contains the Pytorch implementation of our paper:
**Dual Triplet Contrastive Loss Constraint for Hard Instance in Video Anomaly Detection**


## Requirements
<pre>
    * Python 3.7.4
    * PyTorch 1.7.1
    * torchvision 0.8.2
    * visdom 0.2.3
    * numpy 1.21.6
    * tqdm 4.67.1
</pre>


## Training

### Setup
**We use the extracted I3D features for UCF-Crime and XD-Violence datasets from the following works:**
> [**UCF-Crime 10-crop I3D features**](https://github.com/Roc-Ng/DeepMIL)
> 
> [**XD-Violence 5-crop I3D features**](https://roc-ng.github.io/XD-Violence/)


**Pretrained models can be downloaded in here:**
> [**best performance ckpt for UCF**](weight_model/ucf_best.pkl)
>
> [**best performance ckpt for XD**](weight_model/xd_best.pkl)



The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the download datasets above in `list/XD_Train.list` and `list/XD_Test.list`. 
- Feel free to change the hyperparameters in `option.py`


### Train and test the DTCL
After the setup, simply run the following command: 

start the visdom for visualizing the training phase

```
python -m visdom.server -p 2023
```
Traing and infer for XD dataset
```
python xd_main.py
python xd_infer.py
```
Traing and infer for UCFC dataset
```
python ucf_main.py
python ucf_infer.py
```