# This is a Python implementation of the paper "EPCFQA: Explicit Physical Constraint Fusion Framework for Underwater Image Quality Assessment".

### Test
To test the model, first download the pre-trained weights from [BaiduDisk](https://pan.baidu.com/s/1q7Je2b3yK8An8-XSOhVdfA?pwd=0221).  
- For batch processing, run `predict_dir.py`.  
- For single image processing, run `predict_single.py`.  

### Train
To retrain the model:  
1. Download the pre-trained weights from [BaiduDisk](https://pan.baidu.com/s/1q7Je2b3yK8An8-XSOhVdfA?pwd=0221).  
2. Prepare your own dataset and run `train.py`.  

To reproduce the entire training process:  
1. Collect sufficient real-world images.  
2. Run `data.py` for data synthesis.  
3. Use the synthesized dataset to pretrain the model via `pretrain.py`.
4. Run `train.py`.  

### Acknowledgements
1. Underwater image synthesis coefficients are computed using the method from [hainh/sea-thru](https://github.com/hainh/sea-thru).  
2. We thank the authors of [DysenNet](https://ieeexplore.ieee.org/abstract/document/10852362) for providing the complete SOTA dataset.
