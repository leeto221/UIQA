This is a Python implementation of the paper "EPCFQA: Explicit Physical Constraint Fusion Framework for Underwater Image Quality Assessment".

If you want to test the code, please download the weight file in [BaiduDisk](https://pan.baidu.com/s/1q7Je2b3yK8An8-XSOhVdfA?pwd=0221) first.
For batch files, you should run predictability dir. py.
For a single image, you should run predict_stngle. py.

If you want to retrain the code, please download the pre training weight file in [BaiduDisk](https://pan.baidu.com/s/1q7Je2b3yK8An8-XSOhVdfA?pwd=0221) and use your own dataset for training. You should run train.exe.

If you want to redo the entire training process, you need to prepare a sufficient number of real-world datasets, then use data.Py for data synthesis, and proceed with training.

Acknowlegements

1.The coefficients for synthesizing underwater images are computed based on [hainh/sea-thru](https://github.com/hainh/sea-thru).
2.We sincerely thank the author of the [DysenNet](https://ieeexplore.ieee.org/abstract/document/10852362) paper and his team for providing us with the complete SOTA dataset.
