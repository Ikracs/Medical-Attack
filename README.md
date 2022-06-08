# Improved Adaptive Square Attack
This is the implementation of "*Query-based Black-box Attack against Medical Image Segmentation Model*", which has been accpeted by **FGCS, 2022**.

### Abstract
With the extensive deployment of deep learning, the research on *adversarial example* receives more concern than ever before.
By modifying a small fraction of the original image, an adversary can lead a well-trained model to make a wrong prediction.
However, existing works about adversarial attack and defense mainly focus on image classification but pay little attention to more practical tasks like segmentation.
In this work, we propose a query-based black-box attack that could alter the classes of foreground pixels within a limited query budget.
The proposed method improves the Adaptive Square Attack by employing a more accurate gradient estimation of loss and replacing the fixed variance of adaptive distribution with a learnable one.
We also adopt a novel loss function proposed for attacking medical image segmentation models.
Experiments on a widely-used dataset and well-known models demonstrate the effectiveness and efficiency of the proposed method in attacking medical image segmentation models.

### Requirements
Our code is based on the following dependencies
- NumPy == 1.18.1
- Pandas == 1.1.5
- Pytorch == 1.4.0
- torchvision == 0.5.0
- PyMIC == 0.2.3

### Running
After fill the config file in `cfg/`, you can run the attack as follows
```python
python attack.py cfg/examples.cfg
```
You can also train your own models by [PyMIC](https://github.com/HiLab-git/PyMIC) and conduct an attack on it.
