# medical_attack
This is the implementation of "Query-based Black-box Attack against Medical Image Segmentation Model".

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
