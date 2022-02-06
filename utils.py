import torch

import torch
import torch.nn.functional as F
from math import exp
import numpy as np

def ssim(img1, img2, window_size=11):
    def gaussian(window_size, sigma):
        gauss = torch.Tensor([exp(-(x-window_size//2)**2)/float(2*sigma**2) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(window_size, channel=1):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    img1, img2 = img1 / 255.0, img2 / 255.0
    window = create_window(window_size).type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, stride=1, groups=1)
    mu2 = F.conv2d(img2, window, padding=window_size//2, stride=1, groups=1)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, stride=1, groups=1) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, stride=1, groups=1) - mu2_sq
    sigma_12  = F.conv2d(img1*img2, window, padding=window_size//2, stride=1, groups=1) - mu1_mu2
    
    c1 = (0.01) **2
    c2 = (0.03) **2
    
    ssim_map = ((2*mu1_mu2 +c1)*(2*sigma_12 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
    
    return ssim_map.mean()

def p_selection(p_init, it, n_iters):
    if 10 < it <= 50:       p = p_init / 2
    elif 50 < it <= 200:    p = p_init / 4
    elif 200 < it <= 500:   p = p_init / 8
    elif 500 < it <= 800:   p = p_init / 16
    elif 800 < it <= 1000:  p = p_init / 32
    else:                   p = p_init
    return p


class UniformDistribution(object):
    def __init__(self, config, gt):
        self.n_ex = gt.shape[0]
        self.boundary = torch.tensor(gt.shape[2:])

    def sample(self, size):
        coord = [torch.randint(0, b - size, (self.n_ex, 1)) for b in self.boundary]
        return torch.cat(coord, dim=-1)

    def update(self, idx_fool, loss):
        pass


class AdaptiveDistribution(object):
    def __init__(self, config, gt):
        self.n_ex = gt.shape[0]
        self.boundary = torch.tensor(gt.shape[2:])

        self.attack    = config['attacking']['attack']
        
        self.K         = config['attacking']['k']
        self.mean_lr   = config['attacking']['mean_lr']
        self.std_lr    = config['attacking']['std_lr']
        self.min_std   = config['attacking']['min_std']

        self.init_dis_param(gt)

    def init_dis_param(self, gt):
        foreground = gt[:, 1, :, :]
        boundary  = self.boundary.type(torch.float32) + 1
        samples   = [(torch.nonzero(i) + 1e-5) / boundary for i in foreground]
        
        def inv_sigmoid(x):
            return torch.log(x / (1 - x))
        
        if self.attack == 'IASA':
            print('Initializing Adaptive distribution...')
            self.mean = torch.stack([inv_sigmoid(s).mean(dim=0) for s in samples])
            self.std  = torch.stack([inv_sigmoid(s).std(dim=0)  for s in samples])
            self.std = self.std.clamp(min=self.min_std)
        else:
            self.mean = torch.zeros((self.n_ex, 2))
            self.std  = torch.ones((self.n_ex, 2))

        self.m_grad = torch.zeros_like(self.mean)
        self.s_grad = torch.zeros_like(self.std)
        
        self.sample_num = 0

    def get_m_grad(self, samples):
        # the gradient of log prob with respect to mean
        return samples / self.std

    def get_s_grad(self, samples):
        # the gradient of log prob with respect to std
        return (samples ** 2 - 1) / self.std

    def mean_step(self, alpha, grad, idx_fool):
        self.mean[idx_fool] -= alpha * grad

    def std_step(self, alpha, grad, idx_fool):
        self.std[idx_fool] -= alpha * grad
        self.std = self.std.clamp(min=self.min_std)

    def sample(self, size):
        self.samples = torch.randn((self.n_ex, 2))
        samples = (self.samples * self.std + self.mean).sigmoid()
        coord   = (self.boundary - size) * samples
        return coord.type(torch.int32)
    
    def update(self, idx_fool, loss):
        loss = loss.reshape((-1, 1)).repeat((1, 2)).cpu()
        self.m_grad[idx_fool] += self.get_m_grad(self.samples)[idx_fool] * loss
        self.s_grad[idx_fool] += self.get_s_grad(self.samples)[idx_fool] * loss
        # import pdb; pdb.set_trace()
        
        self.sample_num += 1
        if self.sample_num % self.K == 0:
            m_grad = self.m_grad[idx_fool] / self.K
            s_grad = self.s_grad[idx_fool] / self.K
            self.mean_step(self.mean_lr, m_grad, idx_fool)
            if self.attack == 'IASA':
                self.std_step(self.std_lr, s_grad, idx_fool)
            
            self.sample_num = 0
            self.m_grad = torch.zeros_like(self.m_grad)
            self.s_grad = torch.zeros_like(self.s_grad)
