import sys
import pdb
import math
import time
import torch
import numpy as np

from model import SegModel
from utils import p_selection
from utils import UniformDistribution, AdaptiveDistribution
from pymic.util.parse_config import parse_config

def square_attack(model, config):
    attack       = config['attacking']['attack']
    
    eps          = config['attacking']['epsilon']
    p_init       = config['attacking']['p_init']
    query_budget = config['attacking']['query_budget']

    print_freq   = config['attacking']['print_freq']
    visualize    = config['attacking']['visualize']
    
    min_val, max_val = 0, 255

    print('Evaluating on raw dataset...')
    img, gt = model.get_img_and_gt()
    assert(img.min() >= min_val and img.max() <= max_val)
    
    _, avg_dice = model.infer_and_get_loss(img)
    print('Avg. Dice coff in Raw Dataset: {:.3f}'.format(avg_dice.mean()))
    print('Avg. Foreground Dice coff in Raw Dataset: {:.2f}'.format(avg_dice[:, 1].mean()))
    
    if attack in ['ASA', 'IASA']:
        print('Creating Adaptive distribution...')
        D = AdaptiveDistribution(config, gt)
    else:   # attack is 'SA'
        D = UniformDistribution(config, gt)
    
    n, c, h, w = img.shape
    n_features = c * h * w
    
    init_delta = torch.round(torch.rand_like(img)) * (2 * eps) - eps
    perturbed_best = torch.clamp(img + init_delta, min_val, max_val)
    prediction = model.infer_and_get_loss(perturbed_best)
    loss_best, f_dice_best = prediction[0], prediction[1][:, 1]
    
    start_time = time.time()
    for i_iter in range(query_budget):
        idx_fool = loss_best > 0
        if idx_fool.sum() < 1: break
        
        deltas = (perturbed_best - img)[loss_best > 0]

        p = p_selection(p_init, i_iter, query_budget)
        s = int(round(math.sqrt(p * n_features / c)))
        s = min(max(s, 1), h - 1)  # at least c x 1 x 1 window is taken and at most c x h-1 x h-1

        n = torch.sum(idx_fool)
        ul_corner = D.sample(s)[idx_fool].unsqueeze(1)
        bidx = torch.arange(n).reshape((-1, 1, 1, 1))   # broadcast to the size of N * C * S * S
        cidx = torch.arange(c).reshape((1, -1, 1, 1)).repeat((n, 1, 1, 1))  # broadcast to the size of N * C * S * S
        hidx = torch.arange(s).reshape((1, 1, -1, 1)).repeat((n, c, 1, s))
        hidx += ul_corner[:, :, 0: 1].unsqueeze(-1).repeat((1, 1, s, s))
        widx = torch.arange(s).reshape((1, 1, 1, -1)).repeat((n ,c, s, 1))
        widx += ul_corner[:, :, 1:].unsqueeze(-1).repeat((1, 1, s, s))

        deltas[bidx, cidx, hidx, widx] = torch.round(torch.rand((n, c, 1, 1))) * (2 * eps) - eps
        perturbed = torch.clamp(img[idx_fool] + deltas, min_val, max_val)
        prediction = model.infer_and_get_loss(perturbed)
        loss, f_dice = prediction[0], prediction[1][:, 1]
        D.update(idx_fool, loss)

        better_idx = loss < loss_best[idx_fool]
        loss_best[idx_fool] = loss * better_idx + loss_best[idx_fool] * (~better_idx)
        f_dice_best[idx_fool] = f_dice * better_idx + f_dice_best[idx_fool] * (~better_idx)
        
        better_idx = better_idx.reshape((n, 1, 1, 1)).cpu()
        perturbed_best[idx_fool] = perturbed * better_idx + perturbed_best[idx_fool] * (~better_idx)
        
        if i_iter % print_freq == 0:
            print('[Iter {:0>4d}] Avg. Loss: {:.2f}, Avg. Foreground Dice: {:.2f}'.\
                format(i_iter, loss_best.mean(), f_dice_best.mean()))
    
    print('Attack end with Avg. Loss: {:.2f}, Avg. Foreground Dice: {:.2f}'.\
        format(loss_best.mean(), f_dice_best.mean()))
    
    elapsed = time.time() - start_time
    if visualize:
        print('Saving perturbed img and predictions ...')
        model.save_img_and_prediction(perturbed_best)
        deltas_best = perturbed_best - img + eps
        deltas_best[deltas_best > 0] = 255
        print('Saving perturbations ...')
        model.save_perturbation(deltas_best)
    hours = elapsed // 3600; minutes = (elapsed % 3600) // 60
    print('Elapsed Time: {:.0f} h {:.0f}m'.format(hours, minutes))

if __name__ == '__main__':
    config = parse_config(str(sys.argv[1]))

    assert(config['dataset']['task_type'] == 'seg')
    victim_model = SegModel(config)
    
    with torch.no_grad():
        square_attack(victim_model, config)
