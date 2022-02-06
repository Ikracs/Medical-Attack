import os
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms

from pymic.io.nifty_dataset import NiftyDataset
from pymic.net.net_dict_seg import SegNetDict
from pymic.net_run.agent_seg import SegmentationAgent
from pymic.loss.seg.util import get_soft_label
from pymic.io.image_read_write import save_array_as_rgb_image


class ForeGroundDiceLoss(torch.nn.Module):
    '''
    Different from the foreground dice, this loss will ignore 
    the score of background pixel upon foreground class.
    '''
    def __init__(self, params=None):
        super(ForeGroundDiceLoss, self).__init__()
    
    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        img_w   = loss_input_dict['image_weight']
        pix_w   = loss_input_dict['pixel_weight']
        cls_w   = loss_input_dict['class_weight']
        softmax = loss_input_dict['softmax']

        if(softmax): predict = torch.nn.Softmax(dim=1)(predict)        
        img_w = img_w[:, None, None, None]; pix_w = pix_w * img_w
        
        if predict.shape[1] > 2:
            raise NotImplementedError('Only Implemented For Binary Classification!')
        intersect = torch.sum((predict * soft_y * pix_w)[:, 1, :, :], dim=(1, -1))
        soft_y = torch.sum((soft_y  * pix_w)[:, 1, :, :], dim=(1, -1))
        fg_dice = (2 * intersect + 1e-5) / (intersect + soft_y + 1e-5)
        return fg_dice

def get_classwise_dice(soft_p, soft_y):
    y_vol = torch.sum(soft_y, dim=(2, -1))
    p_vol = torch.sum(soft_p, dim=(2, -1))
    intersect = torch.sum(soft_y * soft_p, dim=(2, -1))
    dice_score = (2.0 * intersect + 1e-5)/ (y_vol + p_vol + 1e-5)
    return dice_score


class SegModel(SegmentationAgent):
    def __init__(self, config):
        super().__init__(config, stage='test')
        device_ids = self.config['testing']['gpus']
        self.device = torch.device("cuda:{0:}".format(device_ids[0]))

        self.create_dataset()
        self.create_network()
        self.loss_calculater = ForeGroundDiceLoss()

        checkpoint_name = self.get_checkpoint_name()
        checkpoint = torch.load(checkpoint_name, map_location=self.device)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.to(self.device); self.net.eval()

    def create_network(self):
        if self.net is None:
            net_name = self.config['network']['net_type']
            if net_name in SegNetDict:
                self.net = SegNetDict[net_name](self.config['network'])
            else:
                raise ValueError("Undefined network {0:}".format(net_name))
        if self.tensor_type == 'float':
            self.net.float()
        else:
            self.net.double()
        param_number = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        print('parameter number:', param_number)

    def create_dataset(self):
        root_dir    = self.config['dataset']['root_dir']
        modal_num   = self.config['dataset']['modal_num']
        csv_file    = self.config['dataset']['raw_data']
        trans_names = self.config['dataset']['transforms']
        
        trans_param = self.config['dataset']
        trans_param['task'] = 'segmentation'
        
        self.transform_list = []
        for name in trans_names:
            if(name not in self.transform_dict):
                raise(ValueError("Undefined transform {0:}".format(name))) 
            one_trans = self.transform_dict[name](trans_param)
            self.transform_list.append(one_trans)
        data_trans = transforms.Compose(self.transform_list)
        
        test_set = NiftyDataset(root_dir=root_dir,
                                csv_file=csv_file,
                                modal_num=modal_num,
                                with_label=True,
                                transform=data_trans)

        bn_test = self.config['dataset'].get('test_batch_size', 1)
        self.test_loder = torch.utils.data.DataLoader(test_set,
            batch_size=bn_test, shuffle=False, num_workers=bn_test)
        # mean and std for img normalization
        chn_mean, chn_std = [], []
        for data in self.test_loder:
            inputs = self.convert_tensor_type(data['image']).to(self.device)
            chn_mean.append(inputs.mean(dim=(2, -1), keepdim=True))
            chn_std.append(inputs.std(dim=(2, -1), keepdim=True))
        
        self.chn_m = torch.cat(chn_mean)
        self.chn_s = torch.cat(chn_std)

    def infer_and_get_loss(self, imgs):
        class_num  = self.config['network']['class_num']
        
        loss_list, ASR = [], []
        for batch_id, data in enumerate(self.test_loder):
            nnum  = data['image'].shape[0]
            bsize = self.test_loder.batch_size
            start_id = batch_id * bsize; end_id = start_id + nnum
            # suppose the target model is trained on normalized images
            img = imgs[start_id: end_id]
            chn_m = self.chn_m[start_id: end_id]
            chn_s = self.chn_s[start_id: end_id]
            
            inputs = self.convert_tensor_type(img).to(self.device)
            labels = self.convert_tensor_type(data['label_prob']).to(self.device)
            
            outputs = self.net((inputs - chn_m) / chn_s)
            loss = self.get_loss_value(data, inputs, outputs, labels)
            
            outputs_argmax = torch.argmax(outputs, dim=1, keepdim=True)
            soft_out = get_soft_label(outputs_argmax, class_num, self.tensor_type)
            fail_num = (soft_out[:, 1] * labels[:, 1]).sum(dim=(1, -1))
            asr = 1 - fail_num / labels[:, 1].sum(dim=(1, -1))
            loss_list.append(loss); ASR.append(asr)
        return torch.cat(loss_list, dim=0), torch.cat(ASR, dim=0)

    def freeze_parameters(self):
        for param in self.net.parameters():
            param.requires_grad_(False)

    def get_img_and_gt(self):
        img_list, gt_list = [], []
        for data in self.test_loder:
            inputs = self.convert_tensor_type(data['image'])
            labels_prob = self.convert_tensor_type(data['label_prob'])
            img_list.append(inputs); gt_list.append(labels_prob)
        return torch.cat(img_list), torch.cat(gt_list)

    def save_perturbation(self, perbs):
        perb_dir     = self.config['attacking']['perb_dir']
        raw_data_csv = self.config['dataset']['raw_data']

        if not os.path.exists(perb_dir): os.makedirs(perb_dir)
        
        csv_item = pd.read_csv(raw_data_csv)
        for idx in range(perbs.shape[0]):
            name = csv_item.iloc[idx, 0].split('/')[-1]
            img_name = perb_dir + '/' + name

            perb = perbs[idx].squeeze(0).cpu().numpy().astype(np.uint8)
            save_array_as_rgb_image(perb, img_name)
    
    def save_img_and_prediction(self, imgs):
        adv_img_dir  = self.config['attacking']['adv_img_dir']
        raw_data_csv = self.config['dataset']['raw_data']
        output_dir   = self.config['testing']['output_dir']

        if not os.path.exists(adv_img_dir): os.makedirs(adv_img_dir)
        if not os.path.exists(output_dir): os.makedirs(output_dir)

        csv_item = pd.read_csv(raw_data_csv)
        for idx in range(imgs.shape[0]):
            name = csv_item.iloc[idx, 0].split('/')[-1]
            img_name = adv_img_dir + '/' + name
            res_name = output_dir + '/' + name

            img = imgs[idx].squeeze(0).cpu().numpy().astype(np.uint8)
            save_array_as_rgb_image(img, img_name)

            inputs = self.convert_tensor_type(imgs[idx: idx + 1]).to(self.device)
            chn_mean = self.chn_m[idx: idx + 1]
            chn_std  = self.chn_s[idx: idx + 1]
            inputs = (inputs - chn_mean) / chn_std

            output = torch.argmax(self.net(inputs)[0], dim=0) * 255
            output = output.cpu().numpy().astype(np.uint8)
            save_array_as_rgb_image(output, res_name)
