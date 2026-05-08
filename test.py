import argparse
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F

from rfmnet import RFMNet
from data import get_dataloader
import utils.metrics as Measure
from utils.utils import load_model_params


def test_model(test_loader, model):

    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            # with len(test_loader) as pbar:
            i=1
            for (datadict) in test_loader:
                image = datadict['camo_image'].cuda()
                gt = datadict['camo_gt'].numpy().astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)                     # 标准化处理,把数值范围控制到(0,1)

                # sal_fx = datadict['ref_sod_image'].cuda()
                sal_fx_list = [x.cuda().float() for x in datadict['ref_sod_image_list']]

                res, res_list = model(x=image, sod_x_list=sal_fx_list)
                i=i+1
                # print(mask_list)
                
                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)                        # 标准化处理,把数值范围控制到(0,1)

                WFM.step(pred=res*255, gt=gt*255)
                SM.step(pred=res*255, gt=gt*255)
                EM.step(pred=res*255, gt=gt*255)
                MAE.step(pred=res*255, gt=gt*255)

                pbar.update()
                
        sm = SM.get_results()['sm'].round(3)
        adpem = EM.get_results()['em']['adp'].round(3)
        wfm = WFM.get_results()['wfm'].round(3)
        mae = MAE.get_results()['mae'].round(3)

        return {'Sm':sm, 'adpE':adpem, 'wF':wfm, 'M':mae}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='r2cnet')
    parser.add_argument('--dim', type=int, default=64, help='dimension of our model')
    parser.add_argument('--testsize', type=int, default=512, help='testing image size')
    parser.add_argument('--shot', type=int, default=1)

    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    parser.add_argument('--data_root', type=str, default='./R2C7K/', help='the path to put dataset')  
    parser.add_argument('--save_root', type=str, default='./snapshot/', help='the path to save model params and log')

    opt = parser.parse_args()
    print(opt)

    # load model 
    ref_model = RFMNet(channel=opt.dim, imagenet_pretrained=False, shot=opt.shot, kernel_set=[64, 32, 16], step_set=[64, 32, 16], layers_set=[1, 1, 1]).cuda()
    ref_model = ref_model.to(torch.device("cuda:0"))
    params_path = os.path.join(opt.save_root, 'saved_model', '{}.pth'.format('rfmnet'))
    ref_model = load_model_params(ref_model, params_path)

    # load data
    test_loader = get_dataloader(opt.data_root, opt.shot, opt.testsize, opt.num_workers, mode='test', mode2 = 1)

    # processing
    scores = test_model(test_loader, ref_model)
    print(scores)
