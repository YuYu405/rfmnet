import argparse
import datetime
import os

from tqdm import tqdm
import torch
import torch.nn.functional as F

from data import get_dataloader
from rfmnet import RFMNet


class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def poly_lr(optimizer, current_epoch, max_epoch, initial_lr, power):
    lr = initial_lr * (1 - current_epoch / max_epoch) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def smart_optimizer(model, name='Adam', lr=0.001, momentum=0.9, decay=1e-5):
    params = filter(lambda p: p.requires_grad, model.parameters())
    if name == 'Adam':
        return torch.optim.Adam(params, lr=lr, betas=(momentum, 0.999), weight_decay=decay)
    if name == 'AdamW':
        return torch.optim.AdamW(params, lr=lr, betas=(momentum, 0.999), weight_decay=decay)
    if name == 'RMSProp':
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=decay)
    if name == 'SGD':
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=decay, nesterov=True)
    raise NotImplementedError(f'Optimizer {name} not implemented.')


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def load_backbone_weights(model):
    model_dict = model.state_dict()

    cod_path = os.path.join('./snapshot/', 'base', 'swins_cod_base_45.pth')
    cod_state = torch.load(cod_path)
    model_dict.update({k: v for k, v in cod_state.items() if k in model_dict and 'swins' in k})

    sod_path = os.path.join('./snapshot/', 'base', 'swins_base_sod_45.pth')
    sod_state = torch.load(sod_path)
    model_dict.update({k: v for k, v in sod_state.items() if k in model_dict and 'swinssod' in k})

    model.load_state_dict(model_dict, strict=False)


def train(train_loader, model, optimizer, log_path, opt):
    model.train()
    total_step = len(train_loader)

    for epoch in range(1, opt.epoch + 1):
        loss_record = AvgMeter()
        loss1_record = AvgMeter()
        loss2_record = AvgMeter()
        loss3_record = AvgMeter()
        loss_res_record = AvgMeter()

        poly_lr(
            optimizer=optimizer,
            current_epoch=epoch,
            max_epoch=opt.epoch,
            initial_lr=opt.lr_0,
            power=opt.power,
        )

        for i, data_dict in enumerate(tqdm(train_loader, total=total_step), start=1):
            image = data_dict['camo_image'].cuda().float()
            gt = data_dict['camo_gt'].cuda().float()
            sal_fx_list = [x.cuda().float() for x in data_dict['ref_sod_image_list']]

            res, res_list = model(x=image, sod_x_list=sal_fx_list)

            res_loss1 = structure_loss(res_list[0], gt)
            res_loss2 = structure_loss(res_list[1], gt)
            res_loss3 = structure_loss(res_list[2], gt)
            loss_res_list = (4 * res_loss1 + 3 * res_loss2 + 2 * res_loss3).mean()
            loss_res = structure_loss(res, gt)
            loss = (7 * loss_res + loss_res_list).mean()

            optimizer.zero_grad()
            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            loss_record.update(loss.item(), opt.batchsize)
            loss1_record.update(res_loss1.item(), opt.batchsize)
            loss2_record.update(res_loss2.item(), opt.batchsize)
            loss3_record.update(res_loss3.item(), opt.batchsize)
            loss_res_record.update(loss_res.item(), opt.batchsize)

            if i % 30 == 0 or i == total_step:
                log = '[%3d], [%6d], [%.10f], [%.5f], [%.5f], [%.5f], [%.5f], [%.5f]' % (
                    epoch,
                    i,
                    optimizer.param_groups[0]['lr'],
                    loss_record.avg,
                    loss1_record.avg,
                    loss2_record.avg,
                    loss3_record.avg,
                    loss_res_record.avg,
                )
                print(
                    '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}'.format(
                        datetime.datetime.now(), epoch, opt.epoch, i, total_step, loss.item()
                    )
                )
                open(log_path, 'a').write(log + '\n')

        save_path = os.path.join(opt.ckpt_path, opt.exp_name, '')
        check_mkdir(save_path)
        if epoch < 400:
            if epoch % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, save_path + 'rfmnet_%d.pth' % epoch)
        elif epoch % 25 == 0:
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, save_path + 'rfmnet_%d.pth' % epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=500, help='epoch number')
    parser.add_argument('--lr_0', type=float, default=1.5e-4, help='learning rate')
    parser.add_argument('--decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--power', type=float, default=0.9, help='poly lr power')
    parser.add_argument('--optimizer_name', type=str, default='Adam', help='optimizer')
    parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=512, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--dim', type=int, default=64, help='dimension of our model')
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--data_root', type=str, default='./R2C7K/', help='the path to put dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='the number of workers in dataloader')
    parser.add_argument('--ckpt_path', type=str, default='./ckpt', help='ckpt path')
    parser.add_argument('--exp_name', type=str, default='rfmnet', help='model_name')
    opt = parser.parse_args()
    print(opt)

    model = RFMNet(
        channel=opt.dim,
        imagenet_pretrained=False,
        shot=opt.shot,
        kernel_set=[64, 32, 16],
        step_set=[64, 32, 16],
        layers_set=[1, 1, 1],
    )
    load_backbone_weights(model)

    for name, param in model.named_parameters():
        if 'swins' in name or 'swinssod' in name:
            param.requires_grad = False

    model = model.to(device=torch.device('cuda:0'))
    optimizer = smart_optimizer(model, opt.optimizer_name, opt.lr_0, momentum=0.9, decay=opt.decay)

    check_mkdir(opt.ckpt_path)
    check_mkdir(os.path.join(opt.ckpt_path, opt.exp_name))
    log_path = os.path.join(opt.ckpt_path, opt.exp_name, str(datetime.datetime.now()).replace(':', '-') + '.txt')

    train_loader = get_dataloader(
        opt.data_root,
        opt.shot,
        opt.trainsize,
        batchsize=opt.batchsize,
        num_workers=opt.num_workers,
        mode='train',
    )
    train(train_loader, model, optimizer, log_path, opt)
