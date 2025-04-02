import argparse
from itertools import cycle
import logging
import os
import pprint

import torch
import numpy as np
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml
import timm
import sklearn.metrics as metrics

from model import *
import dataset
from .utils import count_params, AverageMeter, intersectionAndUnion, init_log, F1Score
from datetime import datetime
import pandas as pd
from collections import defaultdict
import torch_geometric


parser = argparse.ArgumentParser(description='POP Eval')
parser.add_argument('--config', type=str, default='seg-vit.yaml')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(model, loader):
    model.eval()
    yps = []
    yts = []
    for pkg in loader:
        img, y, osms = pkg[0], pkg[1], pkg[2]
        img, osms = img.cuda(), osms.cuda()

        y_estimated = model(img, osms).cpu().detach().flatten().numpy()
        y = y.flatten().numpy()

        yps.append(y_estimated)
        yts.append(y)
    yps = np.concatenate(yps, axis=0)
    yts = np.concatenate(yts, axis=0)
    mae=metrics.mean_absolute_error(yts,yps)
    mse=metrics.mean_squared_error(yts,yps,squared=False)
    r2 = metrics.r2_score(yts, yps)
    return mae, mse, r2


def main():
    args = parser.parse_args()
    model_name='finetune'
    ckpt_fp=r'./savedir/ImgOsmMae5-remove16-singapore/model_120.tar' # dropnode mask5 seloss
    train_file = 'osm-train-all.txt'
    save_path=os.path.join(os.path.dirname(ckpt_fp), 'co2-eval-fusion')

    cfg = yaml.load(open(args.config, "r"), Loader= yaml.Loader)['sgp_co2']

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    os.makedirs(save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # for osm_img_fusion model
    img_encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0, global_pool='')
    osm_encoder = OsmEncoder(osm_in_chans=524, osm_out_dim=128)
    fusion_model = Rose(img_encoder, osm_encoder)
    fusion_model.load_state_dict(
        torch.load(ckpt_fp, map_location=torch.device('cpu')))

    model = PopFusionLinear(fusion_model, cross_dim=768)


    if model_name == 'linear':
        # for linear probe
        optimizer = Adam([{'params': model.decoder.parameters(), 'lr': cfg['linear']['lr']}], lr=cfg['linear']['lr'], weight_decay=0.01)
        for name,param in model.named_parameters():
            print(name)
            if 'decoder' not in name:
                param.requires_grad=False
    else:
        # for finetune
        optimizer = Adam([{'params': [param for name, param in model.named_parameters() if 'decoder' not in name], 'lr': cfg['finetune']['lr']},
                         {'params': [param for name, param in model.named_parameters() if 'decoder' in name],
                          'lr': cfg['finetune']['lr'] * cfg['finetune']['lr_multi']}],lr=cfg['finetune']['lr'], weight_decay=0.01)

    logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    model.cuda()

    trainset=dataset.DownstreamFusionDataset(cfg['data_root'], 'train', task=cfg['dataset'], train_file=train_file)
    valset = dataset.DownstreamFusionDataset(cfg['data_root'], 'val' , task=cfg['dataset'], train_file=train_file)

    trainloader = torch_geometric.data.DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=4, drop_last=True, shuffle=True)
    valloader = torch_geometric.data.DataLoader(valset, batch_size=16, pin_memory=True, num_workers=4,
                           drop_last=False)

    csv_name = "{}_{}_{}.csv".format(train_file[:-4],
                                           model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    csv_path = os.path.join(save_path, csv_name)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    dict_iou = defaultdict(list)

    for epoch in range(cfg['epochs']):

        logger.info('===========> Epoch: {:}, LR: {:.4f}, Previous best: {:.2f}'.format(
            epoch, optimizer.param_groups[0]['lr'], previous_best))

        model.train()
        total_loss = 0.0


        for i, (img, pop, osms) in enumerate(trainloader):

            img, pop, osms = img.cuda(), pop.cuda(), osms.cuda()
            # print(torch.unique(mask),mask.shape)

            pred = model(img, osms)
            loss = F.mse_loss(pred, pop)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            iters += 1
            if model_name == 'finetune':
                lr = cfg['finetune']['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * cfg['finetune']['lr_multi']
            else:
                lr = cfg['linear']['lr'] * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr

            if (i % (max(2, len(trainloader) // 8)) == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss / (i+1)))

        mae, mse, r2 = evaluate(model, valloader)

        logger.info('***** Evaluation ***** >>>> mae: {:.2f}, mse: {:.2f}, r2: {:.2f}\n'.format(mae, mse, r2))
        dict_iou['epoch'].append(epoch)
        dict_iou['mae'].append(mae)
        dict_iou['mse'].append(mse)
        dict_iou['r2'].append(r2)
        csv_iou = pd.DataFrame(dict_iou)
        csv_iou.to_csv(csv_path, index=None)

        if r2 > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(save_path, '%s_%.2f.pth' % (model_name, previous_best)))
            previous_best = r2
            torch.save(model.state_dict(),
                       os.path.join(save_path, '%s_%.2f.pth' % (model_name, r2)))


if __name__ == '__main__':
    main()
