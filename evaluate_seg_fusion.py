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
import torch_geometric
from torchvision import transforms
import yaml
import timm

from model import *
import dataset
from .utils import count_params, AverageMeter, intersectionAndUnion, init_log, F1Score
from datetime import datetime
import pandas as pd
from collections import defaultdict


parser = argparse.ArgumentParser(description='Semantic Segmentation Eval')
parser.add_argument('--config', type=str, default='seg-vit.yaml')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


def evaluate(model, loader, cfg):
    model.eval()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    confusion_matrix = np.zeros((cfg['nclass'], cfg['nclass']))

    with torch.no_grad():
        for img, mask, osms in loader:
            img = img.cuda()
            osms = osms.cuda()

            pred = model(img, osms)
            pred=pred.argmax(dim=1)

            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)
            confusion_matrix += F1Score(pred.cpu().numpy(), mask.numpy(), cfg['nclass'])

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIOU = np.mean(iou_class) * 100.0
    precision_cls = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    recall_cls = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)
    F1_cl = precision_cls * recall_cls * 2 / (precision_cls + recall_cls)
    mean_f1 = np.nanmean(F1_cl)

    return mIOU, iou_class, mean_f1


def main():
    args = parser.parse_args()
    model_name='finetune'
    mode='fusion'
    ckpt_fp = r'./savedir/rose-singapore/model_120.tar'
    train_file = 'osm-train-all.txt'
    save_path=os.path.join(os.path.dirname(ckpt_fp), 'segment-eval-fusion120')

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)['seg_sgp']

    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    logger.info('{}\n'.format(pprint.pformat(cfg)))
    os.makedirs(save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    # for osm_img_fusion model
    img_encoder = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0, global_pool='')
    osm_encoder = OsmEncoder(osm_in_chans=524, osm_out_dim=128)
    fusion_model = Rose(img_encoder, osm_encoder)
    fusion_model.load_state_dict(
        torch.load(ckpt_fp, map_location=torch.device('cpu')))

    decoder_cfg = cfg['decoder']
    decoder_cfg['n_cls'] = cfg['nclass']
    decoder = create_decoder(768,
                             fusion_model.img_encoder.patch_embed.patch_size[0],
                             decoder_cfg)
    model = MyFusionSegmenter(fusion_model, decoder, n_cls=cfg['nclass'])


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

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])


    trainset = dataset.DownstreamFusionDataset(cfg['data_root'], 'train', task='lu', train_file=train_file)
    valset = dataset.DownstreamFusionDataset(cfg['data_root'], 'val', task='lu', train_file=train_file)
    trainloader = torch_geometric.data.DataLoader(trainset, batch_size=cfg['batch_size'],
                             pin_memory=True, num_workers=4, drop_last=True, shuffle=True)
    valloader = torch_geometric.data.DataLoader(valset, batch_size=8, pin_memory=True, num_workers=4,
                           drop_last=False)

    cls_names = trainset.classnames()
    csv_name = "{}_{}_{}_{}.csv".format(train_file[:-4], mode,
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


        for i, (img, mask, osms) in enumerate(trainloader):

            img, mask, osms = img.cuda(), mask.cuda().long(), osms.cuda()
            # print(torch.unique(mask),mask.shape)

            pred = model(img, osms)

            loss = criterion(pred, mask)

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

        mIOU, iou_class, f1 = evaluate(model, valloader, cfg)

        for i, iou in enumerate(iou_class):
            logger.info(" * class [{}] IoU {:.2f}".format(cls_names[i], iou * 100))
            dict_iou[cls_names[i]].append(iou * 100)
        logger.info('***** Evaluation ***** >>>> meanIOU: {:.2f}, F1: {:.2f}\n'.format(mIOU, f1))
        dict_iou['epoch'].append(epoch)
        dict_iou['mIoU'].append(mIOU)
        dict_iou['f1'].append(f1)
        csv_iou = pd.DataFrame(dict_iou)
        csv_iou.to_csv(csv_path, index=None)

        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(save_path, '%s_%.2f.pth' % (model_name, previous_best)))
            previous_best = mIOU
            torch.save(model.state_dict(),
                       os.path.join(save_path, '%s_%.2f.pth' % (model_name, mIOU)))


if __name__ == '__main__':
    main()
