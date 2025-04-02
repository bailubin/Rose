import torch
import yaml
import os
import timm
import timm.optim.optim_factory as optim_factory
import argparse
import math
from functools import partial
import torchvision
from prefetch_generator import BackgroundGenerator
from dataset import *
from model import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=r'../data/graphs224code_re')
parser.add_argument('--graph_in_dim', type=int, default=524)
parser.add_argument('--graph_out_dim', type=int, default=128)

parser.add_argument('--lr', type=float, default=None)
parser.add_argument('--blr', type=float, default=1.5e-4) # 1.5
parser.add_argument('--batch_size', type=int, default=10) #64
parser.add_argument('--total_epoch',type=int,default=120)
parser.add_argument('--schedule', default=[90, 110], nargs='*', type=list,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--cos',type=bool,default=False)
parser.add_argument('--loss_alpha', type=int, default=3)
parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

parser.add_argument('--save_root_dir',type=str,default=r'./savedir')
parser.add_argument('--save_dir',type=str,default=r'rose-singapore')


# prepare folder and record
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.path.exists(args.save_root_dir):
    os.mkdir(args.save_root_dir)
save_dir=os.path.join(args.save_root_dir, args.save_dir)
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
#train records
record_path=os.path.join(save_dir,'loss_record.txt')


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.total_epoch))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch == milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


train_set=ImageOsmDataset(data_path=args.data_path)
print("Data size: ", len(train_set))
train_loader=torch_geometric.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, drop_last=True, pin_memory=True)

#image encoder
img_encoder=timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0, global_pool='')

# osm encoder
osm_encoder=OsmEncoder(osm_in_chans=args.graph_in_dim, osm_out_dim=args.graph_out_dim)

# fusion model
model = Rose(img_encoder, osm_encoder).to(device)
model.train()

# optimizer
if args.lr is None:  # only base_lr is specified
    args.lr = args.blr * args.batch_size / 256
param_groups = optim_factory.param_groups_weight_decay(model, args.weight_decay)
optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))

# train
for epoch in range(1, args.total_epoch+1):
    loss_step1=0
    loss_step2=0
    optimizer=adjust_learning_rate(optimizer, epoch, args)
    for step,data in enumerate(train_loader):
        model.zero_grad()
        osm, img=data[0], data[1]

        osm=osm.to(device)
        img=img.to(device)

        loss1, loss2, pred_imgs, mask_imgs, pred_osms, mask_osms=model(img, osm, img_mask_ratio=0.75, osm_mask_ratio=0.15)
        loss = loss1 + loss2
        loss.backward()
        optimizer.step()

        loss_step1 += loss1.item()
        loss_step2 += loss2.item()
        if step % 50 == 0:
            print(
                f"epoch [{epoch}/{args.total_epoch}]\t Step [{step}/{len(train_loader)}]\t Loss1: {loss_step1 / 50}\t Loss2: {loss_step2 / 50}\t")
            record_file = open(record_path, 'a')
            record_file.write(
                f"epoch [{epoch}/{args.total_epoch}]\t Step [{step}/{len(train_loader)}]\t Loss1: {loss_step1 / 50}\t Loss2: {loss_step2 / 50}\n")
            record_file.close()
            loss_step1=0
            loss_step2=0


    if epoch%5==0 or epoch==args.total_epoch:
        out_model_path=os.path.join(save_dir, "model_{}.tar".format(epoch))
        torch.save(model.state_dict(),out_model_path)
