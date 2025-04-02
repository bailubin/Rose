import os
from PIL import Image
import numpy as np
from torch_geometric.data import Dataset, Data
import torch
from torchvision import transforms
import random
from torchvision.transforms import InterpolationMode


class ImageOsmDataset(Dataset):
    def __init__(self, data_path, mode='train', remove_ratio=None):
        super(ImageOsmDataset, self).__init__()

        self.img_path = os.path.join(data_path, 'img')
        self.node_path = os.path.join(data_path, 'node')
        self.edge_path = os.path.join(data_path, 'edge')
        self.edgew_path = os.path.join(data_path, 'edge_w')
        self.mode = mode

        if mode == 'train':
            all_files=os.listdir(self.node_path)
            self.train_files = []
            for f in all_files:
                if 'singapore' in f:
                    self.train_files.append(f)
            # self.train_files=all_files
        elif mode == 'test':
            pass

        if remove_ratio is not None:
            self.remove_ratio = remove_ratio
            self.remove = True
        else:
            self.remove = False

    def get(self, index):
        file_name = self.train_files[index]
        img = Image.open(os.path.join(self.img_path, file_name[:-6] + 'png'))
        img = self.random_aug(img)

        node = torch.load(os.path.join(self.node_path, file_name))
        edge = torch.load(os.path.join(self.edge_path, file_name))
        edgew = torch.load(os.path.join(self.edgew_path, file_name))
        data = Data(x=node.float(), edge_index=edge.long(), edge_attr=edgew.float())

        if self.remove:
            data = self.random_node_remove(data, self.remove_ratio)

        return  data, img

    def random_node_remove(self, data, remove_ratio):
        num_nodes = data.x.shape[0]
        num_nodes_to_remove = int(num_nodes * remove_ratio)
        if num_nodes_to_remove == 0:
            return data

        nodes_to_remove = torch.randperm(num_nodes)[:num_nodes_to_remove]

        # 创建一个布尔掩码来保留未删除的节点
        node_mask = torch.ones(num_nodes, dtype=torch.bool)
        node_mask[nodes_to_remove] = False

        # 删除与这些节点相关的边
        edge_index = data.edge_index
        edge_mask = node_mask[edge_index[0]] & node_mask[edge_index[1]]
        edge_index = edge_index[:, edge_mask]

        # 更新节点索引：为剩余的节点重新分配连续的索引
        remaining_nodes = torch.arange(num_nodes)[node_mask]
        new_index_mapping = torch.full((num_nodes,), -1, dtype=torch.long)
        new_index_mapping[remaining_nodes] = torch.arange(remaining_nodes.size(0))

        # 更新数据对象，保留未被删除的节点和边
        data.edge_index = new_index_mapping[edge_index]
        data.edge_attr = data.edge_attr[edge_mask]
        data.x = data.x[node_mask]

        return data

    def random_aug(self, img):
        RandomResizeCrop = transforms.RandomResizedCrop(224, scale=(0.67, 1.),
                                                        ratio=(3./4., 4./3.), interpolation=InterpolationMode.BICUBIC)
        ColorJitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        RandomGrayscale = transforms.RandomGrayscale(p=0.2)
        RandomHorizontalFlip = transforms.RandomHorizontalFlip()
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.mode == 'train':
            if random.random() > 0.5:
                img=RandomResizeCrop(img)
            if random.random()>0.5:
                img=ColorJitter(img)
            img=RandomGrayscale(img)
            img=RandomHorizontalFlip(img)
            img = toTensor(img)
            img = normalize(img)

        else:
            img = toTensor(img)
            img = normalize(img)

        return img

    def len(self):
        return len(self.train_files)