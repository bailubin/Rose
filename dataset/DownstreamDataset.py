from PIL import Image
from torch.utils.data import Dataset
import os
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import InterpolationMode
import torch
import numpy as np
from torchvision import transforms
import random
import torch_geometric

class DownstreamDataset(Dataset):
    def __init__(self, root, mode, task='lu', train_file='train100.txt'):
        super(DownstreamDataset, self).__init__()
        self.root=root
        self.mode=mode
        self.task=task

        if self.mode=='train':
            with open(os.path.join(root, train_file), 'r') as f:
                self.file_names=f.read().splitlines()
        elif self.mode=='val':
            with open(os.path.join(root, 'val.txt'), 'r') as f:
                self.file_names=f.read().splitlines()
        elif self.mode=='test':
            self.file_names=os.listdir(os.path.join(self.root, 'img'))
        else:
            print('mode wrong!')

    def __getitem__(self, item):
        file_name=self.file_names[item]
        img=Image.open(os.path.join(self.root,'img',file_name))

        if self.task=='lu':
            if self.mode=='train' or self.mode=='val':
                label = Image.open(os.path.join(self.root, 'label', file_name))
                label = torch.from_numpy(np.array(label)).long()
                label[label==9]=255
                img=self.random_aug(img)
                return img, label
            else:
                img=self.random_aug(img)
                img = F.resize(img, size=[224, 224])
                return img, file_name

        elif self.task=='pop':
            if self.mode=='train' or self.mode=='val':
                with open(os.path.join(self.root, 'pop', file_name[:-3]+'txt')) as f:
                    pop = float(f.readline())
                img=self.random_aug(img)
                return img, torch.tensor([pop]).float()
            else:
                img=self.random_aug(img)
                img = F.resize(img, size=[224, 224])
                return img, file_name

        elif self.task=='co2':
            if self.mode=='train' or self.mode=='val':
                with open(os.path.join(self.root, 'CO2', file_name[:-3]+'txt')) as f:
                    co2 = float(f.readline())
                img=self.random_aug(img)
                return img, torch.tensor([co2]).float()
            else:
                img=self.random_aug(img)
                img = F.resize(img, size=[224, 224])
                return img, file_name

    def __len__(self):
        return len(self.file_names)

    def random_aug(self, img):
        ColorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        RandomGrayscale = transforms.RandomGrayscale(p=0.2)
        Blur = transforms.GaussianBlur(kernel_size=3, sigma=[0.1, 2.0])
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.mode == 'train':
            # if random.random()>0.2:
            #     img=ColorJitter(img)
            # img=RandomGrayscale(img)
            # img = Blur(img)
            img = toTensor(img)
            img = normalize(img)

        else:
            img = toTensor(img)
            img = normalize(img)

        return img

    def classnames(self):
        # return ['water','green1','green2','green3','green4','farmland','undevelopment land','residential1','residential2',
        #         'residential3', 'residential4','commercial1', 'commercial2', 'commercial3','commercial4','commercial5',
        #          'institution', 'transportation', 'industrial']
        # return ['industrial area','paddy field','irrigated field','dry cropland','garden land','arbor forest',
        #         'shrub forest','park','natural meadow','artificial meadow','river','urban residential','lake','pond',
        #         'fish pond','snow','bareland','rural residential','stadium','square','road','overpass','railway station','airport']
        return ['water', 'green', 'farmland', 'undev', 'resdi', 'commercial', 'institution', 'ind', 'trans']
        # return ['water','green', 'farmland', 'undev', 'resdi', 'commercial', 'institution', 'ind', 'trans']#,'special']


class DownstreamFusionDataset(torch_geometric.data.Dataset):
    def __init__(self, root, mode, task='lu', train_file='osm-train100.txt', remove_ratio = 0.5):
        super(DownstreamFusionDataset, self).__init__()
        self.root=root
        self.mode=mode
        self.task=task
        self.remove_ratio = remove_ratio

        if self.mode=='train':
            with open(os.path.join(root, train_file), 'r') as f:
                self.file_names=f.read().splitlines()
        elif self.mode=='val':
            with open(os.path.join(root, 'val.txt'), 'r') as f:
                self.file_names=f.read().splitlines()
        elif self.mode=='test':
            self.file_names=os.listdir(os.path.join(self.root, 'img'))
        else:
            print('mode wrong!')

    def get(self, item):
        file_name=self.file_names[item]
        img=Image.open(os.path.join(self.root,'img',file_name))

        if self.task=='lu':
            if self.mode=='train' or self.mode=='val':
                label = Image.open(os.path.join(self.root, 'label', file_name))
                label = torch.from_numpy(np.array(label)).long()
                label[label==9]=255
                img=self.random_aug(img)

                node = torch.load(os.path.join(self.root, 'node', file_name[:-3]+'tensor'))
                edge = torch.load(os.path.join(self.root, 'edge', file_name[:-3]+'tensor'))
                edgew = torch.load(os.path.join(self.root, 'edge_w', file_name[:-3]+'tensor'))
                return img, label, torch_geometric.data.Data(x=node.float(), edge_index=edge.long(), edge_attr=edgew.float())
            else:
                img=self.random_aug(img)

                node = torch.load(os.path.join(self.root, 'node', file_name[:-3] + 'tensor'))
                edge = torch.load(os.path.join(self.root, 'edge', file_name[:-3] + 'tensor'))
                edgew = torch.load(os.path.join(self.root, 'edge_w', file_name[:-3] + 'tensor'))
                return img, torch_geometric.data.Data(x=node.float(), edge_index=edge.long(), edge_attr=edgew.float()), file_name

        if self.task=='lu-remove':
            if self.mode=='train' or self.mode=='val':
                label = Image.open(os.path.join(self.root, 'label', file_name))
                label = torch.from_numpy(np.array(label)).long()
                label[label==9]=255
                img=self.random_aug(img)

                node = torch.load(os.path.join(self.root, 'node', file_name[:-3]+'tensor'))
                edge = torch.load(os.path.join(self.root, 'edge', file_name[:-3]+'tensor'))
                edgew = torch.load(os.path.join(self.root, 'edge_w', file_name[:-3]+'tensor'))
                graph = torch_geometric.data.Data(x=node.float(), edge_index=edge.long(), edge_attr=edgew.float())
                graph = self.random_node_remove(graph, remove_ratio = self.remove_ratio)
                return img, label, graph
            else:
                img=self.random_aug(img)

                node = torch.load(os.path.join(self.root, 'node', file_name[:-3] + 'tensor'))
                edge = torch.load(os.path.join(self.root, 'edge', file_name[:-3] + 'tensor'))
                edgew = torch.load(os.path.join(self.root, 'edge_w', file_name[:-3] + 'tensor'))
                graph = torch_geometric.data.Data(x=node.float(), edge_index=edge.long(), edge_attr=edgew.float())
                graph = self.random_node_remove(graph, remove_ratio=self.remove_ratio)
                return img, graph, file_name

        elif self.task=='pop':
            if self.mode=='train' or self.mode=='val':
                with open(os.path.join(self.root, 'pop', file_name[:-3]+'txt')) as f:
                    pop = float(f.readline())
                img=self.random_aug(img)

                node = torch.load(os.path.join(self.root, 'node', file_name[:-3] + 'tensor'))
                edge = torch.load(os.path.join(self.root, 'edge', file_name[:-3] + 'tensor'))
                edgew = torch.load(os.path.join(self.root, 'edge_w', file_name[:-3] + 'tensor'))
                return img, torch.tensor([pop]).float(), torch_geometric.data.Data(x=node.float(), edge_index=edge.long(), edge_attr=edgew.float())
            else:
                img=self.random_aug(img)

                node = torch.load(os.path.join(self.root, 'node', file_name[:-3] + 'tensor'))
                edge = torch.load(os.path.join(self.root, 'edge', file_name[:-3] + 'tensor'))
                edgew = torch.load(os.path.join(self.root, 'edge_w', file_name[:-3] + 'tensor'))
                return img, torch_geometric.data.Data(x=node.float(), edge_index=edge.long(), edge_attr=edgew.float()), file_name

        elif self.task=='co2':
            if self.mode=='train' or self.mode=='val':
                with open(os.path.join(self.root, 'CO2', file_name[:-3]+'txt')) as f:
                    co = float(f.readline())
                img=self.random_aug(img)

                node = torch.load(os.path.join(self.root, 'node', file_name[:-3] + 'tensor'))
                edge = torch.load(os.path.join(self.root, 'edge', file_name[:-3] + 'tensor'))
                edgew = torch.load(os.path.join(self.root, 'edge_w', file_name[:-3] + 'tensor'))
                return img, torch.tensor([co]).float(), torch_geometric.data.Data(x=node.float(), edge_index=edge.long(), edge_attr=edgew.float())
            else:
                img=self.random_aug(img)

                node = torch.load(os.path.join(self.root, 'node', file_name[:-3] + 'tensor'))
                edge = torch.load(os.path.join(self.root, 'edge', file_name[:-3] + 'tensor'))
                edgew = torch.load(os.path.join(self.root, 'edge_w', file_name[:-3] + 'tensor'))
                return img, torch_geometric.data.Data(x=node.float(), edge_index=edge.long(), edge_attr=edgew.float()), file_name

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

    def len(self):
        return len(self.file_names)

    def random_aug(self, img):
        ColorJitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)
        RandomGrayscale = transforms.RandomGrayscale(p=0.2)
        Blur = transforms.GaussianBlur(kernel_size=3, sigma=[0.1, 2.0])
        toTensor = transforms.ToTensor()
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if self.mode == 'train':
            # if random.random()>0.2:
            #     img=ColorJitter(img)
            # img=RandomGrayscale(img)
            # img = Blur(img)
            img = toTensor(img)
            img = normalize(img)

        else:
            img = toTensor(img)
            img = normalize(img)

        return img

    def classnames(self):
        return ['water','green', 'farmland', 'undev', 'resdi', 'commercial', 'institution', 'ind', 'trans']
