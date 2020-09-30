#!/usr/bin/env python
import os
import numpy as np
import argparse
import random
import pandas as pd
import PIL.Image as Image
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models


parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 tile classifier training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--kpred', type=str, default='TUM', help='kpred filter')
parser.add_argument('--target-k', type=str, default=None, help='target column key')
parser.add_argument('--target-v', type=bool, default=None, help='target column value')
parser.add_argument('--batch_size', type=int, default=512, help='mini-batch size (default: 512)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--model', type=str, default='', help='path to pretrained model')
parser.add_argument('--test_every', default=10, type=int, help='test on val every (default: 10)')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--k', default=1, type=int, help='top k tiles are assumed to be of the same class as the slide (default: 1, standard MIL)')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()

    #cnn
    model = models.resnet34(True) # pretrained imagenet
    #pytorch_set_param_requires_grad(model, False) # for feature extraction
    model.fc = nn.Linear(model.fc.in_features, 2)
    ch = torch.load(args.model)
    model.load_state_dict(ch['state_dict'])
    model = model.cuda()
    cudnn.benchmark = True

    #normalization
    #normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    # imagenet_stats
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    trans = transforms.Compose([transforms.ToTensor(), normalize])

    #load data
    train_dset = MILdataset(args.train_lib, args.target_k, args.target_v, trans, args.kpred)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)
    if args.val_lib:
        val_dset = MILdataset(args.val_lib, args.target_k, args.target_v, trans, args.kpred)
        val_loader = torch.utils.data.DataLoader(
            val_dset,
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=False)

    save_inferences(model, val_loader, val_dset, 'val_inferences.tsv')
    save_inferences(model, train_loader, train_dset, 'train_inferences.tsv')

    

def save_inferences(model, loader, dset, path):
    dset.setmode(1)
    inf_probs = inference(loader, model)
    topk = group_argtopk(np.array(dset.slideIDX), inf_probs, args.k)
    dset.maketraindata(topk)
    fp = open(os.path.join(args.output, path), 'w')
    print('\t'.join(map(str,list(dset.lib.columns.values)+['pred','prob'])),file=fp)
    for idx in topk:
        row = dset.lib.iloc[idx]
        prob = inf_probs[idx]
        print('\t'.join(map(str,list(row.values)+[int(prob>0.5),prob])),file=fp)
    fp.close()

# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def pytorch_set_param_requires_grad(model, req_grad):
    for param in model.parameters():
        param.requires_grad = False

def inference(loader, model):
    model.eval()
    probs = torch.FloatTensor(len(loader.dataset))
    with torch.no_grad():
        for i, input in enumerate(loader):
            if i % 400 == 0: print('Inference\tBatch: [{}/{}]'.format(i+1, len(loader)))
            input = input.cuda()
            output = F.softmax(model(input), dim=1)
            probs[i*args.batch_size:i*args.batch_size+input.size(0)] = output.detach()[:,1].clone()
    return probs.cpu().numpy()

def train(run, loader, model, criterion, optimizer):
    model.train()
    running_loss = 0.
    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        output = model(input)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*input.size(0)
    return running_loss/len(loader.dataset)

def calc_err(pred,real):
    pred = np.array(pred)
    real = np.array(real)
    neq = np.not_equal(pred, real)
    err = float(neq.sum())/pred.shape[0]
    fpr = float(np.logical_and(pred==1,neq).sum())/(real==0).sum()
    fnr = float(np.logical_and(pred==0,neq).sum())/(real==1).sum()
    return err, fpr, fnr

def group_argtopk(groups, data,k=1):
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-k:] = True
    index[:-k] = groups[k:] != groups[:-k]
    return list(order[index])

def group_max(groups, data, nmax):
    out = np.empty(nmax)
    out[:] = np.nan
    order = np.lexsort((data, groups))
    groups = groups[order]
    data = data[order]
    index = np.empty(len(groups), 'bool')
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    out[groups[index]] = data[index]
    return out

class MILdataset(data.Dataset):
    def __init__(self, libraryfile='', target_k=None, target_v=None, transform=None, kpred=None):
        self.lib = pd.read_table(libraryfile)
        print('Number of tiles: {}'.format(len(self.lib)))
        self.target_k = target_k
        self.target_v = target_v
        self.transform = transform

        x2 = self.lib.groupby('slide',sort=True).first().reset_index()
        v, l = pd.factorize(self.lib.slide,sort=True)
        assert all(x2.slide==l)
        self.slidenames = l
        self.slideIDX = v
        self.targets = torch.from_numpy((x2[target_k]==target_v).astype(int).values)
        print('Target: {}={} n={}'.format(target_k, target_v,len(self.targets)))

        self.mode = None
        self.mult = 1
        self.size = 224
    def setmode(self,mode):
        self.mode = mode
    def maketraindata(self, idxs):
        self.t_data = [(self.lib.iloc[x], self.targets[self.slideIDX[x]]) for x in idxs]
        #self.t_data = [(self.slideIDX[x],self.grid[x],self.targets[self.slideIDX[x]]) for x in idxs]
    def shuffletraindata(self):
        self.t_data = random.sample(self.t_data, len(self.t_data))
    def __getitem__(self,index):
        if self.mode == 1:
            img = Image.open(self.lib.iloc[index].filename).convert('RGB') # remove alpha 
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.mode == 2:
            #slideIDX, coord, target = self.t_data[index]
            l, target = self.t_data[index]
            img = Image.open(l.filename).convert('RGB') # remove alpha 
            if self.transform is not None:
                img = self.transform(img)
            return img, target
    def __len__(self):
        if self.mode == 1:
            return len(self.lib)
        elif self.mode == 2:
            return len(self.t_data)

if __name__ == '__main__':
    main()
