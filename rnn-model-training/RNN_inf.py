import os
from PIL import Image
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
import pandas as pd

parser = argparse.ArgumentParser(description='MIL-nature-medicine-2019 RNN aggregator training script')
parser.add_argument('--train_lib', type=str, default='', help='path to train MIL library binary')
parser.add_argument('--val_lib', type=str, default='', help='path to validation MIL library binary. If present.')
parser.add_argument('--output', type=str, default='.', help='name of output file')
parser.add_argument('--target-k', type=str, default=None, help='target column key')
parser.add_argument('--target-v', type=bool, default=None, help='target column value')
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--k', default=10, type=int, help='how many top k tiles to consider (default: 10)')
parser.add_argument('--ndims', default=128, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--model', type=str, help='path to trained model checkpoint')
parser.add_argument('--rnn-model', type=str, help='path to trained model checkpoint')
parser.add_argument('--weights', default=0.5, type=float, help='unbalanced positive class weight (default: 0.5, balanced classes)')
parser.add_argument('--shuffle', action='store_true', help='to shuffle order of sequence')

best_acc = 0
def main():
    global args, best_acc
    args = parser.parse_args()
    
    #load libraries
    #normalize = transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.1,0.1,0.1])
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
    trans = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])
    train_dset = rnndata(args.train_lib, args.k, args.shuffle, trans, args.target_k, args.target_v)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_dset = rnndata(args.val_lib, args.k, False, trans, args.target_k, args.target_v)
    val_loader = torch.utils.data.DataLoader(
        val_dset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=False)

    #make model
    embedder = ResNetEncoder(args.model)
    for param in embedder.parameters():
        param.requires_grad = False
    embedder = embedder.cuda()
    embedder.eval()

    rnn = rnn_single(args.ndims)
    ch = torch.load(args.rnn_model)
    rnn.load_state_dict(ch['state_dict'])
    rnn = rnn.cuda()
    
    #optimization
    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    cudnn.benchmark = True

    save_inferences(embedder, rnn, val_loader, val_dset, 'rnn_val_inferences.tsv')
    save_inferences(embedder, rnn, train_loader, train_dset, 'rnn_train_inferences.tsv')
 

def save_inferences(embedder, rnn, loader, dset, path):
    rnn.eval()
    fp = open(os.path.join(args.output, path), 'w')
    print('\t'.join(map(str,list(dset.slides.columns.values)+['p0','p1','target'])), file=fp)
    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            print('Batch:[{}/{}]'.format(i+1,len(loader)))
            
            batch_size = inputs[0].size(0)
            
            state = rnn.init_hidden(batch_size).cuda()
            for s in range(len(inputs)):
                input = inputs[s].cuda()
                _, input = embedder(input)
                output, state = rnn(input, state)
                #output = F.softmax(model(input), dim=1)
            
            row = dset.slides.iloc[i]
            print('\t'.join(map(str,list(row.values)+output[0].tolist()+target.tolist())), file=fp)
            

def errors(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.squeeze().cpu().numpy()
    real = target.numpy()
    neq = pred!=real
    fps = float(np.logical_and(pred==1,neq).sum())
    fns = float(np.logical_and(pred==0,neq).sum())
    return fps,fns

class ResNetEncoder(nn.Module):

    def __init__(self, path):
        super(ResNetEncoder, self).__init__()

        temp = models.resnet34()
        temp.fc = nn.Linear(temp.fc.in_features, 2)
        ch = torch.load(path)
        temp.load_state_dict(ch['state_dict'])
        self.features = nn.Sequential(*list(temp.children())[:-1])
        self.fc = temp.fc

    def forward(self,x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        return self.fc(x), x

class rnn_single(nn.Module):

    def __init__(self, ndims):
        super(rnn_single, self).__init__()
        self.ndims = ndims

        self.fc1 = nn.Linear(512, ndims)
        self.fc2 = nn.Linear(ndims, ndims)
        self.fc3 = nn.Linear(ndims, 2)

        self.activation = nn.ReLU()

    def forward(self, input, state):
        input = self.fc1(input)
        state = self.fc2(state)
        state = self.activation(state+input)
        output = self.fc3(state)
        return output, state

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.ndims)

class rnndata(data.Dataset):

    def __init__(self, path, k, shuffle=False, transform=None, target_k=None, target_v=None):
        self.lib = pd.read_csv(path,sep='\t')
        self.k = k
        self.transform = transform
        self.target_k = target_k
        self.target_v = target_v

        x2 = self.lib.groupby('slide',sort=True).first().reset_index()
        v, l = pd.factorize(self.lib.slide,sort=True)
        assert all(x2.slide==l)
        self.slidenames = l
        self.slideIDX = v
        self.slides = x2
        self.targets = torch.from_numpy((x2[target_k]==target_v).astype(int).values)
        #self.grid = lib['grid']
        #self.level = lib['level']
        self.mult = 1
        self.size = 224
        self.shuffle = shuffle

    def __getitem__(self,index):

        items = self.lib[self.lib.slide==self.slides.iloc[index].slide]
        items = items.nlargest(self.k, 'prob') # top K tiles from MIL
        #slide = self.slides[index]
        #grid = self.grid[index]
        if self.shuffle:
            items = items.sample(frac=1)

        out = []
        for i, r in items.iterrows():
            img = Image.open(r.filename).convert('RGB')
            if self.mult != 1:
                img = img.resize((224,224), Image.BILINEAR)
            if self.transform is not None:
                img = self.transform(img)
            out.append(img)
        
        return out, self.targets[index]

    def __len__(self):
        
        return len(self.targets)

if __name__ == '__main__':
    main()
