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
parser.add_argument('--batch_size', type=int, default=128, help='mini-batch size (default: 128)')
parser.add_argument('--nepochs', type=int, default=100, help='number of epochs')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--k', default=10, type=int, help='how many top k tiles to consider (default: 10)')
parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
parser.add_argument('--ndims', default=128, type=int, help='length of hidden representation (default: 128)')
parser.add_argument('--model', type=str, help='path to trained model checkpoint')
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
    train_dset = rnndata(args.train_lib, args.k, args.shuffle, trans)
    train_loader = torch.utils.data.DataLoader(
        train_dset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=False)
    val_dset = rnndata(args.val_lib, args.k, False, trans)
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
    rnn = rnn.cuda()
    
    #optimization
    if args.weights==0.5:
        criterion = nn.CrossEntropyLoss().cuda()
    else:
        w = torch.Tensor([1-args.weights,args.weights])
        criterion = nn.CrossEntropyLoss(w).cuda()
    optimizer = optim.SGD(rnn.parameters(), lr=args.lr, momentum=0.9, dampening=0, weight_decay=1e-3, nesterov=True)
    cudnn.benchmark = True

    fconv = open(os.path.join(args.output, 'rnn_convergence.csv'), 'w')
    fconv.write('epoch,train.loss,train.fpr,train.fnr,val.loss,val.fpr,val.fnr\n')
    fconv.close()

    #
    for epoch in range(args.nepochs):

        train_loss, train_fpr, train_fnr = train_single(epoch, embedder, rnn, train_loader, criterion, optimizer)
        val_loss, val_fpr, val_fnr = test_single(epoch, embedder, rnn, val_loader, criterion)

        fconv = open(os.path.join(args.output,'rnn_convergence.csv'), 'a')
        fconv.write('{},{},{},{},{},{},{}\n'.format(epoch+1, train_loss, train_fpr, train_fnr, val_loss, val_fpr, val_fnr))
        fconv.close()

        val_err = (val_fpr + val_fnr)/2
        if 1-val_err >= best_acc:
            best_acc = 1-val_err
            obj = {
                'epoch': epoch+1,
                'state_dict': rnn.state_dict()
            }
            fn = 'rnn_model_{:.5f}_k{:d}_{:0.0e}.pth'.format(best_acc,args.k,args.lr)
            torch.save(obj, os.path.join(args.output,fn))

def train_single(epoch, embedder, rnn, loader, criterion, optimizer):
    rnn.train()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    for i,(inputs,target) in enumerate(loader):
        print('Training Epoch:[{}/{}] Batch:[{}/{}]'.format(epoch+1, args.nepochs, i+1, len(loader)))

        batch_size = inputs[0].size(0)
        rnn.zero_grad()

        state = rnn.init_hidden(batch_size).cuda()
        for s in range(len(inputs)):
            input = inputs[s].cuda()
            _, input = embedder(input)
            output, state = rnn(input, state)

        target = target.cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()*target.size(0)
        fps, fns = errors(output.detach(), target.cpu())
        running_fps += fps
        running_fns += fns

    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    print('Training Epoch:[{}/{}] Loss:{} FPR:{} FNR:{}'.format(epoch+1, args.nepochs, running_loss, running_fps, running_fns))
    return running_loss, running_fps, running_fns

def test_single(epoch, embedder, rnn, loader, criterion):
    rnn.eval()
    running_loss = 0.
    running_fps = 0.
    running_fns = 0.

    with torch.no_grad():
        for i,(inputs,target) in enumerate(loader):
            print('Validating Epoch:[{}/{}] Batch:[{}/{}]'.format(epoch+1,args.nepochs,i+1,len(loader)))
            
            batch_size = inputs[0].size(0)
            
            state = rnn.init_hidden(batch_size).cuda()
            for s in range(len(inputs)):
                input = inputs[s].cuda()
                _, input = embedder(input)
                output, state = rnn(input, state)
            
            target = target.cuda()
            loss = criterion(output,target)
            
            running_loss += loss.item()*target.size(0)
            fps, fns = errors(output.detach(), target.cpu())
            running_fps += fps
            running_fns += fns
            
    running_loss = running_loss/len(loader.dataset)
    running_fps = running_fps/(np.array(loader.dataset.targets)==0).sum()
    running_fns = running_fns/(np.array(loader.dataset.targets)==1).sum()
    err = (running_fps+running_fns)/2
    print('Val Epoch:[{}/{}] Loss:{} acc:{} FPR:{} FNR:{}'.format(epoch+1, args.nepochs, running_loss, 1-err, running_fps, running_fns))
    return running_loss, running_fps, running_fns

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

    def __init__(self, path, k, shuffle=False, transform=None):
        self.lib = pd.read_csv(path,sep='\t')
        self.k = k
        self.transform = transform
        target_k = 'top'
        target_v = True

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
