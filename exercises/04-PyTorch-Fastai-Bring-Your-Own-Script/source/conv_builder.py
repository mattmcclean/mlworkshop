import os

import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_ as kaiming_normal

from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

from resnext import resnext50

model_meta = {
    resnet18:[8,6], resnet34:[8,6], resnet50:[8,6], resnet101:[8,6], resnet152:[8,6],
    resnext50:[8,6],
}

model_features = {} # nasnetalarge: 4032*2}

def cond_init(m, init_fn):
    if not isinstance(m, (nn.BatchNorm1d,nn.BatchNorm2d,nn.BatchNorm3d)):
        if hasattr(m, 'weight'): init_fn(m.weight)
        if hasattr(m, 'bias') and hasattr(m.bias, 'data'): m.bias.data.fill_(0.)

def apply_init(m, init_fn):
    m.apply(lambda x: cond_init(x, init_fn))

def children(m): return m if isinstance(m, (list, tuple)) else list(m.children())

def cut_model(m, cut):
    return list(m.children())[:cut] if cut else [m]

def split_by_idxs(seq, idxs):
    '''A generator that returns sequence pieces, seperated by indexes specified in idxs. '''
    last = 0
    for idx in idxs:
        if not (-len(seq) <= idx < len(seq)):
          raise KeyError('Idx {} is out-of-bounds'.format(idx))
        yield seq[last:idx]
        last = idx
    yield seq[last:]

def num_features(m):
    c=children(m)
    if len(c)==0: return None
    for l in reversed(c):
        if hasattr(l, 'num_features'): return l.num_features
        res = num_features(l)
        if res is not None: return res

class AdaptiveConcatPool2d(nn.Module):
    def __init__(self, sz=None):
        super().__init__()
        sz = sz or (1,1)
        self.ap = nn.AdaptiveAvgPool2d(sz)
        self.mp = nn.AdaptiveMaxPool2d(sz)
    def forward(self, x): return torch.cat([self.mp(x), self.ap(x)], 1)

class Lambda(nn.Module):
    def __init__(self, f): super().__init__(); self.f=f
    def forward(self, x): return self.f(x)

class Flatten(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, x): return x.view(x.size(0), -1)

class ConvnetBuilder():
    """Class representing a convolutional network.
    Arguments:
        f: a model creation function (e.g. resnet34, vgg16, etc)
        c (int): size of the last layer
        is_multi (bool): is multilabel classification?
            (def here http://scikit-learn.org/stable/modules/multiclass.html)
        is_reg (bool): is a regression?
        ps (float or array of float): dropout parameters
        xtra_fc (list of ints): list of hidden layers with # hidden neurons
        xtra_cut (int): # layers earlier than default to cut the model, default is 0
        custom_head : add custom model classes that are inherited from nn.modules at the end of the model
                      that is mentioned on Argument 'f' 
    """

    def __init__(self, f, c, is_multi=False, is_reg=False, ps=None, xtra_fc=None, xtra_cut=0, custom_head=None):
        self.f,self.c,self.is_multi,self.is_reg,self.xtra_cut = f,c,is_multi,is_reg,xtra_cut
        if xtra_fc is None: xtra_fc = [512]
        if ps is None: ps = [0.25]*len(xtra_fc) + [0.5]
        self.ps,self.xtra_fc = ps,xtra_fc

        if f in model_meta: cut,self.lr_cut = model_meta[f]
        else: cut,self.lr_cut = 0,0
        cut-=xtra_cut
        layers = cut_model(f(), cut)
        self.nf = model_features[f] if f in model_features else (num_features(layers)*2)
        if not custom_head: layers += [AdaptiveConcatPool2d(), Flatten()]
        self.top_model = nn.Sequential(*layers)

        n_fc = len(self.xtra_fc)+1
        if not isinstance(self.ps, list): self.ps = [self.ps]*n_fc

        if custom_head: fc_layers = [custom_head]
        else: fc_layers = self.get_fc_layers()
        self.n_fc = len(fc_layers)
        self.fc_model = nn.Sequential(*fc_layers)
        if not custom_head: apply_init(self.fc_model, kaiming_normal)
        self.model = nn.Sequential(*(layers+fc_layers))

    @property
    def name(self): return '{}_{}'.format(self.f.__name__, self.xtra_cut)

    def create_fc_layer(self, ni, nf, p, actn=None):
        res=[nn.BatchNorm1d(num_features=ni)]
        if p: res.append(nn.Dropout(p=p))
        res.append(nn.Linear(in_features=ni, out_features=nf))
        if actn: res.append(actn)
        return res

    def get_fc_layers(self):
        res=[]
        ni=self.nf
        for i,nf in enumerate(self.xtra_fc):
            res += self.create_fc_layer(ni, nf, p=self.ps[i], actn=nn.ReLU())
            ni=nf
        final_actn = nn.Sigmoid() if self.is_multi else nn.LogSoftmax()
        if self.is_reg: final_actn = None
        res += self.create_fc_layer(ni, self.c, p=self.ps[-1], actn=final_actn)
        return res

    def get_layer_groups(self, do_fc=False):
        if do_fc:
            return [self.fc_model]
        idxs = [self.lr_cut]
        c = children(self.top_model)
        if len(c)==3: c = children(c[0])+c[1:]
        lgs = list(split_by_idxs(c,idxs))
        return lgs+[self.fc_model]