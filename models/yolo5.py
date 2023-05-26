import contextlib
import torch
import torch.nn as nn
from copy import deepcopy
from models.bones import *
from utils.tools import *


class Detect(nn.Module):
    # not compelete yet--without inference type output
    # detect head for detection models
    stride = None     #strides computed during build
    dynamic = False  #force grid reconstruction
    export = False   #export mode

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):          #[nc, 9_anchors[3,6],[ch[17],ch[20],ch[23]]]
        super().__init__()
        self.nc = nc   # class_number
        self.no = nc + 5 # number of outputs per anchor including number of class and (tx, ty, th, tw, to)
        self.nl = len(anchors) #number of detection layers each layer got it's own scale
        self.na = len(anchors[0]) // 2 #number of anchors per scale
        self.grid = [torch.empty(0) for _ in range(self.nl)] # init grid  torch.empty(0) is a scalar stands for a s placeholder tensor for compute xy
        self.anchor_grid = [ torch.empty(0) for _ in range(self.nl) ] #init anchor grid for compute wh
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2)) # register_buffer to save anchor choice in the model->shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  #build last output head-conv of each scale
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):
        z = []  #inference output not just tx,ty but real prediction
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  #last output conv of current scale x is a output list of three scale
            bs, _, ny, nx = x[i].shape # x(bs,3*85,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
        
            if not self.training: #durning inference output real coordinates  
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:                 #data in grid[2:4] default as empty at very start
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)   #both shape make as (1, self.na, ny, nx, 2)  stand for grid coordinate of features and anchor size in input
                
                xy, wh, conf = x[i].sigmoid().split((2,2,self.nc+1),4)  #split output in 85 as 2, 2, 81
                xy = (xy * 2 + self.grid[i]) * self.stride[i] #compute real center in input image
                wh = (wh * 2) ** 2 * self.anchor_grid[i]  #compute real size in input image
                y = torch.cat((xy, wh, conf), 4) #cat back as shape of x (bs, 3, 20, 20, 85)
                z.append(y.view(bs, self.na*nx*ny, self.no))   #build real size output as(nl, batchsize, na*feature_size, 85)
        return x if self.training else (torch.cat(z,1), ) if self.export else (torch.cat(z,1), x)  #output x as (3,bs,3,20,20,85) for trainging or (bs, na(sun of all feature_size), 85) for inference
    
    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2 #grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        yv, xv = torch.meshgrid(y, x, indexing=('ij'))
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5   #cause we comput real position with offset by 2.0*f(offset)-0.5+grid so here we do -0.5 at very start
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid


def get_modules(cf,ch):
    anchors, nc, gd, gw, act = cf['anchors'], cf['nc'], cf['depth_multiple'], cf['width_multiple'], cf.get('activation')
    if act:
        Conv.default_act = eval(act) # redefine default activation
    na = len(anchors[0]) // 2  #number of anchors of each scale  //2 due to the data stored in anchors is h,w
    no = na * (nc + 5) #number of output channels of each scale = anchors * (classes + 5)

    layers, save, c2 = [], [], ch[-1]  #layers, save to reocrd the from that's not -1 (concat,detect), output channels
    for i, (f, n, m, args) in enumerate(cf['backbone'] + cf['head']):    #from in ch_list, number, moudle, moudle_args
        m = eval(m) if isinstance(m, str) else m   #eval strings
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a
        
        n = max(round(n*gd), 1) if n > 1 else n     #compute x in csp1_x which result in depth
        if m in { Conv, Bottleneck, SPPF, Focus, C3 }:
            c1, c2 = ch[f], args[0]      #get in and output chaneel of current module
            if c2 != no:     #if not output <- no stands for the output of prediction
                c2 = make_divisible(c2 * gw, 8)   #compute output channel of each module according to factor gw

            args = [c1, c2, *args[1:]]    #update args to construct current module
            if m in {Bottleneck, C3}:  #default no Bottleneck in cfg but in C3
                args.insert(2, n)  #number of repeats
                n = 1
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)   # Concat doesn't need c2 to build but comput c2 for record in ch list
        elif m is Detect:
            args.append([ch[x] for x in f])  #record input channel of each scale
            if isinstance(args[1], int):     #default as list of all scale anchors
                args[1] = [list(range(args[1]*2))] * len(f)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  #bulid current moudle accroding to the args
        t = str(m)[8:-2].replace('__main__.', '')    #module type
        np = sum(x.numel() for x in m_.parameters())  #number of params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np   #record info in moudle's attribution
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  #record the other from of concat and detect besides -1
        layers.append(m_)  #add current builded module into layer_list
        if i == 0:
            ch = []  #only save output not input of first layer
        ch.append(c2) #update output channel of each layer list
    return nn.Sequential(*layers), sorted(save)   #use sequential to define the longest routine but not the data flow, sorted from small to big [4, 6, 10, 14, 17, 20, 23] 
    


class YoLov5(nn.Module):
    def __init__(self, cfg='yolov5s.yaml', input_channel=3, n_class=None):
        super(YoLov5,self).__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)
        self.in_ch = self.yaml['ch'] = self.yaml.get('ch', input_channel)
        if n_class and n_class != self.yaml['nc']:
            self.yaml['nc'] = n_class
        self.n_class = n_class

        self.model, self.save = get_modules(deepcopy(self.yaml), ch=[self.in_ch])    #use sequential to define the module flow but not the data flow, save to record concate data flow sorted from small to big [4, 6, 10, 14, 17, 20, 23]
        self.names = [str(i) for i in range(self.yaml['nc'])]  #default class_names as number-style
        self.inplace = self.yaml.get('inplace', True)

        #build strides, anchors for last detect head
        m = self.model[-1]   #choose the last moudle which is Detect()
        if isinstance(m, Detect):
            s = 256 #as a safe input size to compute stride of 3 scale
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, self.in_ch, s, s))])
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1,1,1)   #view anchors as in feature grid not input-wise
            self.stride = m.stride
            self._initialize_biases()

        #Init
        initialize_weights(self)
    
    def forward(self, x, augment=False):   #not compelete without augment override forword data flow in Sequential
        if augment:
            pass
        else:
            return self._forward_once(x)
    
    def _forward_once(self, x):  #bulid data flow to override forward in Sequential
        y = []  #output of each module
        for m in self.model:
            if m.f != -1:    #if not only from previous layer which means FPN concat and last detect set input x as a list
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f] #from early layer
            x = m(x)  #bulid forward one by one get output x accroding to input x
            y.append(x if m.i in self.save else None)  #save output of current layer for concate in FPN or detect() input
        return x #output of the last detect() 
    
    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)