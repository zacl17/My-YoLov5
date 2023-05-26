import torch.nn as nn
import torch
import warnings


def autopad(k, p=None, d=1):  #d for dilation
    if d > 1:   ##if is dilation conv which makes the fields bigger
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # when use dilation the actuall kernel size is bigger
    if p is None:
        p = k // 2 if isinstance(k,int) else [x//2 for x in k]

    return p

class Conv(nn.Module):
    default_act = nn.SiLU()  #default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):   #d default as 1 no dilation
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()    #default as Ture -> nn.SiLU
    
    def forward(self,x):
        return self.act(self.bn(self.conv(x))) #forward as Conv
    

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 *e)   #default as 1.0 from c3 which won't reduce channel by 1x1
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self,x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shorcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  #e as half to build the input of concate
        self.cv1 = Conv(c1, c_, 1, 1)  #head of main branch
        self.cv2 = Conv(c1, c_, 1, 1)  #skip connect
        self.cv3 = Conv(2 * c_, c2, 1)   # last combination conv
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shorcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)),1))
    

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):    #k for kernel size of maxpooling layer
        super().__init__()
        c_ = c1 // 2    #channel after first conv but before maxpooling
        self.cv1 = Conv(c1, c_, 1, 1) #conv before maxpooling
        self.cv2 = Conv(4*c_, c2, 1, 1) #conv fater maxpooling
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))
        

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = Conv(c1* 4, c2, k, s, p, g, act=act)    #conv after slice
    
    def forward(self, x):
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
    

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)