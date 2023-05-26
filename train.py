import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
np.set_printoptions(supress=True)

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.backends.cudnn as cudnn
import torch.optim
from utils.tools import *
from torch.utils import data
from logger import Logger
from models.yolo5 import YoLov5

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT,Path.cwd()))

def get_args():
    parser = argparse.ArgumentParser(description='Start YOLO project')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=0, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=64, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--warmup', action='store_true', help='set lower initial learning rate to warm up')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--local_rank', default=-1, type=int, metavar='L', help='local_rank of gpu')

    parser.add_argument('-c', '--checkpoint', default='checkpoints', type= str, metavar='PATH',
                        help='path to save checkpoint')                                                                     #set argument parameters

    parser.add_argument('--title',default='',type=str,metavar='TITLE',help='title of checkpoint')
    parser.add_argument('--experiment',default='',type=str,metavar='EXP',help='identity-name of the experiment')
    parser.add_argument('--train-path', default='', type= str, metavar='PATH',
                        help='path to load train data') 

    parser.add_argument('--valid-path', default='', type= str, metavar='PATH',
                        help='path to load valid data') 
    
    parser.add_argument('--version-cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    parser.add_argument('--seed', type=int, default=None, help ='seed of env')

    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.deterministic = True
    cudnn.benchmark = False

def main():
    global args
    args = get_args()
    num_class = 80

    if args.seed:
        set_seed(args.seed)
    #env
    dist.init_process_group(backend='nccl')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    
    #model and env path
    model = YoLov5(str(args.version_cfg), input_channel=3, n_class=num_class)
    net_name = model.__class__.__name__
    args.checkpoint = '{}-{}-{}-{}'.format(args.checkpoint,args.title, net_name, time.strftime("%Y%m%d-%H%M%S")) 
    args.checkpoint = os.path.join(ROOT,args.experiment,args.checkpoint)

    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    #image size
    gs = max(int(model.stride.max()), 32)   #grid size(max stride->one grid in feature maps stands for gs grid in net input)
    imgsz = check_img_size(args.imgsz, gs, floor=gs*2)  #verify imgsz is gs-multiple

    #optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    #dataloader

if __name__ == '__main__':
    main()

