import torch
import torch.nn as nn
import math
import numpy as np
import yaml
import torchvision
import time

def yaml_load(file='data.yaml'):
    # Single-line safe yaml loading
    with open(file, errors='ignore') as f:
        return yaml.safe_load(f)
    
def make_divisible(x, divisor):
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x/divisor) * divisor

def check_img_size(imgsz, s=32, floor=0):
    # Verify image size is a multiple of stride s in each dimension
    if isinstance(imgsz, int):  #just one value represents both h and w
        new_size = max(make_divisible(imgsz, int(s)), floor)
    else: #img_size=[640,480]
        imgsz = list(imgsz)  #if tuple
        new_size = [max(make_divisible(x, int(s)), floor) for x in imgsz]
    if new_size != imgsz:
        print(f'WARNING ⚠️ --img-size {imgsz} must be multiple of max stride {s}, updating to {new_size}')  
    return new_size

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.ReLU, nn.LeakyReLU, nn.SiLU]:
            m.inplace = True

def check_anchor_order(m):
    # Check anchor order against stride order for YOLOv5 Detect() module m, and correct if necessary
    a = m.anchors.prod(-1).mean(-1).view(-1) # mean anchor area per output layer  <-from (3,3,2) to (3,3) to (3,)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da and (da.sign() != ds.sign()): #if not small stride refer to small anchor
        m.anchors[:] = m.anchors.flip(0) #numpy flip along axis 0

def non_max_supression(
        prediction,
        conf_thres=0.25,
        iou_thres=0.45,
        max_det=300,
):
    """
    Non-Maximum Suppression (NMS) on inference results to reject overlapping detections

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls] not 85 anymore
    """
    #Checks
    assert 0 <= conf_thres <=1, f'Invalid Confidence threshold {conf_thres}, please set between 0.0 nd 1.0'
    assert 0 <= iou_thres <=1, f'Invalid IoU {iou_thres}, please set between 0.0 and 1.0'
    if isinstance(prediction, (list, tuple)): #use nms on eval() model prediction=(inference_out, loss_out)
        prediction = prediction[0] #only use inference output  (1,18900,85)
    device = prediction.device

    bs = prediction.shape[0]
    nc = prediction.shape[2] - 5 #number of classes
    xc = prediction[..., 4] > conf_thres  #return bool size as prediction[...] as bool mask ->prediction(xc) to select prediction that's got bigger confidence

    #Settings of nms
    max_wh = 7680
    max_nms = 30000 #maximum number box width and height
    time_limit = 0.5 + 0.05 * bs  #quit if process taking too long
    
    t = time.time() #process starting time
    output = [torch.zeros((0,6), device=device)] * bs  #0 for standplace 6 for xyxy score class
    for xi, x in enumerate(prediction): #image index, image inference result
       x = x[xc[xi]] # select those bigger confidence prediction

       if not x.shape[0]:
           continue
       
       #compute conf
       x[:, 5:] *= x[:, 4:5]
       
       box = xywh2xyxy(x[:,:4])

       conf, j = x[:,5:].max(1,keepdim=True)
       x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
       if not x.shape[0]:
           continue
       x = x[x[:, 4].argsort(descending=True)[:max_nms]]  #sort by confidence and remove excess boxes argsort return index

       #Batched NMS
       c = x[:, 5:6] * max_wh  # classes offset to make overlap booes with different class result in somehow different possition
       boxes, scores = x[:, :4] + c, x[:, 4] #boxes (offset by class), scores
       i = torchvision.ops.nms(boxes, scores, iou_thres)   #do nms Tensor: int64 tensor with the indices of the elements that have been kept by NMS, sorted in decreasing order of scores
       i = i[:max_det]   #limit

       output[xi] = x[i]  #add current img nms result to output

       if (time.time() - t) > time_limit:
            print(f'WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded')
            break  # time limit exceeded
    
    return output   #return (batchsize, n_result, 6)

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y =x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] /2
    return y


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None):
    # Rescale boxes (xyxy) from img1_shape to img0_shape
    if ratio_pad is None: #calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])   #gain = img / img0
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2 #wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    boxes[..., [0,2]] -= pad[0]
    boxes[..., [1,3]] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)  #cut the part that's out of input image
    return boxes
    
def clip_boxes(boxes, shape):
    #cut the box that's out of image
    if isinstance(boxes, torch.Tensor):
        boxes[..., 0].clamp_(0,shape[1])
        boxes[..., 1].clamp_(0,shape[0])
        boxes[..., 2].clamp_(0,shape[1])
        boxes[..., 3].clamp_(0,shape[0])
    else: #np
        boxes[..., [0,2]] = boxes[..., [0,2]].clip(0,shape[1])
        boxes[..., [1,3]] = boxes[..., [1,3]].clip(0,shape[0])
