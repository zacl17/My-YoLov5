import argparse
import os
import platform
import sys
from pathlib import Path
from models.yolo5 import YoLov5
import torch
import torch.nn as nn 
from collections import OrderedDict
from utils.tools import *
from utils.plots import *
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT,Path.cwd()))

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def load_model_cpu(net, model):
    state_dict = torch.load(model, map_location=torch.device('cpu'))['state_dict']
    if 'module.' in list(state_dict.keys())[0]:
       new_state_dict = OrderedDict()
       for k, v in state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    net.load_state_dict(new_state_dict)
    return net

def parse_opt():
    parser = argparse.ArgumentParser(description='yolov5 inference starting')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5.pth', help='model path or triton URL')
    parser.add_argument('--source', type=str, default='data\images\dog.jpg', help='picture for detect')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img-size', nargs='+', type=int, default=640, help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IOU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--save-txt', action='store_false', help='save result')
    parser.add_argument('--save-path', default=ROOT / 'runs/detect', help='save path of result')
    parser.add_argument('--name', default='exp', help='save results to save_path/name')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--version-cfg', type=str, default=ROOT / 'models/yolov5s.yaml', help='model.yaml path')
    return parser.parse_args()

def main():
    global args 
    args = parse_opt()
    source = str(args.source)
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)

    save_dir = os.path.join(args.save_path, args.name)
    if not os.path.isdir(save_dir):
        if args.save_txt:
            os.makedirs(os.path.join(save_dir, 'labels'), exist_ok=True)
        else:
            os.makedirs(save_dir)
    # model = torch.load(args.weights, map_location='cpu')['model']
    # torch.save({'state_dict': model.state_dict()},'yolov5.pth') #load from pt
    model = YoLov5(str(args.version_cfg), input_channel=3, n_class=80)
    model = load_model_cpu(model, args.weights)
    stride, names = max(int(model.stride.max()), 32), model.names
    data_names = yaml_load(args.data)['names'] if args.data else {i: f'class{i}' for i in range(999)}
    if len(names) == len(data_names):
        names = data_names
    model = nn.DataParallel(model).cpu()
    model.eval()                         #load weight prepare model in DP style

    imgsz = check_img_size(args.imgsz, s=stride)
    dataset = LoadImages(source, img_size=imgsz, stride=stride)           #if source from file which may contains txt or IMG or Video

    for path, im, im0s, vid_cap, s in dataset:
        im = torch.from_numpy(im).cpu()
        im = im.float()
        im /= 255
        if len(im.shape)  == 3:
            im = im[None] #expand for batch dim
        
        pred = model(im)                  #len(pred)=2    pred[0].shape=(1,18900,85) len(pred[1]=3)  pred[1][0].shape=(1,3,80,60,85) pred[1][1].shape=(1,3,40,30,85) pred[1][2].shape=(1,3,20,15,85) 18900=3*(80*60+40*30+20*15)
        pred = non_max_supression(pred, args.conf_thres, args.iou_thres, max_det=args.max_det)  #return (batchsize=1, n_detect_result, 6)

        for i, det in enumerate(pred): #per image ->det is (n_detect_result, 6)
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)  #im0 is the image before letter_box

            p = Path(p) # point to image path
            save_path = str(os.path.join(save_dir, p.name))  # save im.jpg
            txt_path = str(os.path.join(save_dir, 'labels', p.stem)) + ('' if dataset.mode == 'image' else f'_{frame}')  #save im.txt
            s += ' %gx%g ' % im.shape[2:]  #print string message
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #normalization gain wh from original img before pad as (w, h, w, h)
            annotator = Annotator(im0, line_width=args.line_thickness)   #define a annotator object for plot
            if len(det):
                # Rescale boxes from img_size to im0 size  which means reverse letter_box
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                for c in det[:,5].unique():
                    n = (det[:, 5] == c).sum()
                    s += f"{n} {names[int(c)]}{'s' * (n>1)},"  #for print
                
                #write results
                for *xyxy, conf, cls in reversed(det):
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1,4)) / gn).view(-1).tolist()  #normalized xywh
                    line = cls, *xywh, conf  #message
                    with open(txt_path+'.txt', 'a+') as f:
                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    c = int(cls)
                    label = f'{names[c]} {conf:.2f}'
                    annotator.box_label(xyxy, label, color=colors(c, True))
            im0 = annotator.result()
            if dataset.mode == 'image':
                cv2.imwrite(save_path, im0)
            
    print(s)
if __name__ == '__main__':
    main()