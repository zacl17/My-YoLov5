from pathlib import Path 
import glob
import os
import cv2
from utils.augmentations import *

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes


class LoadImages:
    # YOLOv5 image/video dataloader, i.e. `python detect.py --source image.jpg/vid.mp4`
    def __init__(self, path, img_size=640, stride=32, auto=True, transforms=None, vid_stride=1):
        if isinstance(path, str) and Path(path).suffix == '.txt': # *.txt file with img/vid/dir on each line
            path = Path(path).read_text().rsplit()
        files = []
        for p in sorted(path) if isinstance(path, (list, tuple)) else [path]:
            p = str(Path(p).resolve())
            if '*' in p:
                files.extend(sorted(glob.glob(p, recursive=True))) #find file accroding to format path
            elif os.path.isdir(p):
                files.extend(sorted(glob.glob(os.path.join(p, '*.*'))))  #dir
            elif os.path.isfile(p):
                files.append(p)                                 #get file in each path
        
        images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
        videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv #number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        self.auto = auto
        self.transforms = transforms
        self.vid_stride = vid_stride
        if any(videos):
            self._new_video(videos[0])
        else:
            self.cap = None
        assert self.nf >0, f'No images or vedios found in{p}.' 

    def _new_video(self, path):
        #Create a new video capture object
        self.frame = 0
        self.cap = cv2.VideoCapture(path)             #open a video object
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.vid_stride)  #get() to obtain attribute of a vedio config, arg means Number of frames in the video file, / means select frams per stride
        self.orientation = int(self.cap.get(48)) #rotation degrees
    
    def __iter__(self):
        self.count = 0
        return self
    
    def __next__(self):
        if self.count == self.nf:
            raise StopIteration
        path = self.files[self.count]  #process current file

        if self.video_flag[self.count]:
            #Read video
            self.mode = 'video'
            for _ in range(self.vid_stride):
                self.cap.grab()    #point frame by skip vedio that much
            ret_val, im0 = self.cap.retrieve()
            while not ret_val:   #when one video is done move to next video
                self.count += 1  
                self.cap.release()
                if self.count == self.nf:  #last video
                    raise StopIteration
                path = self.files[self.count]
                self._new_video(path)
                ret_val, im0 = self.cap.read()

            self.frame += 1

            s = f'video {self.count + 1}/{self.nf} ({self.frame}/{self.frames}) {path}: '
        else:
            #Read image
            self.count += 1
            im0 = cv2.imread(path)  #cv2 get BGR hwc
            assert im0 is not None, f'Image Not Found {path}'
            s = f'image {self.count}/{self.nf} {path}'

        if self.transforms:
            im = self.transforms(im0)  #do given transforms
        else:                          #do auto pad resize set img as net_input style
            im = letterbox(im0, self.img_size, stride=self.stride, auto=self.auto)[0]  #[0] just get the img after padding
            im = im.transpose((2, 0, 1))[::-1]  #change HWC to CHW for net process
            im = np.ascontiguousarray(im)

        return path, im, im0, self.cap, s 
    
    def __len__(self):
        return self.nf  #number of files