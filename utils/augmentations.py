import cv2
import numpy as np

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    #atuo map shape in ference, while training use the fix size(640,640)
    shape = im.shape[:2]  # current shape before changing [h,w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape) #expand one value to [h,w]
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1]) #choose the smaller one to make srue the bigger shape is match to new_shape
    if not scaleup:  #only scale down, do not scale up (for better val map)
        r = min(r, 1.0)

    #compute padding accroding to r
    ratio = r, r #width, height ratios
    new_unpad = int(round(shape[1] *r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  #wh padding(all but won't nesserary pad that's much)
    if auto: #minimum rectangle which just part of whole dwdh
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  #wh real padding
    elif scaleFill: #stretch to aim shape 
        dw, dh = 0.0, 0.0  #stretch no pad
        new_unpad = (new_shape[1], new_shape[0])  #aim to exact new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0] #width, height ratios

    dw /= 2 #divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad: #resize before padding
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1)) #incase all 7 but dh =3.5 spread as 3 and 4
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  #add border
    return im, ratio, (dw, dh)       #return im after padding, and ratio and pad num for reverse