import numpy as np
import cv2

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'

class Annotator:
    # data Annotator for train/val mosaics and jpgs and detect/hub inference annotations
    def __init__(self, im, line_width=None):
        assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to Annotator() input images.'
        self.im = im #cv2
        self.lw = line_width or max(round(sum(im.shape) / 2 * 0.003), 2)  #line width
    
    def box_label(self, box, label='', color=(128,128,128), txt_color=(255, 255, 255)):
        # Add one xyxy box to image with labels
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))  #upper_left, down_right
        cv2.rectangle(self.im, p1, p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:  #plot label text
            tf = max(self.lw -1, 1) #font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=self.lw/3, thickness=tf)[0]  #get text width, height
            outside = p1[1] - h >=3  #if there is a room above the detect box put text box on that
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h +3 #put below
            cv2.rectangle(self.im, p1, p2, color, -1, cv2.LINE_AA)  #plot label txt box
            cv2.putText(
                self.im,
                label,
                (p1[0], p1[1] -2 if outside else p1[1]+h+2),
                0,
                self.lw /3,
                txt_color,
                thickness=tf,
                lineType=cv2.LINE_AA
            )
    def result(self):
        #Return annotated image as array
        return np.asarray(self.im)
