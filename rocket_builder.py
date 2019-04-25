import os
from .model import s3fd
from .utils import *
import torch
import torch.nn.functional as F
from torchvision import transforms
import types
from PIL import Image, ImageDraw

def build():
    model = s3fd()
    model.load_state_dict(torch.load(os.path.join(os.path.realpath(os.path.dirname(__file__)),
                                                  "weights.pth")),
                          strict=True)

    model.postprocess = types.MethodType(postprocess, model)
    model.preprocess = types.MethodType(preprocess, model)

    return model

def preprocess(self, x):
    """Converts PIL Image or Array into pytorch tensor specific to this model

    Handles all the necessary steps for preprocessing such as resizing, normalization.
    Works with both single images and list/batch of images. Input image file is expected
    to be a `PIL.Image` object with 3 color channels.

    Args:
        x (list or PIL.Image): input image or list of images.
    """
    x = x.convert("RGB")
    x = np.array(x)
    x = x - np.array([104, 117, 123])
    x = x.transpose(2, 0, 1)
    x = x.reshape((1,) + x.shape)

    x = torch.from_numpy(x).float()

    return x

def postprocess(self, x: torch.Tensor, input_img: Image, visualize: bool = False):
    """Converts pytorch tensor into PIL Image

    Handles all the steps for postprocessing of the raw output of the model.
    Depending on the rocket family there might be additional options.

    Args:
        x (Tensor): Output Tensor to postprocess
    """
    olist = x.copy()
    bboxlist = []

    for i in range(int(len(olist) / 2)): olist[i * 2] = F.softmax(olist[i * 2], dim=1)
    for i in range(int(len(olist) / 2)):
        ocls, oreg = olist[i * 2].data.cpu(), olist[i * 2 + 1].data.cpu()
        FB, FC, FH, FW = ocls.size()  # feature map size
        stride = 2 ** (i + 2)  # 4,8,16,32,64,128
        anchor = stride * 4
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for Iindex, hindex, windex in poss:
            axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
            score = ocls[0, 1, hindex, windex]
            loc = oreg[0, :, hindex, windex].contiguous().view(1, 4)
            priors = torch.Tensor([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
            variances = [0.1, 0.2]
            box = decode(loc, priors, variances)
            x1, y1, x2, y2 = box[0] * 1.0
            # cv2.rectangle(imgshow,(int(x1),int(y1)),(int(x2),int(y2)),(0,0,255),1)
            bboxlist.append([x1, y1, x2, y2, score])
    bboxlist = np.array(bboxlist)
    if 0 == len(bboxlist): bboxlist = np.zeros((1, 5))

    keep = nms(bboxlist, thresh=0.3)
    bboxlist = bboxlist[keep, :]

    list_detections = []

    for bbox in bboxlist:
        if bbox[4] < 0.3:
            continue
        list_detections.append({
                'topLeft_x': bbox[0],
                'topLeft_y': bbox[1],
                'width': bbox[2] - bbox[0],
                'height': bbox[3] - bbox[1],
                'bbox_confidence': bbox[4]})

    if visualize:
        line_width = 2
        img_out = input_img
        ctx = ImageDraw.Draw(img_out, 'RGBA')
        for detection in list_detections:
            # Extract information from the detection
            topLeft = (detection['topLeft_x'], detection['topLeft_y'])
            bottomRight = (detection['topLeft_x'] + detection['width'] - line_width, detection['topLeft_y'] + detection['height']- line_width)
            bbox_confidence = detection['bbox_confidence']

            # Draw the bounding boxes and the information related to it
            ctx.rectangle([topLeft, bottomRight], outline=(255, 0, 0, 255), width=line_width)
            ctx.text((topLeft[0] + 5, topLeft[1] + 10), text="{:.2f}".format(bbox_confidence))

        del ctx
        return img_out

    return list_detections



