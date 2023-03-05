import cv2
import onnxruntime as ort
import numpy as np 
from utlis import show_seg_result
from general import non_max_suppression
import torch
import threading

def load_model(model_path):
    ort.set_default_logger_severity(4)
    ort_inference = ort.InferenceSession(model_path)
    print("model loaded")
    return ort_inference

def resize_unscale(img, new_shape=(640, 640), color=114):
    shape = img.shape[:2]  
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    canvas = np.zeros((new_shape[0], new_shape[1], 3))
    canvas.fill(color)
    # Scale ratio (new / old) new_shape(h,w)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  
    new_unpad_w = new_unpad[0]
    new_unpad_h = new_unpad[1]
    pad_w, pad_h = new_shape[1] - new_unpad_w, new_shape[0] - new_unpad_h  

    dw = pad_w // 2  
    dh = pad_h // 2

    if shape[::-1] != new_unpad:  
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_AREA)

    canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :] = img
    return canvas, r, dw, dh, new_unpad_w, new_unpad_h  

def infer(frame, Infer_session):

    conf_list = []
    height, width, _ = frame.shape 
    img_rgb = frame[:, :, ::-1].copy()

    canvas, r, dw, dh, new_unpad_w, new_unpad_h = resize_unscale(img_rgb, (640, 640))
    img = canvas.copy().astype(np.float32)  # (3,640,640) RGB
    img /= 255.0
    img[:, :, 0] -= 0.485
    img[:, :, 1] -= 0.456
    img[:, :, 2] -= 0.406
    img[:, :, 0] /= 0.229
    img[:, :, 1] /= 0.224
    img[:, :, 2] /= 0.225

    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, 0)

    det_out, da_seg_out, ll_seg_out = Infer_session.run(
        ['det_out', 'drive_area_seg', 'lane_line_seg'],
        input_feed={"images": img}
    )
    det_out = torch.from_numpy(det_out).float()
    boxes = non_max_suppression(det_out)[0]  # [n,6] [x1,y1,x2,y2,conf,cls]
    boxes = boxes.cpu().numpy().astype(np.float32)

    boxes[:, 0] -= dw
    boxes[:, 1] -= dh
    boxes[:, 2] -= dw
    boxes[:, 3] -= dh
    boxes[:, :4] /= r

    da_seg_out = da_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]
    ll_seg_out = ll_seg_out[:, :, dh:dh + new_unpad_h, dw:dw + new_unpad_w]

    da_seg_mask = np.argmax(da_seg_out, axis=1)[0]  # (?,?) (0|1)
    ll_seg_mask = np.argmax(ll_seg_out, axis=1)[0]  # (?,?) (0|1)

    color_area = np.zeros((new_unpad_h, new_unpad_w, 3), dtype=np.uint8)
    color_area[da_seg_mask == 1] = [0, 255, 0]
    color_area[ll_seg_mask == 1] = [255, 0, 0]
    color_seg = color_area

    color_seg = color_seg[..., ::-1]
    color_mask = np.mean(color_seg, 2)
    img_merge = canvas[dh:dh + new_unpad_h, dw:dw + new_unpad_w, :]
    img_merge = img_merge[:, :, ::-1]

    img_merge[color_mask != 0] = \
        img_merge[color_mask != 0] * 0.5 + color_seg[color_mask != 0] * 0.5
    img_merge = img_merge.astype(np.uint8)
    img_merge = cv2.resize(img_merge, (width, height),
                           interpolation=cv2.INTER_LINEAR)
    
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2, conf, label = boxes[i]
        x1, y1, x2, y2, label = int(x1), int(y1), int(x2), int(y2), int(label)
        img_merge = cv2.rectangle(img_merge, (x1, y1), (x2, y2), (0, 255, 0), 2, 2)
        conf_list.append(conf)

    da_seg_mask = da_seg_mask * 255
    da_seg_mask = da_seg_mask.astype(np.uint8)
    da_seg_mask = cv2.resize(da_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)
    
    ll_seg_mask = ll_seg_mask * 255
    ll_seg_mask = ll_seg_mask.astype(np.uint8)
    ll_seg_mask = cv2.resize(ll_seg_mask, (width, height),
                             interpolation=cv2.INTER_LINEAR)
    
    img_det = show_seg_result(img_merge, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)
    #cv2.imshow("image",img_det)
    #cv2.waitKey(1)
    return img_det, conf_list

