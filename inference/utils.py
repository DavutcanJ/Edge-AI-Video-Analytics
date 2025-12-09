"""
Utility functions for inference pipeline.
"""

import cv2
import numpy as np
from typing import Tuple, List
import torch


def letterbox(
    img: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scale_fill: bool = False,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize and pad image while maintaining aspect ratio.
    
    Args:
        img: Input image
        new_shape: Target shape
        color: Padding color
        auto: Minimum rectangle
        scale_fill: Stretch to exact size
        scaleup: Allow upscaling
        stride: Stride for padding
    
    Returns:
        Tuple of (resized_image, ratio, padding)
    """
    shape = img.shape[:2]  # current shape [height, width]
    
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down
        r = min(r, 1.0)
    
    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    
    dw /= 2
    dh /= 2
    
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return img, ratio, (dw, dh)


def scale_boxes(
    img1_shape: Tuple[int, int],
    boxes: np.ndarray,
    img0_shape: Tuple[int, int],
    ratio_pad: Tuple[Tuple[float, float], Tuple[float, float]] = None
) -> np.ndarray:
    """
    Rescale boxes from img1_shape to img0_shape.
    
    Args:
        img1_shape: Shape boxes were computed on
        boxes: Boxes in x1y1x2y2 format
        img0_shape: Target shape
        ratio_pad: Ratio and padding used in letterbox
    
    Returns:
        Scaled boxes
    """
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= gain
    
    # Clip to image bounds
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, img0_shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, img0_shape[0])
    
    return boxes


def clip_boxes(boxes: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    """
    Clip bounding boxes to image shape.
    
    Args:
        boxes: Boxes in x1y1x2y2 format
        shape: Image shape (height, width)
    
    Returns:
        Clipped boxes
    """
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])
    return boxes


def xyxy2xywh(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from x1y1x2y2 to xywh format."""
    y = np.copy(boxes)
    y[:, 0] = (boxes[:, 0] + boxes[:, 2]) / 2  # x center
    y[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2  # y center
    y[:, 2] = boxes[:, 2] - boxes[:, 0]  # width
    y[:, 3] = boxes[:, 3] - boxes[:, 1]  # height
    return y


def xywh2xyxy(boxes: np.ndarray) -> np.ndarray:
    """Convert boxes from xywh to x1y1x2y2 format."""
    y = np.copy(boxes)
    y[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    y[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    y[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    y[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2
    return y


def non_max_suppression(
    prediction: np.ndarray,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300
) -> List[np.ndarray]:
    """
    Non-Maximum Suppression (NMS) on inference results.
    
    Args:
        prediction: Model predictions [batch, num_boxes, 5 + num_classes]
        conf_thres: Confidence threshold
        iou_thres: IoU threshold for NMS
        max_det: Maximum detections per image
    
    Returns:
        List of detections per image [num_dets, 6] (x1, y1, x2, y2, conf, cls)
    """
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    
    # Settings
    max_wh = 7680  # maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    
    output = [np.zeros((0, 6))] * prediction.shape[0]
    
    for xi, x in enumerate(prediction):  # image index, image inference
        x = x[xc[xi]]  # confidence filter
        
        if not x.shape[0]:
            continue
        
        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Detections matrix nx6 (xyxy, conf, cls)
        conf = x[:, 5:].max(1, keepdims=True)
        j = x[:, 5:].argmax(1).reshape(-1, 1)
        x = np.concatenate((box, conf, j.astype(np.float32)), 1)[conf.reshape(-1) > conf_thres]
        
        # Check shape
        n = x.shape[0]
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort()[::-1][:max_nms]]  # sort by confidence
        
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        
        # Custom NMS
        i = custom_nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:
            i = i[:max_det]
        
        output[xi] = x[i]
    
    return output


def custom_nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Custom NMS implementation."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        
        intersection = w * h
        union = areas[i] + areas[order[1:]] - intersection
        iou = intersection / union
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return np.array(keep)


def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: List[str] = None,
    scores: np.ndarray = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        img: Input image
        boxes: Boxes in x1y1x2y2 format
        labels: List of labels
        scores: Confidence scores
        color: Box color
        thickness: Line thickness
    
    Returns:
        Image with drawn boxes
    """
    img = img.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        if labels is not None or scores is not None:
            label_text = ""
            if labels is not None:
                label_text += labels[i]
            if scores is not None:
                label_text += f" {scores[i]:.2f}"
            
            (label_w, label_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)
            cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(img, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness)
    
    return img
