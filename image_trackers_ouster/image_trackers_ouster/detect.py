#!/usr/bin/env python3

import rclpy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
import time


# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_boxes
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector(Node):
    def __init__(self):
        super().__init__('yolo_node')
        self.declare_parameter("confidence_threshold", 0.5)
        self.conf_thres = self.get_parameter("confidence_threshold").value
        self.declare_parameter("iou_threshold", 0.01)
        self.iou_thres = self.get_parameter("iou_threshold").value
        self.declare_parameter("agnostic_nms", True)
        self.agnostic_nms = self.get_parameter("agnostic_nms").value
        self.declare_parameter("maximum_detections", 30)
        self.max_det = self.get_parameter("maximum_detections").value
        self.declare_parameter("line_thickness", 1)
        self.line_thickness = self.get_parameter("line_thickness").value
        self.declare_parameter("view_image", False)
        self.view_image = self.get_parameter("view_image").value
        # Initialize weights 
        weights = "/home/ubuntu/ros_ws/src/image_trackers_ouster/models/best.pt"
        # Initialize model
        self.declare_parameter("device", "cuda")
        self.device = select_device(self.get_parameter("device").value)
        self.declare_parameter("dnn", True)
        self.declare_parameter("data", False)
        self.model = DetectMultiBackend(weights, device=self.device, dnn=self.get_parameter("dnn").value, data=self.get_parameter("data").value)
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.declare_parameter("inference_size_w", 2048)
        self.declare_parameter("inference_size_h",128)
        self.img_size = [self.get_parameter("inference_size_w").value, 
                         self.get_parameter("inference_size_h").value]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.declare_parameter("half", False)
        self.half = self.get_parameter("half").value      
        self.half &= self.pt and self.device.type != 'cpu'

        if self.pt:
            self.model.model.half() if self.half else self.model.model.float()
        else:
            half = False
            bs = 1  # export.py models default to batch-size 1
            self.device = torch.device('cpu')
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model(torch.zeros(1, 4, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))       
        
        # Initialize subscriber to Image/CompressedImage topic
        self.declare_parameter("input_image_topic", "/ouster_merged")
        input_image_topic = self.get_parameter("input_image_topic").value
        self.declare_parameter("compressed_image", False)
        self.compressed_input = self.get_parameter("compressed_image").value
        if self.compressed_input:
            self.image_sub = self.create_subscription( CompressedImage,input_image_topic, self.image_callback, 1)
        else:
            self.image_sub = self.create_subscription( Image,input_image_topic, self.image_callback, 1)
            
        # Initialize prediction publisher
        self.declare_parameter("output_topic", "/tracker/detections")
        output_topic = self.get_parameter("output_topic").value
        self.pred_pub = self.create_publisher( BoundingBoxes,output_topic, 10)
        # Initialize image publisher
        self.declare_parameter("publish_image", True)
        self.publish_image = self.get_parameter("publish_image").value
        self.declare_parameter("output_image_topic", "/tracker/image_out")
        output_image_topic = self.get_parameter("output_image_topic").value
        if self.publish_image:
            self.image_pub = self.create_publisher( Image,output_image_topic, 10)
        
        # Initialize CV_Bridge
        self.bridge = CvBridge()

    def callback(self, data):
         
        time_1 = int(round(time.time() * 1000))
        """adapted from yolov5/detect.py"""
        if self.compressed_input:
            im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgra8")
        else:
            im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgra8")

        im, im0 = self.preprocess(im)
        
        # Run inference
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, None, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header

        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

        for i, det in enumerate(pred):  # per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            cont = 0
            for *xyxy, conf, cls in reversed(det):
                
                bounding_box = BoundingBox()
                c = int(cls)
                # Fill in bounding box message
                bounding_box.Class = self.names[c] +" "+str(cont)
                bounding_box.probability = conf 
                bounding_box.xmin = int(xyxy[0])
                bounding_box.ymin = int(xyxy[1])
                bounding_box.xmax = int(xyxy[2])
                bounding_box.ymax = int(xyxy[3])

                bounding_boxes.bounding_boxes.append(bounding_box)
                cont= cont+1

                # Annotate the image
                if self.publish_image or self.view_image:  # Add bbox to image
                      # integer class
                    label = f"{bounding_box.Class} {conf:.2f}"
                    annotator.box_label(xyxy, label, color=(255,0,0))  #=colors(c,True)     color=(255,0,0)

        # Publish prediction
        self.pred_pub.publish(bounding_boxes)

        if self.publish_image:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgra8"))        
        time_2 = int(round(time.time() * 1000))
      

    def preprocess(self, img):
        """
        Adapted from yolov5/utils/datasets.py LoadStreams class
        """
        img0 = img.copy()             
        img = letterbox(img0, 2048, stride=self.stride, auto=True)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)

        return img, img0 


def main(args=None):
    check_requirements(exclude=("tensorboard", "thop"))
    
    rclpy.init(args=args)
    detector=Yolov5Detector()
    try:
        rclpy.spin(detector)
    except KeyboardInterrupt:
        pass
    detector.destroy_node()
    rclpy.shutdown()

