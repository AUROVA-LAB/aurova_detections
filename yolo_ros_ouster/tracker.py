#!/usr/bin/env python3

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
from rostopic import get_topic_type
from copy import copy

from sensor_msgs.msg import Image, CompressedImage
from detection_msgs.msg import BoundingBox, BoundingBoxes
from threading import Thread
import termios, tty



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
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Tracker:
    def __init__(self):
        print("Starting initilization.")
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        self.flag_image=False; self.flag_select=False; self.rate=rospy.Rate(rospy.get_param("~rate", 10))
        # Initialize weights 
        weights = rospy.get_param("~weights")
        # Initialize model
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"))
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 2048), rospy.get_param("~inference_size_h",128)]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half
        self.half = rospy.get_param("~half", False)      
        self.half &= self.pt and self.device.type != 'cpu'

        if self.pt:
            self.model.model.half() if self.half else self.model.model.float()
        else:
            half = False
            bs = 1  # export.py models default to batch-size 1
            self.device = torch.device('cpu')
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model(torch.zeros(1, 4, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))

        # Set up tracker.
        params = cv2.TrackerDaSiamRPN_Params()
        params.backend=cv2.dnn.DNN_BACKEND_CUDA
        params.target=cv2.dnn.DNN_TARGET_CUDA
        params.kernel_cls1="/home/ros_ws/src/yolov5_ros/models/dasiamrpn_kernel_cls1.onnx"
        params.kernel_r1="/home/ros_ws/src/yolov5_ros/models/dasiamrpn_kernel_r1.onnx"
        params.model="/home/ros_ws/src/yolov5_ros/models/dasiamrpn_model.onnx"
        self.tracker=cv2.TrackerDaSiamRPN_create(params)
        self.augment=100
        detect_frequency=rospy.get_param("~detect_frequency", 2.0) #Frequency to use Yolo to detect the target and compare with tracker.
        self.detect_period=1.0/detect_frequency
        self.search_time=rospy.get_param("~search_time", 2.0) #Search during 2 seconds before changing mode
        self.threshold_tracker=rospy.get_param("~threshold_tracker", 0.4) #Minimum correlation to consider a detected target as a possible target.
        self.threshold_search=rospy.get_param("~threshold_search", 0.6) #More restrictive to re-identify the target during search mode
        # Define the operation modes
        self.SELECT_TARGET_MODE, self.SEARCH_MODE, self.TRACK_MODE = 0, 1, 2 #Global modes of the detector/tracker
        self.TRACKER_NORMAL, self.TRACKER_LEFT, self.TRACKER_RIGHT =  0, 1, 2 #Tracker modes for considering that the image is 360 degres.
        self.operation_mode=self.SELECT_TARGET_MODE
        self.saved_selection=[]

        #Time measure
        self.detect_t=0; self.track_t=0; self.search_t=0

        # start main thread
        self.thread = Thread(target = main_thread, args = (self, ))
        self.thread.start()       
        
        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.callback, queue_size=1
            )
        else:
            self.image_sub = rospy.Subscriber(
                input_image_topic, Image, self.callback, queue_size=1
            )

        # Initialize prediction publisher
        self.pred_pub = rospy.Publisher(
            rospy.get_param("~output_topic"), BoundingBoxes, queue_size=10
        )
        
        # Initialize image publisher
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub = rospy.Publisher(
                rospy.get_param("~output_image_topic"), Image, queue_size=10
            )
        
        # Initialize CV_Bridge
        self.bridge = CvBridge()
        print("Initilization finished.")

    def detectYolo5v (self, im):

        im, im0 = self.preprocess(im)
        # Run inference
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()

        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

        for i, det in enumerate(pred):  # per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
            
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

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
        return bounding_boxes.bounding_boxes
    
    def select_target(self, im):
        bounding_boxes = self.detectYolo5v(im.copy())
        now=rospy.get_time()
        cv2.rectangle(self.im_output, (int(self.img_size[0]*0.43), -1), (int(self.img_size[0]*0.57), self.img_size[1]+1), (0,155,155), 2, 1)
        i=0
        #Delete the old one because it wasn't detected for too much time.
        while i<len(self.saved_selection):
            if now-self.saved_selection[i]["time"]>self.saved_selection[i]["count"]+2:
                del self.saved_selection[i]
                continue
            i+=1
        for bbox in bounding_boxes:
            # #Select decision

            ### The first bounding_box that is near the robot(only debug) ###
            # if bbox.ymax-bbox.ymin<60: continue
            # self.bbox=bbox
            # # Calculate histogram of the segmented person
            # segment_frame, mask=self.segment_person(im,bbox)
            # self.target_descriptor=self.histogramPartsBody(segment_frame, mask)
            # self.operation_mode=self.TRACK_MODE; self.tracker_mode=self.TRACKER_NORMAL
            # self.tracker_start=rospy.get_time()
            # s,n,r,depth=cv2.split(im)
            # frame=cv2.merge([s,n,r])
            # self.prev_frame=frame
            # #Change to x,y,(top point)w,h(width,heigth) format for opencv
            # self.bbox_tracker=[self.bbox.xmin, self.bbox.ymin, self.bbox.xmax-self.bbox.xmin, self.bbox.ymax-self.bbox.ymin]
            # self.tracker.init(frame, self.bbox_tracker)
            # #draw bounding box
            # if self.publish_image:
            #     self.draw_rectangles((0,0,0))
            # break

            ### A person who have been in front of the robot for 5 seconds. ###
            segment_frame, mask=self.segment_person(im,bbox)
            descriptor=self.histogramPartsBody(segment_frame, mask)
            match=False
            #Filter the persons that aren't at the front or are too far away.
            if bbox.xmin>self.img_size[0]*0.43 and bbox.xmax<self.img_size[0]*0.57 and  bbox.ymax-bbox.ymin>0.33*self.img_size[1]:
                for i in range(len(self.saved_selection)):
                    if self.get_iou(bbox,self.saved_selection[i]["bbox"]) > 0.8:
                        corr=cv2.compareHist(descriptor,self.saved_selection[i]["descriptor"],cv2.HISTCMP_CORREL)
                        if  corr>self.threshold_tracker:
                            match=True
                            if now-self.saved_selection[i]["time"]>self.saved_selection[i]["count"]+1:
                                self.saved_selection[i]["count"]+=1
                                if self.saved_selection[i]["count"]==5:   
                                    self.bbox=bbox; self.target_descriptor=descriptor         
                                    self.operation_mode=self.TRACK_MODE; self.tracker_mode=self.TRACKER_NORMAL
                                    self.tracker_start=rospy.get_time()
                                    s,n,r,depth=cv2.split(im)
                                    frame=cv2.merge([s,n,r])
                                    self.prev_frame=frame
                                    #Change to x,y,(top point)w,h(width,heigth) format for opencv
                                    self.bbox_tracker=[self.bbox.xmin, self.bbox.ymin, self.bbox.xmax-self.bbox.xmin, self.bbox.ymax-self.bbox.ymin]
                                    self.tracker.init(frame, self.bbox_tracker)
                                    #draw bounding box
                                    if self.publish_image:
                                        self.draw_rectangles((0,0,0))
                                    self.saved_selection=[]
                            break
            
                if not match:
                    thisdict = {"bbox":bbox, "descriptor":descriptor, "time":now, "count":0}
                    self.saved_selection.append(thisdict)

    def search_target(self, im):
        start=rospy.get_time()
        best_corr=0
        bounding_boxes = self.detectYolo5v(im.copy())
        for bbox in bounding_boxes:
            segment_frame, mask=self.segment_person(im,bbox)
            descriptor=self.histogramPartsBody(segment_frame, mask)
            corr=cv2.compareHist(self.target_descriptor,descriptor,cv2.HISTCMP_CORREL)
            if corr>self.threshold_search and corr>best_corr:
                best_corr=corr; best_bbox=bbox; best_descriptor=descriptor
        s,n,r,depth=cv2.split(im)
        frame=cv2.merge([s,n,r])
        if best_corr>0:
            self.bbox=best_bbox
            #Change to x,y,(top point)w,h(width,heigth) format for opencv
            self.bbox_tracker=[self.bbox.xmin, self.bbox.ymin, self.bbox.xmax-self.bbox.xmin, self.bbox.ymax-self.bbox.ymin]
            self.tracker.init(frame, self.bbox_tracker); self.prev_frame=frame
            self.tracker_mode=self.TRACKER_NORMAL; self.operation_mode=self.TRACK_MODE
            self.tracker_start=rospy.get_time()
            if self.publish_image:
                self.draw_rectangles((0,255,0))
            # Update descriptor
            self.target_descriptor=self.target_descriptor*0.9+best_descriptor*0.1
        elif rospy.get_time()-self.search_start>self.search_time: self.operation_mode=self.SELECT_TARGET_MODE
        end=rospy.get_time()
        self.search_t=end-start if self.search_t==0 else (self.search_t+end-start)/2

    def track_target(self, im):
        if rospy.get_time()-self.tracker_start>self.detect_period:
            start=rospy.get_time()
            best_corr=0
            bounding_boxes = self.detectYolo5v(im.copy())
            for bbox in bounding_boxes:
                segment_frame, mask=self.segment_person(im,bbox)
                descriptor=self.histogramPartsBody(segment_frame, mask)
                corr=cv2.compareHist(self.target_descriptor,descriptor,cv2.HISTCMP_CORREL)
                if corr>self.threshold_tracker:
                    #Use the percentage of are in the intersection of the bounding boxes.
                    iou=self.get_iou(bbox,self.bbox)
                    corr+=iou/2
                    if iou>0 and corr>best_corr:
                        best_corr=corr; best_bbox=bbox; best_descriptor=descriptor
            s,n,r,depth=cv2.split(im)
            frame=cv2.merge([s,n,r])
            if best_corr>0:
                self.bbox=best_bbox
                #Change to x,y,(top left point)w,h(width,heigth) format for opencv
                self.bbox_tracker=[self.bbox.xmin, self.bbox.ymin, self.bbox.xmax-self.bbox.xmin, self.bbox.ymax-self.bbox.ymin]
                self.tracker.init(frame, self.bbox_tracker); self.prev_frame=frame
                self.tracker_mode=self.TRACKER_NORMAL; self.tracker_start=rospy.get_time()
                if self.publish_image:
                    self.draw_rectangles((0,0,255))
                # Update descriptor
                self.target_descriptor=self.target_descriptor*0.9+best_descriptor*0.1
            else: self.operation_mode=self.SEARCH_MODE; self.search_start=rospy.get_time()
            end=rospy.get_time()
            self.detect_t=end-start if self.detect_t==0 else (self.detect_t+end-start)/2

        else:
            start=rospy.get_time()        
            s,n,r,depth=cv2.split(im)
            frame=cv2.merge([s,n,r])
            copy_frame=frame.copy()
            # Taking account that the image is 360 degrees.
            if self.bbox.xmin<self.augment:
                copy_frame[:,:self.augment]=frame[:,-self.augment:]
                copy_frame[:,self.augment:]=frame[:,:-self.augment]
                if self.tracker_mode!=self.TRACKER_LEFT:
                    prev_copy=self.prev_frame.copy()
                    prev_copy[:,:self.augment]=self.prev_frame[:,-self.augment:]
                    prev_copy[:,self.augment:]=self.prev_frame[:,:-self.augment]
                    self.bbox_tracker[0]+=self.augment
                    self.tracker.init(prev_copy,self.bbox_tracker)
                self.tracker_mode=self.TRACKER_LEFT

            elif self.bbox.xmax>self.img_size[0]-self.augment:
                copy_frame[:,-self.augment:]=frame[:,:self.augment]
                copy_frame[:,:-self.augment]=frame[:,self.augment:]
                if self.tracker_mode!=self.TRACKER_RIGHT:
                    prev_copy=self.prev_frame.copy()
                    prev_copy[:,-self.augment:]=self.prev_frame[:,:self.augment]
                    prev_copy[:,:-self.augment]=self.prev_frame[:,self.augment:]
                    self.bbox_tracker[0]-=self.augment
                    self.tracker.init(prev_copy,self.bbox_tracker)   
                self.tracker_mode=self.TRACKER_RIGHT

            elif self.tracker_mode!=self.TRACKER_NORMAL:
                tracker_mode=self.TRACKER_NORMAL
                self.tracker.init(self.prev_frame,self.bbox_tracker)

            # timer = cv2.getTickCount()
            ok, self.bbox_tracker = self.tracker.update(copy_frame)
            self.bbox_tracker=list(self.bbox_tracker)

            # Bounding box in the original image
            if self.tracker_mode==self.TRACKER_LEFT: self.bbox_tracker[0]-=self.augment
            elif self.tracker_mode==self.TRACKER_RIGHT: self.bbox_tracker[0]+=self.augment

            # Change the side of the image if necessary
            if self.bbox_tracker[0]+self.bbox_tracker[2]/2<0: self.bbox_tracker[0]+=2048
            elif self.bbox_tracker[0]+self.bbox_tracker[2]/2>self.img_size[0]: self.bbox_tracker[0]-=self.img_size[0]

            #Change to x,y,(top left point) x,y (bottom right point)
            self.bbox.xmin=self.bbox_tracker[0]; self.bbox.ymin=self.bbox_tracker[1]
            self.bbox.xmax=self.bbox_tracker[0]+self.bbox_tracker[2]; self.bbox.ymax=self.bbox_tracker[1]+self.bbox_tracker[3]

            # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
            # print(fps)
            # Draw bounding box
            self.prev_frame=frame
            if ok and self.publish_image:
                self.draw_rectangles()
            
            end=rospy.get_time()
            self.track_t=end-start if self.track_t==0 else (self.track_t+end-start)/2
               

    def callback(self, data):
        self.data = data

        if self.compressed_input:
            self.im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgra8")
        else:
            self.im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgra8")
        self.flag_image=True
        

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
    
    def draw_rectangles(self, color=(255,0,0)):
        self.bbox.ymin=max(0,self.bbox.ymin); self.bbox.ymax=min(self.img_size[1],self.bbox.ymax)
        if self.bbox.xmin<0:
            if self.bbox.xmax<0:
                aux=copy(self.bbox); aux.xmin+=self.img_size[0]; aux.xmax+=self.img_size[0]
                self.bounding_boxes.append(aux)
                cv2.rectangle(self.im_output, (aux.xmin, aux.ymin), (aux.xmax, aux.ymax), color, 2, 1)
            else:
                aux=copy(self.bbox); aux.xmin+=self.img_size[0]; aux.xmax=self.img_size[0]
                self.bounding_boxes.append(aux)
                aux2=copy(self.bbox); aux2.xmin=0
                self.bounding_boxes.append(aux2)
                cv2.rectangle(self.im_output, (aux.xmin, aux.ymin), (aux.xmax, aux.ymax), color, 2, 1)
                cv2.rectangle(self.im_output, (aux2.xmin, aux2.ymin), (aux2.xmax, aux2.ymax), color, 2, 1)
        elif self.bbox.xmax>self.img_size[0]:
            if self.bbox.xmin>self.img_size[0]:
                aux=copy(self.bbox); aux.xmin-=self.img_size[0]; aux.xmax-=self.img_size[0]
                self.bounding_boxes.append(aux)
                cv2.rectangle(self.im_output, (aux.xmin, aux.ymin), (aux.xmax, aux.ymax), color, 2, 1)
            else:
                aux=copy(self.bbox); aux.xmin=0; aux.xmax-=self.img_size[0]
                self.bounding_boxes.append(aux)
                aux2=copy(self.bbox); aux2.xmax=self.img_size[0]
                self.bounding_boxes.append(aux2)
                cv2.rectangle(self.im_output, (aux.xmin, aux.ymin), (aux.xmax, aux.ymax), color, 2, 1)
                cv2.rectangle(self.im_output, (aux2.xmin, aux2.ymin), (aux2.xmax, aux2.ymax), color, 2, 1)
        else:
            self.bounding_boxes.append(self.bbox)
            cv2.rectangle(self.im_output, (self.bbox.xmin, self.bbox.ymin), (self.bbox.xmax, self.bbox.ymax), color, 2, 1)

    def segment_person(self, frame, bbox):
        segment_frame=frame[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax]
        shape = np.shape(segment_frame)
        #Interior median depth
        depth=[]
        for i in range(int(shape[0]*0.3), int(shape[0]*0.7)):
            for j in range(int(shape[1]*0.3), int(shape[1]*0.7)):
                depth.append(segment_frame[i,j,3])
        depth.sort()
        if len(depth)==0: return segment_frame,None

        if len(depth)==1: median=depth[0]
        elif len(depth)%2==0:
            median=(depth[int(len(depth)/2)] + depth[int(len(depth)/2)+1])/2
        else:
            median=depth[int(len(depth)/2)]

        #Segment using depth
        mask=np.zeros(shape[:2], dtype="uint8")
        for i in range(shape[0]):
            for j in range(shape[1]):
                if segment_frame[i,j,3]==median: mask[i,j]=255

        return segment_frame, mask
    
    def histogramPartsBody(self, segment_frame, mask):
        #Divide the person in 5 equal parts
        shape = np.shape(segment_frame)
        descriptor=[]
        part_mask=None
        for i in range(5):
            part_frame = segment_frame[int(0.2*i*shape[0]):int(0.2*(i+1)*shape[0])]
            if mask is not None:
                part_mask = mask[int(0.2*i*shape[0]):int(0.2*(i+1)*shape[0])]
            for channel in range(3):
                hist=cv2.calcHist([part_frame], [channel], histSize=[32], mask=part_mask, ranges=[0,256])
                cv2.normalize(hist,hist,0.0,1.0,cv2.NORM_MINMAX,dtype=cv2.CV_32F)
                descriptor.append(hist)
        descriptor=np.array(descriptor); descriptor=descriptor.flatten()
        return descriptor
    
    def get_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.
        """
        # determine the coordinates of the intersection rectangle
        x_left = max(bb1.xmin, bb2.xmin)
        y_top = max(bb1.ymin, bb2.ymin)
        x_right = min(bb1.xmax, bb2.xmax)
        y_bottom = min(bb1.ymax, bb2.ymax)

        if x_right < x_left or y_bottom < y_top:
            return 0.0
        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1.xmax - bb1.xmin) * (bb1.ymax - bb1.ymin)
        bb2_area = (bb2.xmax - bb2.xmin) * (bb2.ymax - bb2.ymin)

        # compute the intersection over union by taking the intersection
        # area and dividing it by the areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def keyboard_callback(self, input):
        # print("Input: ", input)
        if input=='s':
            self.flag_select=True
        elif input=='t':
            print("Tracker mean time: ", self.track_t)
            print("Detect mean time: ", self.detect_t)
            print("Search mean time: ", self.search_t)

# Class for reading keyboard input
class KeyboardThread(Thread):

    def __init__(self, input_cbk = None, name='keyboard-input-thread'):
        self.input_cbk = input_cbk
        super(KeyboardThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            self.input_cbk(self.getch())
    
    def getch(self):
        fd = sys.stdin.fileno()
        orig = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)  # or tty.setraw(fd) if you prefer raw mode's behavior.
            return sys.stdin.read(1)
        finally:
            termios.tcsetattr(fd, termios.TCSAFLUSH, orig)

def main_thread(node: Yolov5Tracker):
  while not rospy.is_shutdown():
    if node.flag_select: node.operation_mode=node.SELECT_TARGET_MODE; node.flag_select=False
    if node.flag_image:
        node.bounding_boxes=[]
        node.flag_image=False
        if node.publish_image: node.im_output=node.im.copy()
        if node.operation_mode==node.SELECT_TARGET_MODE: node.select_target(node.im.copy())
        elif node.operation_mode==node.SEARCH_MODE: node.search_target(node.im.copy())
        elif node.operation_mode==node.TRACK_MODE: node.track_target(node.im.copy())

        #Publish messages
        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = node.data.header
        bounding_boxes.image_header = node.data.header
        for bbox in node.bounding_boxes:
            bounding_boxes.bounding_boxes.append(bbox) #Append the bounding boxes which are between the limits of the image.
        node.pred_pub.publish(bounding_boxes)
        if node.publish_image:
            node.image_pub.publish(node.bridge.cv2_to_imgmsg(node.im_output, "bgra8")) 
      
    node.rate.sleep()


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop", "opencv-python"))
    

    rospy.init_node("tracker", anonymous=True)
    tracker_node = Yolov5Tracker()
    kthread = KeyboardThread(tracker_node.keyboard_callback)    
   
    rospy.spin()