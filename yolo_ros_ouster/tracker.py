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
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Tracker_Base:
    def __init__(self):
        self.bbox=BoundingBox(); self.bb_msg=[]
        self.net=None; self.covariance=None
        self.img_size=None

    def draw_rectangles(self,im_output, color=(255,0,0)):
        self.bbox.ymin=max(0,self.bbox.ymin); self.bbox.ymax=min(self.img_size[1],self.bbox.ymax)
        if self.bbox.xmin<0:
            if self.bbox.xmax<0:
                aux=copy(self.bbox); aux.xmin+=self.img_size[0]; aux.xmax+=self.img_size[0]
                aux.xmin=int(aux.xmin);aux.xmax=int(aux.xmax);aux.ymin=int(aux.ymin);aux.ymax=int(aux.ymax)
                self.bb_msg.append(aux)
                cv2.rectangle(im_output, (aux.xmin, aux.ymin), (aux.xmax, aux.ymax), color, 2, 1)
            else:
                aux=copy(self.bbox); aux.xmin+=self.img_size[0]; aux.xmax=self.img_size[0]
                aux.xmin=int(aux.xmin);aux.xmax=int(aux.xmax);aux.ymin=int(aux.ymin);aux.ymax=int(aux.ymax)
                self.bb_msg.append(aux)
                aux2=copy(self.bbox); aux2.xmin=0
                aux2.xmin=int(aux2.xmin);aux2.xmax=int(aux2.xmax);aux2.ymin=int(aux2.ymin);aux2.ymax=int(aux2.ymax)
                self.bb_msg.append(aux2)
                cv2.rectangle(im_output, (aux.xmin, aux.ymin), (aux.xmax, aux.ymax), color, 2, 1)
                cv2.rectangle(im_output, (aux2.xmin, aux2.ymin), (aux2.xmax, aux2.ymax), color, 2, 1)
        elif self.bbox.xmax>self.img_size[0]:
            if self.bbox.xmin>self.img_size[0]:
                aux=copy(self.bbox); aux.xmin-=self.img_size[0]; aux.xmax-=self.img_size[0]
                aux.xmin=int(aux.xmin);aux.xmax=int(aux.xmax);aux.ymin=int(aux.ymin);aux.ymax=int(aux.ymax)
                self.bb_msg.append(aux)
                cv2.rectangle(im_output, (aux.xmin, aux.ymin), (aux.xmax, aux.ymax), color, 2, 1)
            else:
                aux=copy(self.bbox); aux.xmin=0; aux.xmax-=self.img_size[0]
                aux.xmin=int(aux.xmin);aux.xmax=int(aux.xmax);aux.ymin=int(aux.ymin);aux.ymax=int(aux.ymax)
                self.bb_msg.append(aux)
                aux2=copy(self.bbox); aux2.xmax=self.img_size[0]
                aux2.xmin=int(aux2.xmin);aux2.xmax=int(aux2.xmax);aux2.ymin=int(aux2.ymin);aux2.ymax=int(aux2.ymax)
                self.bb_msg.append(aux2)
                cv2.rectangle(im_output, (aux.xmin, aux.ymin), (aux.xmax, aux.ymax), color, 2, 1)
                cv2.rectangle(im_output, (aux2.xmin, aux2.ymin), (aux2.xmax, aux2.ymax), color, 2, 1)
        else:
            aux=copy(self.bbox)
            aux.xmin=int(aux.xmin);aux.xmax=int(aux.xmax);aux.ymin=int(aux.ymin);aux.ymax=int(aux.ymax)
            self.bb_msg.append(aux)
            cv2.rectangle(im_output, (self.bbox.xmin, self.bbox.ymin), (self.bbox.xmax, self.bbox.ymax), color, 2, 1)

class FusionTracker:
    def __init__(self):
        self.yolo_tracker=Tracker_Base(); self.dasiam_tracker=Tracker_Base()
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
        self.yolo_tracker.net = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"))
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.yolo_tracker.net.stride,
            self.yolo_tracker.net.names,
            self.yolo_tracker.net.pt,
            self.yolo_tracker.net.jit,
            self.yolo_tracker.net.onnx,
            self.yolo_tracker.net.engine,
        )

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 2048), rospy.get_param("~inference_size_h",128)]
        self.img_size = check_img_size(self.img_size, s=self.stride)
        self.yolo_tracker.img_size=self.img_size; self.dasiam_tracker.img_size=self.img_size

        # Half
        self.half = rospy.get_param("~half", False)      
        self.half &= self.pt and self.device.type != 'cpu'

        if self.pt:
            self.yolo_tracker.net.half() if self.half else self.yolo_tracker.net.float()
        else:
            half = False
            bs = 1  # export.py models default to batch-size 1
            self.device = torch.device('cpu')
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.yolo_tracker.net(torch.zeros(1, 4, *self.img_size).to(self.device).type_as(next(self.yolo_tracker.net.parameters())))

        # Set up tracker.
        params = cv2.TrackerDaSiamRPN_Params()
        params.backend=cv2.dnn.DNN_BACKEND_CUDA
        params.target=cv2.dnn.DNN_TARGET_CUDA
        params.kernel_cls1="/home/ros_ws/src/yolov5_ros/models/dasiamrpn_kernel_cls1.onnx"
        params.kernel_r1="/home/ros_ws/src/yolov5_ros/models/dasiamrpn_kernel_r1.onnx"
        params.model="/home/ros_ws/src/yolov5_ros/models/dasiamrpn_model.onnx"
        self.dasiam_tracker.net=cv2.TrackerDaSiamRPN_create(params)
        self.augment=100
        detect_frequency=rospy.get_param("~detect_frequency", 2.0) #Frequency to use Yolo to detect the target and compare with tracker.
        self.detect_period=1.0/detect_frequency
        self.search_time=rospy.get_param("~search_time", 2.0) #Search during 2 seconds before changing mode
        self.threshold_tracker=rospy.get_param("~threshold_tracker", 0.4) #Minimum correlation to consider a detected target as a possible target.
        # self.threshold_search=rospy.get_param("~threshold_search", 0.6) #More restrictive to re-identify the target during search mode
        self.aug_per=rospy.get_param("~search_augmentation_percentage", 0.1) #More restrictive to re-identify the target during search mode
        #If the target is lost by YOLO and the covariance of the ekf filter is greater than this value, chanbe to select mode.
        self.limit_covariance=rospy.get_param("~limit_covariance", 5.0) 

        self.output_video_dir=rospy.get_param("~output_video_dir", 0.1) #More restrictive to re-identify the target during search mode
        # Define the operation modes
        self.SELECT_TARGET_MODE, self.SEARCH_MODE, self.TRACK_MODE = 0, 1, 2 #Global modes of the detector/tracker
        self.TRACKER_NORMAL, self.TRACKER_LEFT, self.TRACKER_RIGHT =  0, 1, 2 #Tracker modes for considering that the image is 360 degres.
        self.operation_mode=self.SELECT_TARGET_MODE
        self.saved_selection=[]
        self.search_area=BoundingBox()
        self.search_area.xmin=0; self.search_area.xmax=self.img_size[0]
        self.search_area.ymin=0; self.search_area.ymax=self.img_size[1]
        self.target_covariance=0.0
        #Time measure
        self.detect_t=0; self.track_t=0; self.search_t=0

        # start main thread
        self.thread = Thread(target = main_thread, args = (self, ))
        self.thread.start()     
        self.reset_dasiam=False  
        
        # Initialize subscriber to Image/CompressedImage topic
        input_image_type, input_image_topic, _ = get_topic_type(rospy.get_param("~input_image_topic"), blocking = True)
        search_topic=rospy.get_param("~search_topic")
        self.compressed_input = input_image_type == "sensor_msgs/CompressedImage"

        if self.compressed_input:
            self.image_sub = rospy.Subscriber(
                input_image_topic, CompressedImage, self.image_callback, queue_size=1
            )
        else:
            self.image_sub = rospy.Subscriber(
                input_image_topic, Image, self.image_callback, queue_size=1
            )
        self.search_sub=rospy.Subscriber(search_topic, BoundingBoxes, self.search_area_callback, queue_size=1)
        # Initialize prediction publisher
        self.dasiam_pub = rospy.Publisher(
            rospy.get_param("~output_topic_dasiamrpn"), BoundingBoxes, queue_size=1
        )
        self.yolo_pub = rospy.Publisher(
            rospy.get_param("~output_topic_yolo"), BoundingBoxes, queue_size=1
        )
        
        # Initialize image publisher
        self.image_pub = rospy.Publisher(rospy.get_param("~output_image_topic"), Image, queue_size=10)
        
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

        pred = self.yolo_tracker.net(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        ### To-do move pred to CPU and fill BoundingBox messages
        
        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()


        for i, det in enumerate(pred):  # per image

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            
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


        # Publish prediction
        return bounding_boxes.bounding_boxes
    
    def select_mode(self, im):
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
            ### A person who have been in front of the robot for 3 seconds. ###
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
                                if self.saved_selection[i]["count"]==3:   
                                    self.yolo_tracker.bbox=bbox; self.target_descriptor=descriptor 
                                    self.dasiam_tracker.bbox=bbox        
                                    self.operation_mode=self.TRACK_MODE; self.tracker_mode=self.TRACKER_NORMAL
                                    self.tracker_start=rospy.get_time()
                                    s,n,r,depth=cv2.split(im)
                                    frame=cv2.merge([s,n,r])
                                    self.prev_frame=frame; self.search_start=rospy.get_time()
                                    #Change to x,y,(top point)w,h(width,heigth) format for opencv
                                    self.dasiam_tracker.bbxywh =[bbox.xmin, bbox.ymin, bbox.xmax-bbox.xmin, bbox.ymax-bbox.ymin]
                                    self.dasiam_tracker.net.init(frame, self.dasiam_tracker.bbxywh)
                                    #draw bounding box
                                    self.yolo_tracker.draw_rectangles(self.im_output,(0,0,0))
                                    self.yolo_tracker.covariance=0.1 #The first detection has low covariance.
                                    self.saved_selection=[]
                            break
            
                if not match:
                    thisdict = {"bbox":bbox, "descriptor":descriptor, "time":now, "count":0}
                    self.saved_selection.append(thisdict)

    # def search_mode(self, im):
    #     start=rospy.get_time()
    #     best_corr=0
    #     bounding_boxes = self.detectYolo5v(im.copy())
    #     for bbox in bounding_boxes:
    #         if self.get_iou(bbox,self.search_area)>0:
    #             segment_frame, mask=self.segment_person(im,bbox)
    #             descriptor=self.histogramPartsBody(segment_frame, mask)
    #             corr=cv2.compareHist(self.target_descriptor,descriptor,cv2.HISTCMP_CORREL)
    #             # cv2.rectangle(self.im_output, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0,0,0), 2, 1)
    #             # cv2.putText(self.im_output, f'{corr:.2f}',(bbox.xmin, bbox.ymin+10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
    #             if corr>self.threshold_search and corr>best_corr:
    #                 best_corr=corr; best_bbox=bbox; best_descriptor=descriptor
    #     end=rospy.get_time()
    #     if best_corr>0:
    #         self.bbox=best_bbox
    #         s,n,r,depth=cv2.split(im)
    #         frame=cv2.merge([s,n,r])
    #         #Change to x,y,(top point)w,h(width,heigth) format for opencv
    #         self.bbox_tracker=[self.bbox.xmin, self.bbox.ymin, self.bbox.xmax-self.bbox.xmin, self.bbox.ymax-self.bbox.ymin]
    #         self.dasiam_tracker.net.init(frame, self.bbox_tracker); self.prev_frame=frame
    #         self.tracker_mode=self.TRACKER_NORMAL; self.operation_mode=self.TRACK_MODE
    #         self.tracker_start=rospy.get_time()
    #         self.draw_rectangles(best_bbox,self.bounding_boxes_yolo,(0,255,0))
    #         # Update descriptor
    #         self.target_descriptor=self.target_descriptor*0.5+best_descriptor*0.5
    #     elif rospy.get_time()-self.tracker_start<self.detect_period: self.track_target(im); self.increase_search()
    #     elif rospy.get_time()-self.search_start>self.search_time: self.operation_mode=self.SELECT_TARGET_MODE
    #     self.search_t=end-start if self.search_t==0 else (self.search_t+end-start)/2

    def track_yolo(self, im):
        # if rospy.get_time()-self.tracker_start>self.detect_period:
            start=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
            best_corr=0
            bounding_boxes = self.detectYolo5v(im.copy())
            candidate=None; best_corr2=0
            for bbox in bounding_boxes:
                iou=self.get_iou(bbox,self.search_area)
                # cv2.rectangle(self.im_output, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0,0,0), 2, 1)
                if iou>0:
                    segment_frame, mask=self.segment_person(im,bbox)
                    descriptor=self.histogramPartsBody(segment_frame, mask)
                    corr=cv2.compareHist(self.target_descriptor,descriptor,cv2.HISTCMP_CORREL)
                    # cv2.rectangle(self.im_output, (bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax), (0,0,0), 2, 1)
                    # cv2.putText(self.im_output, f'{corr:.2f}',(bbox.xmin, bbox.ymin+10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255))
                    if corr>self.threshold_tracker:
                        #Use the percentage of are in the intersection of the bounding boxes.
                        corr+=iou/2
                        if corr>best_corr:
                            best_corr=corr; best_bbox=bbox; best_descriptor=descriptor
                    elif corr>best_corr2: candidate=bbox; best_corr2=corr
            
            if best_corr>0:
                self.search_start=rospy.get_time(); self.tracker_start=rospy.get_time()
                #Use the intersection with the previous bounding box to calculate the covariance.
                #We use the actuar target covariance because a higher covariance in the target will 
                #produce a higher search area and therefore more desviation.
                self.yolo_tracker.covariance=self.target_covariance*(1.0-self.get_iou(best_bbox,self.yolo_tracker.bbox))+0.3
                self.yolo_tracker.bbox=best_bbox
                self.yolo_tracker.draw_rectangles(self.im_output,(0,0,255))
                # Update descriptor
                self.target_descriptor=self.target_descriptor*0.5+best_descriptor*0.5
                if self.reset_dasiam or self.get_iou(self.yolo_tracker.bbox,self.dasiam_tracker.bbox)<0.25:
                    self.dasiam_tracker.bbox=best_bbox
                    s,n,r,depth=cv2.split(im)
                    frame=cv2.merge([s,n,r])
                    #Change to x,y,(top left point)w,h(width,heigth) format for opencv
                    self.dasiam_tracker.bbxywh=[best_bbox.xmin, best_bbox.ymin, best_bbox.xmax-best_bbox.xmin, best_bbox.ymax-best_bbox.ymin]
                    self.dasiam_tracker.net.init(frame, self.dasiam_tracker.bbxywh); self.prev_frame=frame
                    self.tracker_mode=self.TRACKER_NORMAL; self.reset_dasiam=False
                    

            elif rospy.get_time()-self.search_start>self.search_time and candidate is not None:
                if self.target_covariance<self.limit_covariance:
                    self.search_start=rospy.get_time(); self.tracker_start=rospy.get_time()
                    self.yolo_tracker.bbox=candidate
                    self.yolo_tracker.covariance=self.target_covariance
                    self.yolo_tracker.draw_rectangles(self.im_output,(0,0,255))
                    # New descriptor
                    segment_frame, mask=self.segment_person(im,candidate)
                    self.target_descriptor=self.histogramPartsBody(segment_frame, mask)
                else:
                    #Select mode
                    self.operation_mode=self.SELECT_TARGET_MODE
            end=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)  
            self.detect_t=end-start if self.detect_t==0 else (self.detect_t+end-start)/2



    def track_dasiam(self, im,):
        start=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)     
        s,n,r,depth=cv2.split(im)
        frame=cv2.merge([s,n,r])
        copy_frame=frame.copy()
        # Taking account that the image is 360 degrees.
        search_area=copy(self.search_area)
        if self.dasiam_tracker.bbox.xmin<self.augment:
            copy_frame[:,:self.augment]=frame[:,-self.augment:]
            copy_frame[:,self.augment:]=frame[:,:-self.augment]
            search_area.xmin+=self.augment; search_area.xmax+=self.augment
            if self.tracker_mode!=self.TRACKER_LEFT:
                prev_copy=self.prev_frame.copy()
                prev_copy[:,:self.augment]=self.prev_frame[:,-self.augment:]
                prev_copy[:,self.augment:]=self.prev_frame[:,:-self.augment]
                self.dasiam_tracker.bbxywh[0]+=self.augment
                self.dasiam_tracker.net.init(prev_copy,self.dasiam_tracker.bbxywh)
            self.tracker_mode=self.TRACKER_LEFT

        elif self.dasiam_tracker.bbox.xmax>self.img_size[0]-self.augment:
            copy_frame[:,-self.augment:]=frame[:,:self.augment]
            copy_frame[:,:-self.augment]=frame[:,self.augment:]
            search_area.xmin-=self.augment; search_area.xmax-=self.augment
            if self.tracker_mode!=self.TRACKER_RIGHT:
                prev_copy=self.prev_frame.copy()
                prev_copy[:,-self.augment:]=self.prev_frame[:,:self.augment]
                prev_copy[:,:-self.augment]=self.prev_frame[:,self.augment:]
                self.dasiam_tracker.bbxywh[0]-=self.augment
                self.dasiam_tracker.net.init(prev_copy,self.dasiam_tracker.bbxywh)   
            self.tracker_mode=self.TRACKER_RIGHT

        elif self.tracker_mode!=self.TRACKER_NORMAL:
            self.tracker_mode=self.TRACKER_NORMAL
            self.dasiam_tracker.net.init(self.prev_frame,self.dasiam_tracker.bbxywh)

        ok, self.dasiam_tracker.bbxywh = self.dasiam_tracker.net.update(copy_frame)
        self.dasiam_tracker.bbxywh=list(self.dasiam_tracker.bbxywh)

        # Bounding box in the original image
        if self.tracker_mode==self.TRACKER_LEFT: self.dasiam_tracker.bbxywh[0]-=self.augment
        elif self.tracker_mode==self.TRACKER_RIGHT: self.dasiam_tracker.bbxywh[0]+=self.augment

        # Change the side of the image if necessary
        if self.dasiam_tracker.bbxywh[0]+self.dasiam_tracker.bbxywh[2]/2<0: self.dasiam_tracker.bbxywh[0]+=2048
        elif self.dasiam_tracker.bbxywh[0]+self.dasiam_tracker.bbxywh[2]/2>self.img_size[0]: self.dasiam_tracker.bbxywh[0]-=self.img_size[0]

        #Change to x,y,(top left point) x,y (bottom right point)
        prev_bbox=copy(self.dasiam_tracker.bbox)
        self.dasiam_tracker.bbox.xmin=self.dasiam_tracker.bbxywh[0]; self.dasiam_tracker.bbox.ymin=self.dasiam_tracker.bbxywh[1]
        self.dasiam_tracker.bbox.xmax=self.dasiam_tracker.bbxywh[0]+self.dasiam_tracker.bbxywh[2]; 
        self.dasiam_tracker.bbox.ymax=self.dasiam_tracker.bbxywh[1]+self.dasiam_tracker.bbxywh[3]

        #Possible lost the target, update with Yolo if possible
        if self.get_iou(self.dasiam_tracker.bbox,self.search_area)==0: self.reset_dasiam=True


        # Draw bounding box
        self.prev_frame=frame
        self.dasiam_tracker.draw_rectangles(self.im_output)
        self.dasiam_tracker.covariance=self.target_covariance*(1.0-self.get_iou(prev_bbox,self.dasiam_tracker.bbox))+0.3
        
        end=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
        self.track_t=end-start if self.track_t==0 else (self.track_t+end-start)/2
               

    def image_callback(self, data):
        self.data = data

        if self.compressed_input:
            self.im = self.bridge.compressed_imgmsg_to_cv2(data, desired_encoding="bgra8")
        else:
            self.im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgra8")
        self.flag_image=True

    def search_area_callback(self, data):
        self.search_area = data.bounding_boxes[0]
        #We use the 'probability' field to pass the covariance.
        self.target_covariance=data.bounding_boxes[0].probability
        # self.operation_mode=self.SEARCH_MODE
        # self.search_start=rospy.get_time()
        

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
                if segment_frame[i,j,3]==median: 
                    mask[i,j]=255
                    # self.im_output[bbox.ymin+i,bbox.xmin+j]=(255,255,255,255)

        return segment_frame, mask
    
    def histogramPartsBody(self, segment_frame, mask):
        #Divide the person in 5 equal parts
        shape = np.shape(segment_frame)
        descriptor=[]
        part_mask=None
        for i in range(1,5):
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
        The second bbox can be outside the limits, considering the 360 degrees.
        """
        outside_limits=False
        if(bb2.xmax>self.img_size[0] and bb2.xmin<0):
            return (bb1.xmax - bb1.xmin) * (bb1.ymax - bb1.ymin) / (self.img_size[0]*self.img_size[1])
        elif(bb2.xmax>self.img_size[0]):
            outside_limits=True
            bb3=copy(bb2); bb3.xmin=0; bb3.xmax=bb2.xmax-self.img_size[0]
            bb2.xmax=self.img_size[0]
        elif(bb2.xmin<0):
            outside_limits=True
            bb3=copy(bb2); bb3.xmin=self.img_size[0]+bb2.xmin; bb3.xmax=self.img_size[0]
            bb2.xmin=0

        if outside_limits:
            x_left = max(bb1.xmin, bb2.xmin)
            y_top = max(bb1.ymin, bb2.ymin)
            x_right = min(bb1.xmax, bb2.xmax)
            y_bottom = min(bb1.ymax, bb2.ymax)
            intersection_area=0
            if x_right > x_left and y_bottom > y_top:
                intersection_area+=(x_right - x_left) * (y_bottom - y_top)

            x_left = max(bb1.xmin, bb3.xmin)
            y_top = max(bb1.ymin, bb3.ymin)
            x_right = min(bb1.xmax, bb3.xmax)
            y_bottom = min(bb1.ymax, bb3.ymax)
            if x_right > x_left and y_bottom > y_top:
                intersection_area+=(x_right - x_left) * (y_bottom - y_top)

            # compute the area of both AABBs
            bb1_area = (bb1.xmax - bb1.xmin) * (bb1.ymax - bb1.ymin)
            bb2_area = (bb2.xmax - bb2.xmin) * (bb2.ymax - bb2.ymin)
            bb3_area = (bb3.xmax - bb3.xmin) * (bb3.ymax - bb3.ymin)
            # compute the intersection over union by taking the intersection
            # area and dividing it by the areas - the interesection area
            iou = intersection_area / float(bb1_area + bb2_area + bb3_area - intersection_area)
            assert iou >= 0.0
            assert iou <= 1.0
            return iou
        
        else:
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
            print("DaSiamRPN mean time: ", self.track_t*(10**-9))
            print("YOLO mean time: ", self.detect_t*(10**-9))
            # print("Search mean time: ", self.search_t)

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

def main_thread(node: FusionTracker):
  while not rospy.is_shutdown():
    if node.flag_select: node.operation_mode=node.SELECT_TARGET_MODE; node.flag_select=False
    if node.flag_image:
        node.yolo_tracker.bb_msg=[]; node.dasiam_tracker.bb_msg=[]
        node.flag_image=False
        node.im_output=node.im.copy()
        if node.operation_mode==node.SELECT_TARGET_MODE: node.select_mode(node.im.copy())
        # elif node.operation_mode==node.SEARCH_MODE: node.search_mode(node.im.copy())
        elif node.operation_mode==node.TRACK_MODE: 
            node.track_dasiam(node.im.copy())
            node.track_yolo(node.im.copy())

        #Publish messages
        bounding_boxes_dasiam = BoundingBoxes()
        bounding_boxes_dasiam.header = node.data.header
        bounding_boxes_dasiam.image_header = node.data.header
        for bbox in node.dasiam_tracker.bb_msg:
            #We use the 'probability' field to pass the covariance.
            bbox.probability=node.dasiam_tracker.covariance
            bounding_boxes_dasiam.bounding_boxes.append(bbox) #Append the bounding boxes which are between the limits of the image.
        node.dasiam_pub.publish(bounding_boxes_dasiam)

        bounding_boxes_yolo = BoundingBoxes()
        bounding_boxes_yolo.header = node.data.header
        bounding_boxes_yolo.image_header = node.data.header
        for bbox in node.yolo_tracker.bb_msg:
            #We use the 'probability' field to pass the covariance.
            bbox.probability=node.yolo_tracker.covariance
            bounding_boxes_yolo.bounding_boxes.append(bbox) #Append the bounding boxes which are between the limits of the image.
        node.yolo_pub.publish(bounding_boxes_yolo)
        # cv2.rectangle(node.im_output,(node.search_area.xmin,node.search_area.ymin),(node.search_area.xmax,node.search_area.ymax), (255,255,0))
        node.image_pub.publish(node.bridge.cv2_to_imgmsg(node.im_output, "bgra8")) 
      
    node.rate.sleep()


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop", "opencv-python"))
    

    rospy.init_node("tracker", anonymous=True)
    tracker_node = FusionTracker()
    kthread = KeyboardThread(tracker_node.keyboard_callback)    
   
    rospy.spin()