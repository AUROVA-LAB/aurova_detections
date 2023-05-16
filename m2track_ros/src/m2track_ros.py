#!/usr/bin/env python
# -*- coding: utf-8 -*-


#########################################
# Add any ROS dependency 
#########################################
import rospy
from dynamic_reconfigure.server import Server as DynamicReconfigureServer
from sensor_msgs.msg import Image, PointCloud2, PointField
from geometry_msgs.msg import PoseWithCovarianceStamped
from vision_msgs.msg import BoundingBox3D
# from ros_numpy import numpify
import message_filters
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pcl2
from std_msgs.msg import Header
from detection_msgs.msg import BoundingBox, BoundingBoxes

#########################################
# Add any non ROS dependency 
#########################################
import sys
from threading import Thread
import importlib
import numpy as np
import termios, tty
import time, copy

class pc_tracker_Node:

  def __init__(self):
    print("Starting initialization.")
    # get the main thread desired rate of the node
    self.rate_value = rospy.get_param('~rate', 10)
    self.rate=rospy.Rate(self.rate_value)
    #Get parameters
    self.inference_path = rospy.get_param('~inference_path')
    #Init variables
    self.flag_pc=False; self.flag_target=False; self.this_frame=None; self.mutex=False
    self.predict_time, self.reconstruction_time= 0, 0
    self.bridge = CvBridge()
    self.search_area=None

    #Load net
    sys.path.append(self.inference_path)
    self.inference=importlib.import_module("inference")
    self.net=self.inference.get_net(self.inference_path)

    #Size of the 3D bounding box.
    self.target=BoundingBox3D()
    self.target.size.z=2.0; self.target.size.x=0.75; self.target.size.y=0.75
    self.target.center.position.x=0.0; self.target.center.position.y=0.0; self.target.center.position.z=0.0

    # Create topic publishers
    self.publisher_tracked = rospy.Publisher("~tracked_object", PoseWithCovarianceStamped, queue_size=1)
    # Create topic subscribers
    self.subscriber_depth_image = message_filters.Subscriber("~depth_topic", Image)
    self.subscriber_intensity_image = message_filters.Subscriber("~intensity_topic", Image)
    self.subscriber_search_area=rospy.Subscriber("~search_area_topic", BoundingBoxes, self.search_area_callback, queue_size=1)
    self.subscriber_image = message_filters.TimeSynchronizer([self.subscriber_depth_image, self.subscriber_intensity_image], 1)
    self.subscriber_image.registerCallback(self.image_callback)
    self.subscriber_target = rospy.Subscriber("~target", PoseWithCovarianceStamped, self.target_callback, queue_size=1)

    # start main thread
    self.thread = Thread(target = main_thread, args = (self, ))
    self.thread.start()
    print("Initilization finished.")


  #########################################
  # subscriber callbacks
  #########################################
  def image_callback(self,depth_image, intensity_image):
    if not self.mutex:
      start=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
      self.flag_pc=True
      depth = self.bridge.imgmsg_to_cv2(depth_image)
      #Note: 65536=2**16, reducing calculations
      depth = (depth*(261/65536)).transpose()
      intensity = self.bridge.imgmsg_to_cv2(intensity_image).transpose()
      iy=np.arange(depth.shape[1]); ix=np.arange(depth.shape[0])
      if self.search_area is None:
        self.search_area=BoundingBox()
        self.search_area.xmin, self.search_area.xmax = 0, depth.shape[0]
        self.search_area.ymin, self.search_area.ymax = 0, depth.shape[1]

      search_area=copy.deepcopy(self.search_area)
      if search_area.xmin>=0:
        depth=depth[search_area.xmin:search_area.xmax, search_area.ymin:search_area.ymax]
        intensity=intensity[search_area.xmin:search_area.xmax, search_area.ymin:search_area.ymax]
        ix=ix[search_area.xmin:search_area.xmax]; iy=iy[search_area.ymin:search_area.ymax]
      else:
        n_rows=depth.shape[0]; search_area.xmin+=n_rows
        depth=np.append(depth[0:search_area.xmax, search_area.ymin:search_area.ymax],
                        depth[search_area.xmin:n_rows, search_area.ymin:search_area.ymax], axis=0)
        intensity=np.append(intensity[0:search_area.xmax, search_area.ymin:search_area.ymax],
                        intensity[search_area.xmin:n_rows, search_area.ymin:search_area.ymax], axis=0)
        ix=np.append(ix[0:search_area.xmax],ix[search_area.xmin:n_rows])
        iy=iy[search_area.ymin:search_area.ymax]


      sang_h=np.diagflat(np.sin((22.5-iy*(45.0/128.0))*np.pi/180.0))
      ang_w=(184.0 - (360.0/2048.0)*ix)*np.pi/180.0
      sang_w=np.diagflat(np.sin(ang_w)); cang_w=np.diagflat(np.cos(ang_w))
      Z=np.matmul(depth,sang_h)
      aux=(depth**2-Z**2)**0.5
      self.this_frame=np.array([np.matmul(cang_w,aux).flatten(),np.matmul(sang_w,aux).flatten(),Z.flatten(),intensity.flatten()])
      diff=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)-start
      self.reconstruction_time=diff if self.reconstruction_time==0 else (self.reconstruction_time+diff)/2


  def search_area_callback(self,data):
    if not self.mutex:
      self.search_area = data.bounding_boxes[0]

  def target_callback(self,data):
    dist=np.sqrt((self.target.center.position.x-data.pose.pose.position.x)**2+(self.target.center.position.y-data.pose.pose.position.y)**2 
                   + (self.target.center.position.z-data.pose.pose.position.z)**2)
    # Do not update the target if it is close to the actual target in order to obtain a better net return.
    if (not self.mutex and self.this_frame is not None and dist > 0.25):
      self.flag_pc=False; self.flag_target=True
      # Transform to  a 3D box
      self.target.center=data.pose.pose
      self.prev_frame=self.this_frame
      self.frame_id = data.header.frame_id
     
     

  def keyboard_callback(self, input):
        # print("Input: ", input)
        if input=='s':
            self.flag_target=False
        elif input=='t':
          print("Predict time ",self.predict_time*10**-9)
          print("Reconstruction time ",self.reconstruction_time*10**-9)

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

def main(args):

  rospy.init_node('pc_tracker', anonymous=False)
  tn = pc_tracker_Node()
  kthread = KeyboardThread(tn.keyboard_callback)
  rospy.spin()

def main_thread(node: pc_tracker_Node):
  while not rospy.is_shutdown():
    if node.flag_target and node.flag_pc:
      node.mutex=True
      node.flag_pc=False
      start=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)
      predicted_box = node.inference.predict(node.net, node.prev_frame, node.this_frame, node.target)
      diff=time.clock_gettime_ns(time.CLOCK_THREAD_CPUTIME_ID)-start
      node.predict_time=diff if node.predict_time==0 else (node.predict_time+diff)/2
      #Calculate covariance
      dist=np.sqrt((node.target.center.position.x-predicted_box.center.position.x)**2+(node.target.center.position.y-predicted_box.center.position.y)**2 
                   + (node.target.center.position.z-predicted_box.center.position.z)**2)
      covariance=0.1+dist 
      #Update target
      node.target=predicted_box
      node.prev_frame=node.this_frame
      #Publish marker message
      target_msg = PoseWithCovarianceStamped()
      target_msg.header.frame_id = node.frame_id
      target_msg.header.stamp = rospy.get_rostime()
      target_msg.pose.pose = predicted_box.center
      target_msg.pose.covariance[0], target_msg.pose.covariance[7], target_msg.pose.covariance[14] = covariance, covariance,1
      node.publisher_tracked.publish(target_msg)
      node.mutex=False
      
    node.rate.sleep()

if __name__ == '__main__':
  main(sys.argv)

