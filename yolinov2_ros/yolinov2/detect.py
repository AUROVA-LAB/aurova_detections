#!/usr/bin/env python3

import message_filters
import rclpy
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
from yolinov2.utils import *
from yolinov2.models import UnetPlusPlus
import numpy as np
import sys

def callback(merged_in):
    global bridge, merge_img_pub, model, test_transform, DEVICE, detection_img_pub, node

    ini = node.get_clock().now().nanoseconds / 1e9

    image = bridge.imgmsg_to_cv2(merged_in, "bgr8")

    image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    image = image.unsqueeze(0).to(DEVICE)  # BCHW

    pred = torch.sigmoid(model(image))
    pred = (pred > 0.5).float()

    detection = pred.cpu().detach().numpy()
    final_mask = detection[0][0].astype(np.uint8)*255
    image_message = bridge.cv2_to_imgmsg(final_mask, "mono8")

    image_message.header.stamp = merged_in.header.stamp
    image_message.header.frame_id = "os_sensor"
    detection_img_pub.publish(image_message)

    end = node.get_clock().now().nanoseconds / 1e9
    print(f"Time: {end - ini}")




def main(args=None):

    global bridge, merge_img_pub, model, test_transform, DEVICE, detection_img_pub, node

    # ROS node initialization
    rclpy.init(args=args)
    node = rclpy.create_node("ground_lines_detection")
    bridge = CvBridge()
    node.create_subscription(Image, '/ouster_merged', callback, 10)
    detection_img_pub = node.create_publisher(Image, '/ground_lines_img', 10)

    # Load segmentation model (yolinov2)
    PATH = sys.argv[1] #"/docker_shared/yolinov2_shared/experiments/exp_2023-10-17/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    n_epoch = int(sys.argv[2]) #149
    model = UnetPlusPlus("resnet18", None, in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(PATH + "epochs/checkpoint_epoch_" + str(n_epoch) + ".pth.tar"), model)

    print(f"Network model loaded with {DEVICE}")
    print("seq + time")
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

