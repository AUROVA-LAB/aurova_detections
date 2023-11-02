#!/usr/bin/env python3

import message_filters
import rospy
import os
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import torch
from utils import *
from models import UnetPlusPlus, PSPNet, DeepLabV3Plus
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import sys

def callback(merged_in):
    global flag, bridge, merge_img_pub, model, test_transform, DEVICE
    if flag:
        ini = rospy.Time.now().to_sec()

        image = bridge.imgmsg_to_cv2(merged_in, "bgr8")

        image = test_transform(image=image)
        image = image["image"]

        image = image.to(DEVICE).unsqueeze(0)
        pred = torch.sigmoid(model(image))
        pred = (pred > 0.5).float()

        flag=False
        detection = pred.cpu().detach().numpy()
        final_mask = detection[0][0].astype(np.uint8)*255
        image_message = bridge.cv2_to_imgmsg(final_mask, "mono8")
        image_message.header.seq = merged_in.header.seq
        image_message.header.stamp = merged_in.header.stamp
        image_message.header.frame_id = "os_sensor"
        detection_img_pub.publish(image_message)

        end = rospy.Time.now().to_sec()
        print(f"Time: {end - ini}")




if __name__ == '__main__':

    global flag, bridge, merge_img_pub, model, test_transform, DEVICE

    # ROS node initialization
    rospy.init_node("ground_lines_detection")
    flag=False
    bridge = CvBridge()
    rospy.Subscriber('/ouster_merged', Image, callback)
    detection_img_pub = rospy.Publisher('/ground_lines_img', Image, queue_size=10)

    # Load segmentation model (yolinov2)
    PATH = sys.argv[1] #"/docker_shared/yolinov2_shared/experiments/exp_2023-10-17/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    n_epoch = sys.argv[2] #149
    model = UnetPlusPlus("resnet18", None, in_channels=3, out_channels=1).to(DEVICE)
    load_checkpoint(torch.load(PATH + "epochs/checkpoint_epoch_" + str(n_epoch) + ".pth.tar"), model)
    test_transform = A.Compose(
        [
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0  # value you want to divide by the pixels
            ),
            ToTensorV2(),
        ]
    )
    print(f"Network model loaded with {DEVICE}")
    print("seq + time")

    r = rospy.Rate(100)
    while not rospy.is_shutdown():
        flag=True
        r.sleep()
