import cv2
import sys
import os
import numpy as np


if len(sys.argv)<3:
    print("USAGE: python merge_channels.py /path/to/folder maxImageID.")
    print("In the folder, it must be nearir, reflec and signal folders.")
    exit(0)

path=sys.argv[1]
id=sys.argv[2]
if not os.path.isfile(path+"nearir/nearir_"+id+".png"):
    print("The path or the ID is incorrect.")
    exit(0)
if not os.path.exists(path+"train"):
    os.mkdir(path+"train")

for i in range(int(id)+1):
    depht = cv2.imread(path+"range/range"+str(i)+".png",cv2.IMREAD_GRAYSCALE)
    nearir = cv2.imread(path+"nearir/nearir_"+str(i)+".png",cv2.IMREAD_GRAYSCALE)
    reflec = cv2.imread(path+"reflec/reflec"+str(i)+".png",cv2.IMREAD_GRAYSCALE)
    signal = cv2.imread(path+"signal/signal"+str(i)+".png",cv2.IMREAD_GRAYSCALE)

    merged=cv2.merge([nearir,reflec,depht])

    if cv2.imwrite(path+"train/merged_"+str(i)+".png", merged):
        print(i)

# cv2.namedWindow("Merged", cv2.WINDOW_NORMAL)
# cv2.resizeWindow("Merged", 2160, 200)
# cv2.imshow("Merged", merged)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

