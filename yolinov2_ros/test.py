import torch
import torch.nn as nn
from models import UnetPlusPlus, PSPNet, DeepLabV3Plus
from dataset import SegmentationDataset
from utils import *
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import time
import sys



PATH = "/docker_shared/yolinov2_shared/experiments/exp_2023-02-13/"
TEST_IMG_DIR = PATH
TEST_MASK_DIR = PATH
TEST_PREDS_DIR = PATH
IMAGE_HEIGHT = 2048
IMAGE_WIDTH = 128  # 240 for unet, 256 for unet smp with resnet18
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_WORKERS = os.cpu_count()
PIN_MEMORY = True


'''
def main():

        # UNET model
        #model = UNET(in_channels=3, out_channels=1).to(DEVICE)


    for nn in ["UnetPlusPlus", "PSPNet", "DeepLabV3Plus"]:

        for backbone in ["resnet18", "efficientnet-b3"]:

            for dataset in ["dataset1", "dataset2", "dataset3"]:

                if nn is "UnetPlusPlus":
                    # unet smp with resnet18 as backbone
                    model = UnetPlusPlus(backbone, "imagenet", in_channels=3, out_channels=1).to(DEVICE)
        
                if nn is "PSPNet":
                    model = PSPNet(backbone, "imagenet", in_channels=3, out_channels=1).to(DEVICE)

                if nn is "DeepLabV3Plus":
                    model = DeepLabV3Plus(backbone, "imagenet", in_channels=3, out_channels=1).to(DEVICE)

                load_checkpoint(torch.load(f"experimentation/full_exp_segmentation/{nn}/{dataset}/{backbone}/epoch_29.pth.tar"), model)

                test_transform = A.Compose(
                    [
                        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                        A.Normalize(
                            mean=[0.0, 0.0, 0.0],
                            std=[1.0, 1.0, 1.0],
                            max_pixel_value=255.0  # value you want to divide by the pixels
                        ),
                        ToTensorV2(),
                    ]
                )

                test_dataset = SegmentationDataset(os.path.join(TEST_IMG_DIR, dataset, "test"), os.path.join(TEST_MASK_DIR, dataset, "test_masks"), test_transform)

                test_dataloader = DataLoader(
                    test_dataset,
                    batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS,
                    pin_memory=PIN_MEMORY,
                    shuffle=False,
                )

                test_loss, test_dice, test_dice_std, test_iou, test_iou_std, avg_inf_time, std_inf_time = dice_iou_calculation(test_dataloader, model)
                print(f"Results for {nn} model with {backbone} backbone and dataset {dataset}")
                print(f"Test Dice score: {test_dice} +- {test_dice_std} \nTest IoU score: {test_iou} +- {test_iou_std} \nAvg inference time: {avg_inf_time} +- {std_inf_time} \n\n\n")

                #save_predictions_as_imgs_test(test_dataloader, model, TEST_PREDS_DIR, DEVICE)
'''
'''
def test_one_image():
    model = UNET_SMP("resnet18", "imagenet", in_channels=3, out_channels=1).to(DEVICE)

    load_checkpoint(torch.load("checkpoint_epoch_4.pth.tar"), model)

    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0  # value you want to divide by the pixels
            ),
            ToTensorV2(),
        ]
    )

    image = np.array(Image.open(PATH + "new/" + "1_color.png").convert("RGB"))

    with torch.inference_mode():

        image = test_transform(image=image)
        image = image["image"]

        image = image.to(DEVICE).unsqueeze(0)
        pred = torch.sigmoid(model(image))
        pred = (pred > 0.5).float()

        torchvision.utils.save_image(
          pred, f"new/pred.png"
      )
'''

def test_single_shot():

    n_epoch = sys.argv[1]

    model = UnetPlusPlus("resnet18", "imagenet", in_channels=3, out_channels=1).to(DEVICE)

    load_checkpoint(torch.load(PATH + "epochs/checkpoint_epoch_" + str(n_epoch) + ".pth.tar"), model)

    test_transform = A.Compose(
        [
            #A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0  # value you want to divide by the pixels
            ),
            ToTensorV2(),
        ]
    )

    id_val = sys.argv[2]

    image = np.array(Image.open(PATH + "train/merged_" + str(id_val) + ".png").convert("RGB"))

    with torch.inference_mode():

        image = test_transform(image=image)
        image = image["image"]

        image = image.to(DEVICE).unsqueeze(0)
        pred = torch.sigmoid(model(image))
        pred = (pred > 0.5).float()

        torchvision.utils.save_image(
          pred, PATH + "out/merged_" + str(id_val) + "_pred.png"
      )
        torchvision.utils.save_image(
          image, PATH + "out/merged_" + str(id_val) + "_imag.png"
      )

if __name__ == '__main__':
    test_single_shot()