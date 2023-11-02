# training code
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from models import UnetPlusPlus, PSPNet, DeepLabV3Plus
#from unet import UNET, UNET_SMP
from utils import *
#import wandb
import os
import sys

from segmentation_models_pytorch.losses import DiceLoss, JaccardLoss, FocalLoss

# login wandb account
#os.system("wandb login")
#wandb.init(project="ground_lines", entity="aurova_lab")

# hyperparameters
MODEL_NAME = "UnetPlusPlus"
DATASET_NUMBER = 1
BACKBONE = "resnet18"
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2
NUM_EPOCHS = 150
NUM_WORKERS = os.cpu_count()
IMAGE_HEIGHT = sys.argv[5] # 288 # 2048
IMAGE_WIDTH = sys.argv[4] # 128 
PIN_MEMORY = True
LOAD_MODEL = False
PATH = sys.argv[1] #"/docker_shared/yolinov2_shared/experiments/exp_2023-10-17/"
TRAIN_IMG_DIR = PATH + "train/"
TRAIN_MASK_DIR = PATH + "train_masks/"
VAL_IMG_DIR = PATH + "val/"
VAL_MASK_DIR = PATH + "val_masks/"


# training function, this train only runs an epoch

def train_fn(loader, model, optimizer, loss_fn):

	# progress bar
	loop = tqdm(loader)

	running_loss = 0

	for batch_idx, (data, targets) in enumerate(loop):

		# data to device
		data, targets = data.to(DEVICE), targets.float().unsqueeze(1).to(DEVICE)
		
		# forward
		predictions = model(data)
		
		# loss
		loss = loss_fn(predictions, targets)
		running_loss += loss.item()

		# zero grad
		optimizer.zero_grad()
		#backward
		loss.backward()
		# update weights
		optimizer.step()

		# update tqdm loop
		loop.set_postfix(loss=loss.item())

	return running_loss/len(loader)



def main():

	# save hyper-parameters in wandb
	'''
	wandb.config = {
		"model_name": MODEL_NAME,
		"learning_rate": LEARNING_RATE,
		"batch_size": BATCH_SIZE,
		"num_epochs": NUM_EPOCHS,
		"im_height": IMAGE_HEIGHT,
		"im_width": IMAGE_WIDTH,
		"backbone": BACKBONE,
		"optimizer": "adam",
		"loss_name": "BCEWithLogits",
		"dataset_number": DATASET_NUMBER
	}
	'''

	# with A.Normalize we just divide the pixels by 255 to have them
	# in 0-1 range
	
	train_transform = A.Compose(
		[
		#A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
		#A.Rotate(limit=35, p=0.5),
		A.HorizontalFlip(p=0.5),
		#A.VerticalFlip(p=0.5),
		A.Normalize(
			mean=[0.0, 0.0, 0.0],
			std=[1.0, 1.0, 1.0],
			max_pixel_value=255.0
		),
		ToTensorV2(),
		],
	)

	val_transform = A.Compose(
		[
		#A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
		A.Normalize(
			mean=[0.0, 0.0, 0.0],
			std=[1.0, 1.0, 1.0],
			max_pixel_value=255.0
		),
		ToTensorV2(),
		],
	)

	# model
	model = UnetPlusPlus(BACKBONE, None, in_channels=3, out_channels=1).to(DEVICE)
	#model = UnetPlusPlus(BACKBONE, "imagenet", in_channels=3, out_channels=1).to(DEVICE)
	#model = UNET_SMP(BACKBONE, "imagenet", in_channels=3, out_channels=1).to(DEVICE)
	n_epoch = sys.argv[3]
	if sys.argv[2]:
		load_checkpoint(torch.load(PATH + "epochs/checkpoint_epoch_" + str(n_epoch) + ".pth.tar"), model)
	#loss fn
	loss_fn = nn.BCEWithLogitsLoss()
	#loss_fn = JaccardLoss('binary')
	#loss_fn = DiceLoss('binary')
	#loss_fn = FocalLoss('binary')

	# optimizer
	optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

	# data loaders
	train_loader, val_loader = get_loaders(
		TRAIN_IMG_DIR,
		TRAIN_MASK_DIR,
		VAL_IMG_DIR,
		VAL_MASK_DIR,
		BATCH_SIZE,
		train_transform,
		val_transform,
		NUM_WORKERS,
		PIN_MEMORY
	)

	for epoch in range(NUM_EPOCHS):

		print(f"------------- Epoch {epoch} -------------")
		print(f"------------- {DEVICE} -------------")

		train_loss = train_fn(train_loader, model, optimizer, loss_fn)
		#wandb.log({'epoch': epoch + 1, 'train_loss': train_loss})
		print(f"Training loss: {train_loss} \n")

		checkpoint = {
			"state_dict": model.state_dict(),
			"optimizer": optimizer.state_dict(),
		}

		save_checkpoint(checkpoint, epoch, PATH + "/epochs")

		#val_loss, val_dice, val_iou = dice_iou_calculation(val_loader, model, loss_fn)
		#wandb.log({'epoch': epoch + 1, 'val_loss': val_loss})
		#wandb.log({'epoch': epoch + 1, 'val_dice': val_dice})
		#wandb.log({'epoch': epoch + 1, 'val_iou': val_iou})
		#print(f"Val loss: {val_loss} \nVal Dice score: {val_dice} \nVal IoU score: {val_iou} \n")

		#save_predictions_as_imgs(
		#	train_loader, model, folder= PATH + "/out/", device=DEVICE, epoch=epoch
		#)

if __name__ == '__main__':
	main()
