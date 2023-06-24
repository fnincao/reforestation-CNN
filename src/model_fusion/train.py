import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import wandb
from tqdm import tqdm
from loss_fn import TverskyLoss, DiceLoss # noqa
import torch.optim as optim
from model import UNET
from utils import (load_checkpoint, # noqa
                   save_checkpoint,
                   get_loaders,
                   check_accuracy,
                   save_predictions_as_imgs)

import random
import numpy as np

random.seed(42)
np.random.seed(42)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 2
IMAGE_HEIGHT = 400
IMAGE_WIDTH = 400
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = '../../data/ai_data/train_images'
TRAIN_MASK_DIR = '../../data/ai_data/train_masks'
VAL_IMG_DIR = '../../data/ai_data/val_images'
VAL_MASK_DIR = '../../data/ai_data/val_masks'

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="Reforestation",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": LEARNING_RATE,
    "architecture": "UNET-NDVI",
    "epochs": NUM_EPOCHS,
    "dataset": 'Nordeste'}
)


# One epoch of training
def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    loss_list = []


    for batch_idx, (planet, s1, palsar, mask) in enumerate(loop):
        planet = planet.to(device=DEVICE)
        s1 = s1.to(device=DEVICE)
        palsar = palsar.to(device=DEVICE)
        mask = mask.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(planet, s1 , palsar)
            loss = loss_fn(predictions, mask)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        loss_list.append(loss.item())

    
    return sum(loss_list) / len(loss_list)


def main():
    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),

        }

        save_checkpoint(checkpoint)

        # check accuracy
        accuracy, precision, recall, dice_score = check_accuracy(val_loader, model, device=DEVICE)
        
        wandb.log({"accuracy": accuracy, "mean_loss": loss,
                   "Dice-score": dice_score, "precision":precision,
                  'recall':recall })

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, device=DEVICE
        )


if __name__ == '__main__':
    main()
    wandb.finish()
