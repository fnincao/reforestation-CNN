import torch
import torchvision
from dataset import PlanetDataset
from torch.utils.data import DataLoader


def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
    print('=> Saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print('=> Loading Checkpoint')
    model.load_state_dict(checkpoint['state_dict'])


def get_loaders(
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        batch_size,
        train_transform,
        num_workers=4,
        pin_memory=True,):

    train_ds = PlanetDataset(
        image_dir=train_img_dir,
        mask_dir=train_mask_dir,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True
    )

    val_ds = PlanetDataset(
        image_dir=val_img_dir,
        mask_dir=val_mask_dir,
        transform=None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, val_loader


def check_accuracy(loader, model, device='cuda'):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    true_positives = 0
    total_positives = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            true_positives += ((preds == 1) & (y == 1)).sum()
            total_positives += (y == 1).sum()

            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

        print(
            f'Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100}' # noqa
        )

        print(f'Dice Score: {dice_score/len(loader)}')
        print(f'Producers accuracy: {true_positives/total_positives}')
        model.train()
    
    return (num_correct/num_pixels*100), (dice_score/len(loader)), (true_positives/total_positives), 


def save_predictions_as_imgs(
        loader,
        model,
        folder='../../data/ai_data/saved_images',
        device='cuda',):

    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(
            preds, f'{folder}/pred_{idx}.png'
        )
        torchvision.utils.save_image(y.unsqueeze(1),
                                     f'{folder}/gt_{idx}.png')

        model.train()
