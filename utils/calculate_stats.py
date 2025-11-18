import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys

project_root = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(project_root)

from dataset import HandDigitDataset


def get_dataset_stats(root_dir):
    """
    Calculate and return the mean and std of all pixels
    within dataset.

    Args:
        root_dir (str):  dataset root path.

    Returns:
        tuple: (mean, std) tuple contains mean and std.
    """
    print(f"start calculate the statistic data of '{root_dir}'")

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ])

    try:
        dataset = HandDigitDataset(root_dir=root_dir, transform=transform)
        if len(dataset) == 0:
            print("Error: dataset is empty")
            return None, None
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return None, None
        
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

    channel_sum = 0.
    channel_sum_sq = 0.
    num_pixels = 0
    
    for images, _ in loader:
        channel_sum += torch.sum(images)
        channel_sum_sq += torch.sum(images**2)
        num_pixels += images.nelement()

    mean = channel_sum / num_pixels
    std = torch.sqrt((channel_sum_sq / num_pixels) - (mean ** 2))

    mean_val = mean.item()
    std_val = std.item()

    print("\n--- calculation complete ---")
    print(f"total image number: {len(dataset)}")
    print(f"total pixels: {num_pixels}")
    print(f"Mean: {mean_val:.4f}")
    print(f"Std: {std_val:.4f}")
    
    return mean_val, std_val

if __name__ == '__main__':
    DATA_DIRECTORY = './data/train/' 
    
    mean, std = get_dataset_stats(root_dir=DATA_DIRECTORY)

    if mean is not None and std is not None:
        print("-" * 50)
        print(f"transform = transforms.Compose([\n"
              f"    transforms.Resize((32, 32)),\n"
              f"    transforms.ToTensor(),\n"
              f"    transforms.Normalize(({mean:.4f},), ({std:.4f},))\n"
              f"])")
        print("-" * 50)