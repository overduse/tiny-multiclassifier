import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class HandDigitDataset(Dataset):
    """Custom Hand Digit Dataset"""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): root path of the dataset.
            transform (callable, optional): used for transformation of examples.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # check the exsistence of the root path.
        if not os.path.isdir(self.root_dir):
            raise FileNotFoundError(f"The dataset path do not exsit: {self.root_dir}")

        # label 0-9 -> the 10 Traditional Chinese
        for label_dir in sorted(os.listdir(self.root_dir)):
            dir_path = os.path.join(self.root_dir, label_dir)
            if os.path.isdir(dir_path):
                # transfor the 1-10 among label into 0-9
                label = int(label_dir) - 1
                for img_name in os.listdir(dir_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(dir_path, img_name)
                        self.samples.append((img_path, label))

    def __len__(self):
        """return the number of samples"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        get an example.
        Args:
            idx (int): index of sample.
        Returns:
            tuple: (image, label)
        """
        img_path, label = self.samples[idx]
        
        image = Image.open(img_path).convert('L')

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':
    # define data transformation
    # 1. transform the image into Torch.Tensor
    # 2. normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset_path = './data/train'
    my_dataset = HandDigitDataset(root_dir=train_dataset_path, transform=transform)

    print(f"the dataset have  {len(my_dataset)} images.")

    if len(my_dataset) > 0:
        first_image, first_label = my_dataset[0]
        print(f"the first example:")
        print(f"  - label: {first_label}")
        print(f"  - the shape of image tensor: {first_image.shape}")
        print(f"  - the type of image tensor: {first_image.dtype}")

        from torch.utils.data import DataLoader
        train_loader = DataLoader(dataset=my_dataset, batch_size=64, shuffle=True)
        
        images, labels = next(iter(train_loader))

        print("\nuse DataLoader to get the 1st batch of dataset:")
        print(f"  - batch image shape: {images.shape}") # [batch_size, channels, height, width]
        print(f"  - batch label shape: {labels.shape}")
