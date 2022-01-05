import torch
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

# Main class for the dataloader, overrides Dataset class
class CassavaDataset(Dataset):
    def __init__(self, data, labels, do_transform):
        self.files = data
        self.targets = labels
        self.classes = list(set(labels))
        self.do_transform = do_transform

    def __len__(self):
        return len(self.files)

    # Return the transformed image and its corresponding label
    def __getitem__(self, i):
        if torch.is_tensor(i):
            i = i.tolist()

        image = Image.open(os.path.join(self.files[i]))
        SIZE = 364
        pop_mean = [0.4308398 , 0.49935585, 0.31198692]
        pop_std = [0.22837807, 0.2308237 , 0.21775971]

        if self.do_transform:
        
            transform = transforms.Compose([
                transforms.RandomResizedCrop((SIZE, SIZE)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=pop_mean, std=pop_std)
            ])

            image = transform(image)

        else:

            transform = transforms.Compose([
                transforms.Resize((SIZE, SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(mean=pop_mean, std=pop_std)
            ])

            image = transform(image)
            
        label = self.targets[i]

        return image, label
