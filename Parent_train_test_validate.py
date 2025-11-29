

import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)

from torchvision.datasets import OxfordIIITPet

# Download images and masks
train_set_raw = OxfordIIITPet(root='data', split='trainval', target_types='segmentation', download=True)
test_set_raw= OxfordIIITPet(root='data', split='test', target_types='segmentation', download=True)
# Check an example
img, mask = train_set_raw[0]   # use train_set_raw, not dataset
print("Image size:", img.size, "Mask size:", mask.size)

class PetDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.mask_transform = T.Compose([
            T.Resize((160,160), interpolation=T.InterpolationMode.NEAREST)
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, mask = self.dataset[idx]

        if self.transform:
            img = self.transform(img)

        # Proper mask handling
        mask = self.mask_transform(mask)
        mask = np.array(mask)

        # Binary pet vs background
        mask = (mask == 1).astype(np.float32)

        mask = torch.tensor(mask).unsqueeze(0)
        return img, mask

transform = T.Compose([
        T.Resize((160,160)),
        T.ToTensor(),  T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

train_dataset = PetDataset(train_set_raw, transform=transform)
test_dataset = PetDataset(test_set_raw,transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader=DataLoader(test_dataset, batch_size=8, shuffle=False)

#****visualisation for sanity checking*****#
import matplotlib.pyplot as plt
import numpy as np

# Get the first item from the processed dataset
img, mask = train_dataset[0]

# Convert image tensor to numpy for plotting
img_np = np.transpose(img.numpy(), (1, 2, 0))  # (C,H,W) -> (H,W,C)
mask_np = mask[0].numpy()                       # (1,H,W) -> (H,W)

plt.figure(figsize=(8,4))

# Plot the image
plt.subplot(1,2,1)
plt.imshow(img_np)
plt.title("Image")
plt.axis('off')

# Plot the mask
plt.subplot(1,2,2)
plt.imshow(mask_np, cmap='gray')
plt.title("Mask (pet=1, background=0)")
plt.axis('off')

plt.show()


class SimpleSegNet(nn.Module):
    def __init__(self):
        super(SimpleSegNet, self).__init__()

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)

        # Decoder
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 1)

        # Activation
        self.act = nn.LeakyReLU(0.1)

        # Downsampling
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.pool(x)

        x = self.act(self.conv2(x))
        x = self.pool(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.act(self.conv3(x))

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = torch.sigmoid(self.conv4(x))

        return x
model = MicroUNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def dice_loss(pred, target, eps=1e-6):
    pred = pred.contiguous().view(pred.shape[0], -1)
    target = target.contiguous().view(target.shape[0], -1)

    intersection = (pred * target).sum(dim=1)
    union = pred.sum(dim=1) + target.sum(dim=1)

    dice = (2. * intersection + eps) / (union + eps)
    return 1 - dice.mean()

criterion = lambda p, t: 0.5 * dice_loss(p,t) + 0.5 * nn.BCELoss()(p,t)


for epoch in range(50):
    epoch_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {epoch_loss / len(train_loader):.4f}")
