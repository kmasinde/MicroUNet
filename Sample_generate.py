import matplotlib.pyplot as plt
import numpy as np

# Grab a batch from test_loader
imgs, masks = next(iter(test_loader))

# Move to device
imgs, masks = imgs.to(device), masks.to(device)

# Run the model (no gradient needed)
with torch.no_grad():
    preds = model(imgs)