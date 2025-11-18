#For the RBF parameters, use the data DIGIT

from PIL import Image
import torchvision.transforms as T
import torch

image_paths = [
    "img001-00001.png",  # Digit 0
    "img002-00001.png",  # Digit 1
    "img003-00001.png",  # Digit 2
    "img004-00001.png",  # Digit 3
    "img005-00001.png",  # Digit 4
    "img006-00001.png",  # Digit 5
    "img007-00001.png",  # Digit 6
    "img008-00001.png",  # Digit 7
    "img009-00001.png",  # Digit 8
    "img010-00001.png",  # Digit 9
]

transform = T.Compose([
    T.Grayscale(),
    T.Resize((7, 12)),
    T.ToTensor()
])

rbf_targets = torch.stack([
    transform(Image.open(path)).view(-1)
    for path in image_paths
])

# Final shape: [10, 84] to have RBF vector per digit class
torch.save(rbf_targets, "rbf_targets.pt")
