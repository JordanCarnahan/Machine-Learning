# import required libraries
import torch
from PIL import Image
import torchvision.transforms as transforms

# Read input image from computer
img = Image.open('a.jpg')

# define an transform
transform = transforms.RandomAffine((50, 60))

# apply the above transform on image
img = transform(img)

# display image after apply transform
img.show()