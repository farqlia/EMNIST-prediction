from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective, RandomRotation, RandomResizedCrop
import torch 

from emnist_prediction.constants import IMG_SIZE

image_transform = transforms.Compose(
    [RandomHorizontalFlip(p=0.25),
    RandomVerticalFlip(p=0.25),
    RandomResizedCrop((28, 28), scale=(0.8,1.0), ratio=(0.9,1.1), antialias=True),
    RandomPerspective(distortion_scale=0.5, p=0.25, fill=0),
    RandomRotation(degrees=45)]
)

class ImgTransform:

    def __init__(self, img_transform, color_channel=False):
        self.img_transform = img_transform
        self.color_channel = color_channel

    def __call__(self, sample):
        x, y = sample
        x = x.reshape(1, *IMG_SIZE) # add color channel
        if not self.color_channel:
            x = self.img_transform(x).reshape(*IMG_SIZE) # remove color channel
        return x, y
    

def visualize_img_transform(sample_img, label, transform):
    img_tensor = torch.from_numpy(sample_img.reshape(1, *IMG_SIZE))
    plt.title(label)
    plt.imshow(transform(img_tensor).reshape(*IMG_SIZE), cmap='Greys')

class Reshape(object):
    
    def __init__(self, shape=(-1, )):
        self.shape = shape
    
    def __call__(self, sample):
        x, y = sample 
        return x.reshape(self.shape), y