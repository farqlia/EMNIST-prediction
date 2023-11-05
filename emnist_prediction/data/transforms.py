import numpy as np
import torch
from torchvision import transforms
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomPerspective, RandomRotation, \
    RandomResizedCrop

from emnist_prediction.utils.constants import IMG_SIZE, INPUT_DATA_DIR
from emnist_prediction.utils.utils import get_classes_count

image_transform = transforms.Compose(
    [RandomHorizontalFlip(p=0.25),
     RandomVerticalFlip(p=0.25),
     RandomResizedCrop((28, 28), scale=(0.8, 1.0), ratio=(0.9, 1.1), antialias=True),
     RandomPerspective(distortion_scale=0.5, p=0.25, fill=0),
     RandomRotation(degrees=45)]
)


def get_classes_weights():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    y_train = np.load(INPUT_DATA_DIR / 'y_train.npy')
    classes_count = get_classes_count(y_train)
    class_share = classes_count / sum(classes_count)
    class_weights = 1 / (class_share * 100)
    class_weights = class_weights.sort_index()
    class_weights = torch.tensor(class_weights).to(device).type(torch.float32)
    return class_weights


class ImgTransform:

    def __init__(self, img_transform, color_channel=False):
        self.img_transform = img_transform
        self.color_channel = color_channel

    def __call__(self, sample):
        x, y = sample
        x = x.reshape(1, *IMG_SIZE)  # add color channel
        if not self.color_channel:
            x = self.img_transform(x).reshape(*IMG_SIZE)  # remove color channel
        return x, y


class Reshape(object):

    def __init__(self, shape=(-1,)):
        self.shape = shape

    def __call__(self, sample):
        x, y = sample
        return x.reshape(self.shape), y
