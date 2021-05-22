import numpy as np
from skimage.filters import frangi
from skimage.color import rgb2gray
from torchvision.transforms import transforms
import matplotlib.pyplot as plt
import torch

np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]

class ContrastiveLearningFrangiGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):

        image = self.base_transform(x)
        img = (np.transpose(image.cpu().detach().numpy(), (1, 2, 0))*255)
        gray_img = rgb2gray(img)
        
        plt.imsave('frame.jpg', gray_img, cmap='gray' )

        filtered_1 = frangi(gray_img,sigmas=range(5, 10, 5)) # gray_img is uint8 while filtered img is between 0 and 1

        filtered_2 = frangi(gray_img,sigmas=range(5, 10, 5)) # gray_img is uint8 while filtered img is between 0 and 1

        filtered_3 = frangi(gray_img,sigmas=range(5, 10, 5)) # gray_img is uint8 while filtered img is between 0 and 1
        plt.imsave('filtered_1.jpg', filtered_1, cmap='gray')
        plt.imsave('filtered_2.jpg', filtered_2, cmap='gray')
        plt.imsave('filtered_3.jpg', filtered_3, cmap='gray')
        filtered_1 = transforms.ToTensor()(filtered_1).float() # gray_img is uint8 while filtered img is between 0 and 1
        filtered_2 = transforms.ToTensor()(filtered_2).float() # gray_img is uint8 while filtered img is between 0 and 1
        filtered_3 = transforms.ToTensor()(filtered_3).float() # gray_img is uint8 while filtered img is between 0 and 1

        filtered = torch.cat([filtered_1, filtered_2, filtered_3], dim=0)
        #print(self.base_transform(x).size())
        return [self.base_transform(x), filtered]
