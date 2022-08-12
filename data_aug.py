import torchvision.transforms.functional as TF
import random
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import torch
   
def segmentation_randrot90(image, segmentation):
    """Randomly rotates PIL image with segmentation mask by 90 degree multiple

    Args:
        image (PIL image): image
        segmentation (PIL image): segmentation mask

    Returns:
        iterable : Tensor of image and segmentation rotated
        
    """
    #q = random.randint(-30, 30)
    # print(angle)
    image = TF.pil_to_tensor(image)
    segmentation = TF.pil_to_tensor(segmentation)
    angle = 90*random.randint(1, 3)
    image = TF.rotate(image, angle)
    segmentation = TF.rotate(segmentation, angle)
    return image, segmentation


def segmentation_randflip(image, segmentation):
    """randomly flips PIL image with segmentation mask and returns 
    Tensor of output
    """
    image = TF.pil_to_tensor(image)
    segmentation = TF.pil_to_tensor(segmentation)
    #haha=random.random()
    if random.random()<0.5:
        #print("h")
        image=TF.hflip(image)
        segmentation=TF.hflip(segmentation)
    else:
        #print("v")
        image=TF.vflip(image)
        segmentation=TF.vflip(segmentation)
    return image, segmentation


def segmentation_randscale(image, segmentation):
    """randomly scales by scale (0.8, 1.2)"""
    S = random.uniform(0.8, 1.2)
    #print(S)
    image = TF.pil_to_tensor(image)
    segmentation = TF.pil_to_tensor(segmentation)
    #print(image.size())
    if S < 1:
        temp = 255*torch.ones(image.size(), dtype=torch.uint8)
        image = TF.resize(image, [int(temp.size()[1]*S), int(temp.size()[2]*S)])
        #print(image.size())
        temp[:, :image.size()[1], :image.size()[2]] = image
        image = temp

        temp = torch.zeros(image.size(), dtype=torch.uint8)
        segmentation = TF.resize(
            segmentation, [int(temp.size()[1]*S), int(temp.size()[2]*S)])
        temp[:, :segmentation.size()[1], :segmentation.size()[2]] = segmentation
        segmentation = temp
    elif S > 1:
        d, h, w = image.size()
        image = TF.resize(image, [int(h*S), int(w*S)])
        segmentation = TF.resize(segmentation, [int(h*S), int(w*S)])
        segmentation = TF.center_crop(segmentation, [h, w])
        image = TF.center_crop(image, [h, w])

    return image, segmentation
