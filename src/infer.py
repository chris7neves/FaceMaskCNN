import os
from skimage.io import imread
import torch

def prep_image(im_path, transforms):

    image = imread(im_path)
    image = transforms(image)
    image = torch.unsqueeze(image, 0)

    return image

def infer(im_path, model, transforms, as_label=False, label_dict=None):
    
    image = prep_image(im_path, transforms)
    model.eval()
    probabilities = model(image)
    prediction = torch.argmax(probabilities, 1)

    if as_label and label_dict is not None:
        return label_dict[prediction.item()]

    return probabilities, prediction
