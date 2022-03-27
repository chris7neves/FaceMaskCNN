from skimage.io import imread


def prep_image(im_path, transforms):
    """
    Prepare an image to be fed to the model.
    :im_path: path to the image
    :transforms: transforms to apply to the image
    """

    image = imread(im_path)

    if (image.shape[2] == 4):
        image = image[:, :, :3]
    image = transforms(image)
    image = torch.unsqueeze(image, 0)

    return image

def infer(im_path, model, transforms, as_label=False, label_dict=None):
    """
    Function that allows inference using a model
    :model: model to use for inference. It should have its parameters already loaded.
    """

    image = prep_image(im_path, transforms)
    model.eval()
    probabilities = model(image)
    prediction = torch.argmax(probabilities, 1)

    if as_label and label_dict is not None:
        return probabilities, label_dict[prediction.item()]

    return probabilities, prediction.item()
