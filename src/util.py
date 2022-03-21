import os
import numpy as np

def rename_images(im_dir, prefix="", suffix=""):
    """
    Utility function that renames all the images in a folder to have increasing integer numbers.
    """
    
    if not os.path.isdir(im_dir):
        print("The specified directory '{}' does not exist.")
        return False
    
    for i, fname in enumerate(os.listdir(im_dir)):
        _, ext = os.path.splitext(fname)
        new_name = prefix + str(i) + suffix + ext
        
        try:
            os.rename(os.path.join(im_dir,fname), os.path.join(im_dir,new_name))
        except FileExistsError:
            continue
            
    return True


def get_class(classes, label):
    return classes[np.where(label == 1)[0].item()]