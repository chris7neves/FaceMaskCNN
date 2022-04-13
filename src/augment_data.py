from datetime import datetime
import os
from shutil import rmtree

import numpy as np
import skimage.transform as t
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.color import rgba2rgb

from util import rename_images
from configs.paths import paths_aug, paths_cropped, get_paths

#TODO: Link the augmenting to a cmd line entrypoint so module does not have to be manually run

def make_aug_dir(dir_path, overwrite=False):
    """
    Simply generates the directory to contain augmented images.
    If overwrite is specified, overwrites already existing dir.
    """
    new_dir = dir_path + "_aug"
    if (os.path.exists(new_dir)):  
        if overwrite:
            rmtree(new_dir)
        else:
            datetimeobj = datetime.now()
            tmo = datetimeobj.time()
            new_dir = new_dir + "{}_{}_{}".format(tmo.hour, tmo.minute, tmo.second)

    os.mkdir(new_dir)

    return new_dir

def masktype_augments(source, dest, save_original=False, final_size=64):
    """
    Applied the following transforms to images:
    - rotation, 15 deg in either direction
    - vertical flipping
    """

    # iterate through images in the original folder
    for img_name in os.listdir(source):

        to_save = {}

        # Load the original image            
        img_path = os.path.join(source, img_name)
        img = imread(img_path) # The img is automatically divided by 255 when loaded into a tensor, so we dont do it here
        
        if (len(img.shape)>3):
            img = rgba2rgb(img)

        # Start creating transforms
        
        # rotations
        for i in [15, -15]:
            if i == -15:
                label = "neg15"
            else:
                label = "15"

            spath = os.path.join(dest, "rot_{}".format(label) + img_name)
            rot = t.rotate(img, angle=i, cval=1, mode="edge", resize=True)
            to_save[spath] = rot

        # Vertical mirroring
        spath = os.path.join(dest, "vflip_" + img_name)
        vflipped = np.fliplr(img)
        to_save[spath] = vflipped

        # save the original
        if save_original:
            img = img_as_ubyte(img)
            to_save[os.path.join(dest, img_name)] = img

        for save_path, image in to_save.items():
            image = img_as_ubyte(image)
            try:
                imsave(save_path, image)
            except:
                try:
                    image = rgba2rgb(image)
                    imsave(save_path, image)
                except:
                    print("Error while converting rgba img to rgb")
                    print("{}".format(save_path))
                    continue

if __name__ == "__main__":

    # for c, p in paths_cropped.items():
    #     if "procedural" not in p:
    #         continue
    #     dest = make_aug_dir(p, overwrite=False)
    #     print(dest)
    #     masktype_augments(p, dest)
    paths_dict = get_paths("paths_aug_unbalanced_race")
    for _class, dir in paths_dict.items():
        for path, subdirs, files in os.walk(dir):
            # for name in files:
            #
            if not subdirs:
                print("--------------- AUGMENTING {}".format(path))
                masktype_augments(path, path, save_original=False)