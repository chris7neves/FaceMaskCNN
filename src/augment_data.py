import shutil
from util import rename_images
from configs.paths import paths_aug, paths_cropped

from datetime import datetime
import os
from shutil import rmtree

import numpy as np

import skimage.transform as t
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from skimage.color import rgba2rgb

import matplotlib.pyplot as plt


def make_aug_dir(dir_path, overwrite=False):

    # Get new folder name
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

def masktype_augments(source, dest, final_size=64):

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
            spath = os.path.join(dest, "rot_{}".format(i) + img_name)

            rot = t.rotate(img, angle=i, cval=1, mode="edge", resize=True)
            #rot_size_down = t.resize(rot, (final_size, final_size))

            #to_save[spath] = rot_size_down
            to_save[spath] = rot

        # Vertical mirroring
        spath = os.path.join(dest, "vflip_{}".format(i) + img_name)
        vflipped = np.fliplr(img)
        #vflipped_size_down = t.resize(vflipped, (final_size, final_size))
        #to_save[spath] = vflipped_size_down
        to_save[spath] = vflipped

        # save the original, scaled down
        #img = t.resize(img, (final_size, final_size))
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

        


# for c, p in paths.items():
#     rename_images(p, suffix=c.split("_")[0])

# rename_images(paths_cropped["procedural_mask"])
# dest = make_aug_dir(paths_cropped["procedural_mask"], overwrite=True)
# masktype_augments(paths_cropped["procedural_mask"], dest)

for c, p in paths_cropped.items():
    if "procedural" not in p:
        continue
    dest = make_aug_dir(p, overwrite=False)
    print(dest)
    masktype_augments(p, dest)

    #  while choice not in ['o', 'n']:
    #         choice = input("Augment folders already exist. Overwrite (o) or save new (n)?")