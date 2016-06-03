import glob
import os

import imageio
from PIL import Image

DIRECTORY = 'temp/'


def sort_filenames(filenames):
    return sorted(filenames, key=lambda x: int(x.split('/')[-1].split('_')[-1][:-4]))


def offline_plotting(folder_loc):
    if not os.path.exists(DIRECTORY):
        os.makedirs(DIRECTORY)
    gif_list = []
    for fileloc in glob.glob(folder_loc + "/*.png"):
        if not fileloc.split('/')[-1].split('_')[0] in gif_list:
            gif_list.append(fileloc.split('/')[-1].split('_')[0])
        im = Image.open(fileloc)
        im.save(DIRECTORY + fileloc.split('/')[-1])

    for prefix in gif_list:
        images = []
        png_list = []
        for fileloc in glob.glob(DIRECTORY + prefix + '*.png'):
            png_list.append(fileloc)
        print(png_list)
        png_list = sort_filenames(png_list)
        print(png_list)
        for fileloc in png_list:
            images.append(imageio.imread(fileloc))
        imageio.mimsave(prefix + '.gif', images, duration=0.6)


offline_plotting('time_size')
