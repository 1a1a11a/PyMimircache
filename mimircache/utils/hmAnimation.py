# coding=utf-8
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
        if not "_".join(fileloc.split('/')[-1].split('_')[:-1]) in gif_list:
            gif_list.append("_".join(fileloc.split('/')[-1].split('_')[:-1]))
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
        imageio.mimsave("{}/{}.gif".format(folder_loc, prefix), images, duration=0.6)
        print("saved to {}/{}.gif".format(folder_loc, prefix))



def create_animation(file_list, figname="animation.gif", **kwargs):
    """
    create animations from a list of images
    :param file_list:
    :param figname:
    :param kwargs:
    :return:
    """

    duration = kwargs.get("duration", 0.6)


    # transform to gif
    # this might be a problem in Windows and
    # when number of gif files are too large to fit in /tmp
    new_gifs_list = []
    for f in file_list:
        im = Image.open(f)
        im.save("/tmp/".format(os.path.basename(f)))
        new_gifs_list.append("/tmp/".format(os.path.basename(f)))

    # now make gifs
    images = []
    for f in new_gifs_list:
        images.append(imageio.imread(f))
    imageio.mimsave(figname, images, duration=duration)


if __name__ == "__main__":
    offline_plotting('/home/jason/pycharm/mimircache/A1a1a11a/1127EVA_Value/smallTest')
