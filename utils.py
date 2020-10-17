import os
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from sklearn.utils import shuffle


def imread(fn):
    im = Image.open(fn)
    return np.array(im)


def imsave(fn, arr):
    im = Image.fromarray(arr)
    im.save(fn)


def imresize(arr, sz):
    im = Image.fromarray(arr)
    im.resize(sz)
    return np.array(im)


def get_mnist(limit=None):
    df = pd.read_csv('train.csv.zip')
    data = df.values
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    X, Y = shuffle(X, Y)
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    return X, Y


def get_celeb():
    if not os.path.exists('../large_files/img_align_celeba-cropped'):
        # --- check for original data
        if not os.path.exists('../large_files/img_align_celeba'):
            print("Extracting img_align_celeba.zip...")
            with zipfile.ZipFile('../large_files/img_align_celeba.zip') as zf:
                zf.extractall('../large_files')
        # --- load in the original images
        filenames = glob("../large_files/img_align_celeba/*.jpg")
        N = len(filenames)
        print("Found %d files!" % N)
        # --- crop the images to 64x64
        os.mkdir('../large_files/img_align_celeba-cropped')
        print("Cropping images, please wait...")
        for i in range(N):
            crop_and_resave(filenames[i], '../large_files/img_align_celeba-cropped')
            if i % 1000 == 0:
                print("%d/%d" % (i, N))
    # --- return the cropped version
    filenames = glob("../large_files/img_align_celeba-cropped/*.jpg")
    return filenames


def crop_and_resave(inputfile, outputdir):
    # --- assume that the middle 108 pixels contain the face
    im = imread(inputfile)
    height, width, color = im.shape
    edge_h = int(round((height - 108) / 2.0))
    edge_w = int(round((width - 108) / 2.0))
    cropped = im[edge_h:(edge_h + 108), edge_w:(edge_w + 108)]
    small = imresize(cropped, (64, 64))
    filename = inputfile.split('/')[-1]
    imsave("%s/%s" % (outputdir, filename), small)


def scale_image(im):
    # --- scale to (-1, +1)
    return (im / 255.0) * 2 - 1


def files2images(filenames):
    return [scale_image(imread(fn)) for fn in filenames]


