import glob
import numpy as np
import scipy.misc as misc
import scipy.ndimage.morphology as morph
import scipy.ndimage.filters as filters
from scipy import ndimage
import random
from config import *
import h5py
import hickle as hkl
import re
import pdb
import pickle
import os, os.path
import imageio
import skimage.transform as transform
import sys
import warnings
warnings.filterwarnings("ignore")


def normalize(img):
    return (img - img.min())/(img.max()-img.min())


def rand_im_resize(im, mass = None, mask = None, target_size = (2750, 1500)):
    rescale_range = [ float(target_size[k]) / im.shape[k] for k in [0, 1] ]
    rescale_factor = np.random.uniform(min(rescale_range), max(rescale_range))
    im = misc.imresize(im, rescale_factor)
    if mask is None:
        return im
    elif mass is not None:
        mask = misc.imresize(mask, rescale_factor)
        mass = misc.imresize(mass, rescale_factor)
        return im, mask, mass
    else:
        mask = misc.imresize(mask, rescale_factor)
        return im, mask


def generate_cpatches(nsamples, patches_path ='./patches.npz' ,ctype=None):
    print("g",nsamples)
    combined_dims = 4

    while True:
        counts = nsamples
        #n_samples -> n_experiments, probabity sample_rates
        X_all = None
        #load all images from each class (i)
        X_masks = np.load(patches_path)['x_mask']
        X_reals = np.load(patches_path)['x_real']
        try:
            # get (c) images
            shuffle_idx = np.random.choice(len(X_masks), counts, replace=False)
        except:
            pdb.set_trace()
        
        #get mask and reals
        X_masks = X_masks[shuffle_idx]
        X_reals = X_reals[shuffle_idx]
        label = np.array([0,1]) #always pathological
        y_group = np.repeat(label.reshape((1, 1, 1, -1)), counts, axis=0)

        X_ = []
        for i,_ in enumerate(X_masks):
            X_mask = X_masks[i].reshape((patch_size, patch_size))
            X_real = normalize(X_reals[i].reshape((patch_size, patch_size)))

            X_rand = np.random.uniform(0, 1, X_mask.shape)*X_mask
            X_corrupt = (np.multiply(X_real, np.logical_not(X_mask).astype(int))+X_rand)
            boundary = np.multiply(np.invert(morph.binary_erosion(X_mask)), X_mask)
            X_boundary = normalize(filters.gaussian_filter(255.0*boundary,10)).reshape((patch_size, patch_size, 1))

            X_combined = np.concatenate((X_corrupt.reshape(patch_size, patch_size, 1), 
                                        X_mask.reshape(patch_size, patch_size, 1),
                                        X_real.reshape(patch_size, patch_size, 1),
                                        X_boundary), 
                                        axis=-1)
            X_.append(X_combined)
        X_ = np.stack(X_)
        print("Patches size {},{}".format(X_.shape,y_group.shape))
        shuffle_idx = np.random.choice(len(X_), len(X_), replace=False)
        yield X_[shuffle_idx], y_group[shuffle_idx]

def generate_nc_patches(nsamples, patches_path ='./patches.npz',
                        cancer_path = './non_cancer_patches.npz',
                        ctype=None):
    print(nsamples)
    combined_dims = 4

    while True:
        counts = nsamples
        #n_samples -> n_experiments, probabity sample_rates
        X_all = None
        #load all images from each class (i)
        X_masks = np.load(patches_path)['x_mask']
        X_reals = np.load(cancer_path)['x_real']
        try:
            # get (c) images
            shuffle_idx_mask = np.random.choice(len(X_masks), counts, replace=False)
            shuffle_idx_cancer  = np.random.choice(len(X_reals), counts, replace=False)
        except:
            pdb.set_trace()
        
        #get mask and reals
        X_masks = X_masks[shuffle_idx_mask]
        X_reals = X_reals[shuffle_idx_cancer]
        label = np.array([0,1]) #always pathological
        y_group = np.repeat(label.reshape((1, 1, 1, -1)), counts, axis=0)

        X_ = []
        for i,_ in enumerate(X_masks):
            X_mask = X_masks[i].reshape((patch_size, patch_size))
            X_real = normalize(X_reals[i].reshape((patch_size, patch_size)))

            X_rand = np.random.uniform(0, 1, X_mask.shape)*X_mask
            X_corrupt = (np.multiply(X_real, np.logical_not(X_mask).astype(int))+X_rand)
            boundary = np.multiply(np.invert(morph.binary_erosion(X_mask)), X_mask)
            X_boundary = normalize(filters.gaussian_filter(255.0*boundary,10)).reshape((patch_size, patch_size, 1))

            X_combined = np.concatenate((X_corrupt.reshape(patch_size, patch_size, 1), 
                                        X_mask.reshape(patch_size, patch_size, 1),
                                        X_real.reshape(patch_size, patch_size, 1),
                                        X_boundary), 
                                        axis=-1)
            X_.append(X_combined)
        X_ = np.stack(X_)
        print("Patches_ncexirt size {},{}".format(X_.shape,y_group.shape))
        shuffle_idx = np.random.choice(len(X_), len(X_), replace=False)
        yield X_[shuffle_idx], y_group[shuffle_idx]
