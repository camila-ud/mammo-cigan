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

"""
#Function modified by C.A
def generate_cpatches(nsamples, sample_rates=[0.5, 0.5], add_fake=False, limits=[None, 128], augmentation=False, is_vanilla=False, is_val=False, ctype=None):

    if len(sample_rates) == 4: sample_rates = sample_rates[0:3]

    def get_file_and_label(i):
        # CHANGE folder i = 0 : benign
        # Get random image
        if len(sample_rates) == 2:
            if ctype == 'mal':
                if i == 0: # benign
                    folder = ['nocancer/']
                    label = [1,0]
                elif i == 1:
                    folder = ['cancer/']
                    label = [0,1]
        file = None
        if add_fake:
            label.append(0)
        for f in folder:
            append = '' if limits[i] is None else '_'+str(limits[i])
            path = patches_path + append + '/' + f
            num_files = len(os.listdir(path))
            rand_id = np.random.randint(0, num_files)
            if file is not None:
                file = np.concatenate([hkl.load(path+str(rand_id)+'.hkl'), file], axis=0)
            else:
                file = hkl.load(path+str(rand_id)+'.hkl')
        return file, np.array(label)

    combined_dims = 4

    while True:
        counts = np.random.multinomial(nsamples, sample_rates)
        #n_samples -> n_experiments, probabity sample_rates
        X_all = None
        for i, c in enumerate(counts):
            if c > 0:
                #load all images from each class (i)
                file, label = get_file_and_label(i)
                try:
                    # get (c) images
                    shuffle_idx = np.random.choice(len(file), c, replace=False)
                except:
                    pdb.set_trace()
                X_group = file[shuffle_idx]
                y_group = np.repeat(label.reshape((1, 1, 1, -1)), c, axis=0)

                X_ = np.zeros(X_group.shape[0:-1]+(combined_dims,))

                print('')
                for i,X in enumerate(X_group):
                    if is_val:
                        X_mask = X[:,:,1:2]
                        X_real = X[:,:,0:1]
                    else:
                        X_mask = X[:, :, 0:1]
                        X_real = X[:, :, 1:2]

                    X_mask_ = X_mask.reshape((patch_size, patch_size))
                    X_rand = np.random.uniform(0, 255, X_mask.shape)*X_mask
                    X_corrupt = (np.multiply(X_real, np.logical_not(X_mask).astype(int))+X_rand)
                    
                    X_boundary = normalize(filters.gaussian_filter(255.*np.multiply(np.invert(morph.binary_erosion(X_mask_)), X_mask_), 10.0)).reshape((patch_size, patch_size, 1))
                    X_combined = np.concatenate((X_corrupt*1.0/255., X_mask, X_real*1.0/255., X_boundary), axis=-1)
                    X_[i] = X_combined
                if X_all is None:
                    X_all = X_
                    y_all = y_group
                else:
                    X_all = np.concatenate((X_all, X_), axis=0)
                    y_all = np.concatenate((y_all, y_group), axis=0)
        shuffle_idx = np.random.choice(len(X_all), len(X_all), replace=False)
        yield X_all[shuffle_idx], y_all[shuffle_idx]
"""
def generate_cpatches(nsamples, patches_path ='./patches.npz' ,ctype=None):
    
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
