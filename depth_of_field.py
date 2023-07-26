import imageio
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import scipy.signal as sig
import scipy.optimize as optimize

def crop_vertical(image, toprows, botrows):
    return image[toprows:len(image)-botrows]

# image is numpy array 
def rotate_90(image):
    return np.rot90(image, k=3)

def mean_max_min(row):
    center = (row.max() + row.min()) / 2
    flipped_row = (-1*(row - center)) + center
    max_ins = sig.find_peaks(row)
    min_ins = sig.find_peaks(flipped_row)
    mean_max = np.nanmean(row[max_ins[0]])
    mean_min = np.nanmean(row[min_ins[0]])
    return (mean_max, mean_min)

def contrast_ratio(max_val, min_val):
    return (max_val - min_val) / (max_val + min_val)

def calc_dof_len(in_dist, mag, pixel_size):
    return (in_dist * pixel_size) / mag

def median_filter(row, k=5):
    if k % 2 == 0:
        k += 1
    return sig.medfilt(row, k)

def calc_DOF(im, threshold, mag, pixel_size, bin_size, savefig, save_name):
    orig = im
    plt.figure(dpi=150)
    plt.imshow(im, cmap='gray', aspect='auto')
    plt.show(block=False)
    crop_pixels = 1
    while (crop_pixels != 0):
        crop_pixels = int(input("Enter number of pixels to crop off top and bottom: "))
        im = crop_vertical(im, crop_pixels, crop_pixels) 
        plt.figure(dpi=100)
        plt.imshow(im, cmap='gray')
        plt.show(block=False)
    rotate = int(input("1 to rotate image, 0 to keep the same: "))
    if rotate == 1: 
        im = rotate_90(im)
    contrast_ratios = np.array([contrast_ratio(mean_max_min(row)[0], 
                                               mean_max_min(row)[1]) for row in im])
    smooth_ratios = median_filter(contrast_ratios, bin_size)
    center = (contrast_ratios.max() + contrast_ratios.min()) / 2
    max_index = np.argmax(contrast_ratios)
    max_ratio = contrast_ratios[max_index]   
    min_ins = np.argwhere(contrast_ratios < threshold*max_ratio)
    # get the first indices where the threshold is crossed 
    min_ins = np.array([np.amax(np.where(min_ins<max_index, min_ins, -np.inf)), 
                            np.amin(np.where(min_ins>max_index, min_ins, np.inf))])
    plt.figure(dpi=150)
    plt.imshow(orig, cmap='gray')
    plt.axvline(max_index, color='y', linewidth=1)
    plt.plot((smooth_ratios*-1*orig.shape[0])+orig.shape[0], color='r')
    plt.vlines(min_ins, 0, orig.shape[0], color='b', linewidth=1)
    if savefig:
        plt.savefig(save_name, format='pdf', dpi=150)
    plt.show()
    pix_len = abs(min_ins[1] - min_ins[0])
    dof = calc_dof_len(pix_len, mag, pixel_size)
    print("DOF in mm: ", dof)

if __name__ == '__main__':
    image = imageio.imread('DOF_test_50mm_.4mag.png', as_gray=True)
    calc_DOF(image, .08, .4, .0024, 6, False, 'DOF_image_overlay_.pdf')
