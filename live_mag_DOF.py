import os
import PySpin
import datetime
import matplotlib.pyplot as plt
import sys
import keyboard
import time
from datetime import datetime
import imageio
import math
import numpy as np
import scipy.signal as sig
from scipy.special import expit, logit
from scipy.stats import norm
import scipy.optimize as optimize
import scipy.fftpack as fftpack
import warnings
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import mean_squared_error
from scipy.spatial import KDTree

""" MODE: 10 for contrast ratio display, 9 for full display, 8 for slice display, 7 for CR vs height, 
6 for illumination curve, 5 for illumination sweep, 3 for DOF, 2 for save image, 1 for live feed, 0 for magnification """
MODE = 10
continue_recording = True # always True
PLOT_PHASES = False # plot phase vs x in mode 0
PLOT_FITS = False # plot fits in mode 0
PLOT_HIST = False # plot histogram in mode 0
SAVE_OVERLAY = False # save overlay in mode 3
DOME_POINTS = False # plot points at center of image and at dome LED positions
BIN_SIZE = 8 # bin size for binning image in mode 0
BIN_SIZE_MED = 45 # bin size for median filter on contrast ratios (mode 3)
PIXEL_SIZE = .0024 # in mm
LINES_MM = .98524 # lines/mm for mode 0
EXPOSURE_TIME = 25.0 # in microseconds
START_IN = 500 # starting index when displaying slices (modes 8, 9, 10)
END_IN = 1500 # ending index when displaying slices (modes 8, 9, 10)
CROP_H = 0
GAIN = 0.0
MAG = .6 # assumed magnification in mode 3
THRESHOLD = .2 # threshold to mark DOF in mode 3 
CROP_ROWS = 570 # # rows to crop from top and bottom in mode 3 
OVERLAY_NAME = '25mm_f14_09mag_DOF_overlay.pdf' # image name if saving overlay in mode 3 
CROP_FACTOR = 1 # < 1 automatically crops image displayed. set to 1 for no crop
CROP_MAG = .2 # relative size of image to calc magnification of 
RADIUS = 10 # radius to calculate ill curve for mode 5
CR_K = 7 # median filter k when in mode 10
ILLUMINATION_K = 7
now = datetime.now() # current date and time
date_time = now.strftime("%m-%d-%Y_%H-%M-%S")
# mode 2 saves image as both bmp and png, specify names below 
IMAGE_NAME_STRING = '35mm_.1mag_69mmwd_1lpmm_DOF_target'

persistent_points = np.array(
[[331, 1153],
[383, 1814],
[479, 1451],
[545, 2246],
[688, 1854],
[717, 975],
[721, 1270],
[728, 1593],
[877, 2126],
[891, 654],
[962, 1324],
[1055, 1044],
[1062, 1903],
[1222, 1264],
[1247, 1671],
[1265, 2255],
[1318, 869],
[1349, 1942],
[1455, 1527],
[1506, 1214],
[1655, 1751],
[1744, 2117],
[1824, 1342]]) # nominal positions of dome LEDs

# image is numpy array 
def fix_rotation(image):
    return np.rot90(image, k=3)

# image is numpy array 
def rotate_90(image):
    return np.rot90(image, k=1)

# crop the image by the same factor along both the x and y axis
def crop_center(img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2) 
    return img[starty:starty+cropy,startx:startx+cropx]


# crop the image by the same factor along both the x and y axis
def crop_center_mag(img, cropx, cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2) 
    endy = starty+cropy
    endx = startx+cropx
    return (img[starty:endy,startx:endx], startx, starty, endx, endy)


def crop_vertical(image, toprows, botrows):
    return image[toprows:len(image)-botrows]


def crop_horizontal(image, leftcols, rightcols):
    length = image.shape[1]
    new_image = image[:, leftcols:length-rightcols]
#     new_image = np.array([i[leftcols:length-rightcols] for i in image])
    return new_image


def mean_max_min(row):
    center = (int(np.amax(row)) + int(np.amin(row))) / 2
    flipped_row = (-1*(row - center)) + center
    max_ins = sig.find_peaks(row)
    min_ins = sig.find_peaks(flipped_row)
    mean_max = np.median(row[max_ins[0]])
    mean_min = np.median(row[min_ins[0]])
    return (mean_max, mean_min)


def contrast_ratio(max_val, min_val):
    return (max_val - min_val) / (max_val + min_val)


def calc_dof_len(in_dist, mag, pixel_size):
    return (in_dist * pixel_size) / mag


def median_filter(row, k=5):
    if k % 2 == 0:
        k += 1
    return sig.medfilt(row, k)


# takes in image name and bin size, returns binned image (numpy array)
def bin_image(im, bin_size):
    shape = im.shape
    while (len(im) % bin_size) != 0:
        im = im[:((len(im) // bin_size)*bin_size)]
    new_shape = (im.shape[0] // bin_size, im.shape[1])
    shape = (new_shape[0], im.shape[0] // new_shape[0],
             new_shape[1], im.shape[1] // new_shape[1])
    new_image = im.reshape(shape).mean(-1).mean(1)
    return new_image


# takes in a row of pixels and outputs a list of the transition pixels
def get_transitions(row):
    row = row.squeeze()
    threshold = (row.max(axis=0) + row.min(axis=0)) / 2.
    transitions = np.diff(row > threshold, prepend=False)
    transitions = np.argwhere(transitions)[:,0]
    return transitions


def mean_strip_width_array(image, bin_size):
    binned_image = bin_image(image, bin_size)
    return np.concatenate([np.ediff1d(get_transitions(row))for row in binned_image], axis=0)


# implement magnification formula 
def magnification(pix_width, cam_pix_width, line_width):
    size_on_sensor = cam_pix_width * pix_width
    return size_on_sensor / line_width


def calc_mag_fit(image, cam_pix_width, line_width, plot_fits, plot_phases):
    im = image
    binned_image = bin_image(im, 8)
    mag_array = []
    amps = []
    phases = []
    frequencies = []
    amp_errs = []
    phase_errs = []
    freq_errs = []
    for row in binned_image:
        pi = np.pi
        b_mid = (row.max() + row.min()) / 2
        amp = (row.max() - row.min()) / 2
        row -= b_mid

        def sine(x, a1, a2, a3):
            return a1 * np.sin(a2 * x + a3)

        N = xmax = im.shape[1]
        xReal = np.linspace(0, xmax, N)

        yhat = fftpack.rfft(row)
        idx = (yhat**2).argmax()
        freqs = fftpack.rfftfreq(N, d = (xReal[1]-xReal[0])/(2*pi))
        frequency = freqs[idx]

        amplitude = row.max()
        guess = [amplitude, frequency, 0.]
        (amplitude, frequency, phase), pcov = optimize.curve_fit(
            sine, xReal, row, guess, maxfev=5000)
        
        errors = np.sqrt(np.diag(pcov))
        amp_errs.append(errors[0])
        freq_errs.append(errors[1])
        phase_errs.append(errors[2])
        
        num_pixels = pi*(1/frequency) # pi because we only want one strip width
        
        amps.append(amplitude)
        phases.append(phase)
        frequencies.append(frequency)
        
        mag = magnification(num_pixels, cam_pix_width, line_width)
        
        if plot_fits:
            plt.figure(figsize = (15, 5))
            xx = xReal
            yy = sine(xx, amplitude, frequency, phase)
            plt.plot(xReal, row, 'r', linewidth = 1, label = 'Data')
            plt.plot(xx, yy , linewidth = 1, label = 'Fit')
            plt.legend()
            plt.show()
    
        mag_array.append(mag)
        
    mag_array = np.array(mag_array)
    if plot_phases:
        plt.plot(phases)
    return mag_array


def calc_mag(image, crop_factor, bin_size, pixel_size, linesmm, plot_histogram, plot_fits, plot_phases):
    line_size = 1 / (linesmm * 2)
    hist_bins = 100
    im = image
    crop_data = crop_center_mag(im, math.floor(im.shape[1]*crop_factor), 
                                            math.floor(im.shape[0]*crop_factor))
    im = crop_data[0]
    startx = crop_data[1]
    starty = crop_data[2]
    endx = crop_data[3]
    endy = crop_data[4]
#     plt.axvline(startx, color='b')
#     plt.axvline(endx, color='b')
#     plt.axhline(starty, color='b')
#     plt.axhline(endy, color='b')
    mag_array = calc_mag_fit(im, pixel_size, line_size, plot_fits, plot_phases)
    mag_array = mag_array[np.isfinite(mag_array)]
    mean, std = norm.fit(mag_array)
    if plot_histogram:
        x = np.linspace(mag_array.min(), mag_array.max(), hist_bins)
        plt.hist(mag_array, bins=x)
        plt.axvline(mean, color='red')
        plt.show()
    plt.text(1500, 200, fr"{mean:.6f} +/- {std:.6f}", color='r') 
#     plt.savefig('25mm_f14_09mag_magnification.pdf', dpi=150)


def calc_mag_rotated(image, crop_factor, bin_size, pixel_size, linesmm, plot_histogram, plot_fits, plot_phases, ax1):
    line_size = 1 / (linesmm * 2)
    hist_bins = 100
    im = image
    crop_data = crop_center_mag(im, math.floor(im.shape[1]*crop_factor), 
                                            math.floor(im.shape[0]*crop_factor))
    im = crop_data[0]
    startx = crop_data[1]
    starty = crop_data[2]
    endx = crop_data[3]
    endy = crop_data[4]
    mag_array = calc_mag_fit(im, pixel_size, line_size, plot_fits, plot_phases)
    mag_array = mag_array[np.isfinite(mag_array)]
    mean, std = norm.fit(mag_array)
    if plot_histogram:
        x = np.linspace(mag_array.min(), mag_array.max(), hist_bins)
        plt.hist(mag_array, bins=x)
        plt.axvline(mean, color='red')
        plt.show()
    ax1.text(1500, 200, fr"{mean:.6f} +/- {std:.6f}", color='r') 
#     plt.savefig('25mm_f14_09mag_magnification.pdf', dpi=150)


def contrast_ratio_alt(height, offset):
    return (height / offset)


def sine(x, a1, a2, a3):
   return a1 * np.sin(a2 * x + a3)


def square(x, a1, a2, a3):
    return a1 * sig.square(a2*x+a3)


def find_offset(row, estimate):
    first_in = np.argwhere(row > estimate)[0]
    flipped_row = row[::-1]
    second_in = np.argwhere(flipped_row > estimate)[0]
    offset = np.nanmean(row[first_in[0]:row.shape[0] - second_in[0]])
    return offset


def sigmoid(x, a1, a2, a3, a4):
    return a1*expit(a2*np.sin(a3*x + a4))


def fit_sine(row, k, plot_fits, startx, starty, ax5=None):
    row = median_filter(row, 1)
    pi = np.pi
    b_mid = np.nanmean(row)
    amp = (np.amax(row) - np.amin(row)) / 2
    offset = find_offset(row, b_mid)
    row -= offset

    N = xmax = row.shape[0]
    xReal = np.linspace(0, xmax, N)

    yhat = fftpack.rfft(row)
    idx = (yhat**2).argmax()
    freqs = fftpack.rfftfreq(N, d = (xReal[1]-xReal[0])/(2*pi))
    frequency = freqs[idx]

    amplitude = row.max() / 2
    guess = [amplitude, frequency, 0.]
    (amplitude, frequency, phase), pcov = optimize.curve_fit(
        sine, xReal, row, guess, maxfev=5000)
    if plot_fits:
        xx = xReal
#         yy = sine(xx, amplitude, frequency, phase)
#         yy = sigmoid(xx, real_amp, steepness, freq, new_phase, (-1*np.abs(real_amp / 2)))
        yy = square(xx, real_amp, frequency, phase)
        plt.xlabel('X pixel address')
        plt.ylabel('Pixel value')
        ax5.plot(xReal[startx:starty], row[startx:starty], 'r', linewidth = 1, label = 'Data')
        ax5.plot(xx[startx:starty], yy[startx:starty], linewidth = 1, label = 'Fit')
    return amplitude, offset


def fit_square(row, k, plot_fits, startx, starty, ax5=None):
    row = median_filter(row, 1)
    pi = np.pi
    b_mid = np.nanmean(row)
    amp = (np.amax(row) - np.amin(row)) / 2
    offset = find_offset(row, b_mid)
    row -= offset

    N = xmax = row.shape[0]
    xReal = np.linspace(0, xmax, N)

    yhat = fftpack.rfft(row)
    idx = (yhat**2).argmax()
    freqs = fftpack.rfftfreq(N, d = (xReal[1]-xReal[0])/(2*pi))
    frequency = freqs[idx]

    amplitude = row.max()
    guess = [amplitude, frequency, 0.]
    (amplitude, frequency, phase), pcov = optimize.curve_fit(
        sine, xReal, row, guess, maxfev=5000)
    xx = xReal
    yy = sine(xx, amplitude, frequency, phase)
    if plot_fits:
        plt.xlabel('X pixel address')
        plt.ylabel('Pixel value')
        ax5.plot(xReal[startx:starty], row[startx:starty], 'r', linewidth = 1, label = 'Data')
        ax5.plot(xx[startx:starty], yy[startx:starty], linewidth = 1, label = 'Fit')
    return amplitude, offset, yy


def calc_DOF_alt(image, crop_pixels, method, k, ax1, ax2, ax3, ax4, ax5, crop_pixels_h):
    im0 = crop_horizontal(image, crop_pixels_h, crop_pixels_h)
    im0 = crop_vertical(im0, crop_pixels, crop_pixels) 
    im0 = fix_rotation(im0)
    plot_index = np.random.randint(im0.shape[0])
    if method == 'average':
        contrast_ratios = np.array([(contrast_ratio(*mean_max_min(row))) for row in im0])
    elif method == 'sine':
        contrast_ratios = []
        offsets = []
        heights = []
        mid_row = im0[im0.shape[0] // 2]
        i = 0
        while i < im0.shape[0]:
            if i == plot_index:
                plot_fits = True
            else:
                plot_fits = False
            sine_ret = fit_sine(im0[i], k, plot_fits, 500, 1500, ax5)
            sine_fit = sine_ret[2]
            avg_max = np.nanmean(np.where[sine_fit == np.amax(sine_fit)])
            avg_min = np.nanmean(np.where[sine_fit == np.amin(sine_fit)])
            height = (avg_max - avg_min) / 2
#             height = abs(sine_ret[0])
            heights.append(height)
            offsets.append(sine_ret[1])
            cr = contrast_ratio_alt(height, sine_ret[1])
            contrast_ratios.append(cr)
            i += 1
        contrast_ratios = np.array(contrast_ratios)
        contrast_ratios_norm = contrast_ratios / (np.amax(contrast_ratios))
        heights = np.array(heights)
        heights /= np.amax(heights)
        offsets = np.array(offsets)  
        offsets /= np.amax(offsets)
    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Offset')
    ax3.set_ylabel('CR')
    ax4.set_ylabel('Actual CR')
    ax1.set_ylim([.4, 1])
    ax2.set_ylim([.4, 1])
    ax3.set_ylim([.4, 1])
    ax4.set_ylim([.2, 1])
    ax1.plot(heights)
    ax2.plot(offsets)
    ax3.plot(contrast_ratios_norm)
    ax4.plot(contrast_ratios)
    return contrast_ratios, heights, offsets


def calc_DOF_alt_sine(image, crop_pixels, method, k, ax1, ax2, ax3, ax4, ax5, crop_pixels_h):
    im0 = crop_horizontal(image, crop_pixels_h, crop_pixels_h)
    im0 = crop_vertical(im0, crop_pixels, crop_pixels) 
    im0 = fix_rotation(im0)
    plot_index = np.random.randint(im0.shape[0])
    if method == 'average':
        contrast_ratios = np.array([(contrast_ratio(*mean_max_min(row))) for row in im0])
    elif method == 'sine':
        contrast_ratios = []
        offsets = []
        heights = []
        mid_row = im0[im0.shape[0] // 2]
        i = 0
        while i < im0.shape[0]:
            if i == plot_index:
                plot_fits = True
            else:
                plot_fits = False
            sine_ret = fit_sine(im0[i], k, plot_fits, 500, 1500, ax5)
            sine_fit = sine_ret[2]
            avg_max = np.nanmean(np.where[sine_fit == np.amax(sine_fit)])
            avg_min = np.nanmean(np.where[sine_fit == np.amin(sine_fit)])
            height = (avg_max - avg_min) / 2
#             height = abs(sine_ret[0])
            heights.append(height)
            offsets.append(sine_ret[1])
            cr = contrast_ratio_alt(height, sine_ret[1])
            contrast_ratios.append(cr)
            i += 1
        contrast_ratios = np.array(contrast_ratios)
        contrast_ratios_norm = contrast_ratios / (np.amax(contrast_ratios))
        heights = np.array(heights)
        heights /= np.amax(heights)
        offsets = np.array(offsets)  
        offsets /= np.amax(offsets)
    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Offset')
    ax3.set_ylabel('CR')
    ax4.set_ylabel('Actual CR')
    ax1.set_ylim([.4, 1])
    ax2.set_ylim([.4, 1])
    ax3.set_ylim([.4, 1])
    ax4.set_ylim([.2, 1])
    ax1.plot(heights)
    ax2.plot(offsets)
    ax3.plot(contrast_ratios_norm)
    ax4.plot(contrast_ratios)
    return contrast_ratios, heights, offsets


def full_display(image, crop_pixels, method, k, ax1, ax2, ax3, ax4, ax6, ax7, ax8, crop_pixels_h, startx, starty):
    im0 = crop_horizontal(image, crop_pixels_h, crop_pixels_h)
    im0 = crop_vertical(im0, crop_pixels, crop_pixels) 
    im0 = fix_rotation(im0)
    plot_indices = np.array([image.shape[1] // 3, image.shape[1] // 2, (image.shape[1] // 3)*2])
    axes = np.array([ax6, ax7, ax8])
    if method == 'average':
        contrast_ratios = np.array([(contrast_ratio(*mean_max_min(row))) for row in im0])
    elif method == 'sine':
        contrast_ratios = []
        offsets = []
        heights = []
        mid_row = im0[im0.shape[0] // 2]
        i = 0
        flag = 0
        while i < im0.shape[0]:
            axis = None
            if i in plot_indices:
                plot_fits = True
                axis = axes[flag]
                flag += 1
            else:
                plot_fits = False
            sine_ret = fit_sine(im0[i], k, plot_fits, startx, starty, axis)
            height = abs(sine_ret[0])
            heights.append(height)
            offsets.append(sine_ret[1])
            cr = contrast_ratio_alt(height, sine_ret[1])
            contrast_ratios.append(cr)
            i += 1
        contrast_ratios = np.array(contrast_ratios)
        contrast_ratios_norm = contrast_ratios / (np.amax(contrast_ratios))
        heights = np.array(heights)
        heights /= np.amax(heights)
        offsets = np.array(offsets)  
        offsets /= np.amax(offsets)
    ax1.set_ylabel('Amplitude')
    ax2.set_ylabel('Offset')
    ax3.set_ylabel('CR')
    ax4.set_ylabel('Actual CR')
    ax1.set_ylim([.4, 1])
    ax2.set_ylim([.4, 1])
    ax3.set_ylim([.4, 1])
    ax4.set_ylim([.2, 1])
    ax1.plot(heights)
    ax2.plot(offsets)
    ax3.plot(contrast_ratios_norm)
    ax4.plot(contrast_ratios)
    return contrast_ratios, heights, offsets


def calc_DOF(im, orig, threshold, mag, pixel_size, bin_size, savefig, save_name):
    im = fix_rotation(im)
    contrast_ratios = []
    offsets = []
    heights = []
    mid_row = im[im.shape[0] // 2]
    i = 0
    while i < im.shape[0]:

        sine_ret = fit_sine(im[i], 25, False, 500, 1500, ax5)
        sine_fit = sine_ret[2]
        avg_max = np.nanmean(np.where[sine_fit == np.amax(sine_fit)])
        avg_min = np.nanmean(np.where[sine_fit == np.amin(sine_fit)])
        height = (avg_max - avg_min) / 2
#             height = abs(sine_ret[0])
        heights.append(height)
        offsets.append(sine_ret[1])
        cr = contrast_ratio_alt(height, sine_ret[1])
        contrast_ratios.append(cr)
        i += 1
    contrast_ratios = np.array(contrast_ratios)
    smooth_ratios = median_filter(contrast_ratios, bin_size)
    center = (contrast_ratios.max() + contrast_ratios.min()) / 2
    max_index = np.argmax(contrast_ratios)
    max_ratio = contrast_ratios[max_index]   
    min_ins = np.argwhere(smooth_ratios < threshold*max_ratio)
    
    # get the first indices where the threshold is crossed 
    try:
        min_ins = np.array([np.amax(np.where(min_ins<max_index, min_ins, -np.inf)), 
                            np.amin(np.where(min_ins>max_index, min_ins, np.inf))])
    except ValueError:
        pass
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
    try:
        y_index = np.nanmean(min_ins)
    except RuntimeWarning:
        y_index = np.NaN()
    plt.plot(smooth_ratios*orig.shape[0], 'r-')
    plt.axvline(y_index, color='y', linewidth=1)
    plt.vlines(min_ins, 0, orig.shape[0], color='b', linewidth=1)
    if savefig:
        plt.savefig(save_name, format='pdf', dpi=150)
    pix_len = 1
    if len(min_ins) >= 2:
        pix_len = abs(min_ins[1] - min_ins[0])
    dof = calc_dof_len(pix_len, mag, pixel_size)*np.sqrt(2)
    print("DOF in mm: ", dof)
    return dof


def calc_center_ill(image, radius):
    side = 2*radius
    centerx = image.shape[1] // 2
    centery = image.shape[0] // 2
    startx = centerx - radius
    starty = centery - radius
    endx = centerx + radius
    endy = centery + radius
    plt.axvline(startx, color='b')
    plt.axvline(endx, color='b')
    plt.axhline(starty, color='b')
    plt.axhline(endy, color='b')
    center_pixels = image[starty:endy, startx:endx]
    ill = (np.sum(center_pixels)) // (side**2)
    plt.text(1850, 200, ill, color='r')
    return ill


def mid_row_plot(image, k, ax1):
    mid_row = image[image.shape[0] // 2]
    mid_row = median_filter(mid_row, k)
    ax1.plot(mid_row)


def plot_slices_horiz(image, ax2, ax3, ax4, start_in, end_in, plotting):
    in_1 = image.shape[0] // 3
    in_2 = image.shape[0] // 2
    in_3 = in_1*2
    rotated_im = image
    slice1 = rotated_im[in_1]
    slice2 = rotated_im[in_2]
    slice3 = rotated_im[in_3]
    if plotting:
        ax2.plot(slice1[start_in:end_in])
        ax3.plot(slice2[start_in:end_in], color='red')
        ax4.plot(slice3[start_in:end_in])
    return in_1, in_2, in_3, slice1, slice2, slice3


def plot_slices(image, ax2, ax3, ax4, start_in, end_in, plotting):
    in_1 = image.shape[1] // 3
    in_2 = image.shape[1] // 2
    in_3 = in_1*2
    rotated_im = rotate_90(image)
    slice1 = rotated_im[in_1]
    slice2 = rotated_im[in_2]
    slice3 = rotated_im[in_3]
    if plotting:
        ax2.plot(slice1[start_in:end_in])
        ax3.plot(slice2[start_in:end_in], color='red')
        ax4.plot(slice3[start_in:end_in])
    return in_1, in_2, in_3


def configure_exposure(cam):
    """
     This function configures a custom exposure time. Automatic exposure is turned
     off in order to allow for the customization, and then the custom setting is
     applied.

     :param cam: Camera to configure exposure for.
     :type cam: CameraPtr
     :return: True if successful, False otherwise.
     :rtype: bool
    """

    print('*** CONFIGURING EXPOSURE ***\n')

    try:
        result = True

        # Turn off automatic exposure mode
        #
        # *** NOTES ***
        # Automatic exposure prevents the manual configuration of exposure
        # times and needs to be turned off for this example. Enumerations
        # representing entry nodes have been added to QuickSpin. This allows
        # for the much easier setting of enumeration nodes to new values.
        #
        # The naming convention of QuickSpin enums is the name of the
        # enumeration node followed by an underscore and the symbolic of
        # the entry node. Selecting "Off" on the "ExposureAuto" node is
        # thus named "ExposureAuto_Off".
        #
        # *** LATER ***
        # Exposure time can be set automatically or manually as needed. This
        # example turns automatic exposure off to set it manually and back
        # on to return the camera to its default state.

        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print('Unable to disable automatic exposure. Aborting...')
            return False

        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
        print('Automatic exposure disabled...')

        # Set exposure time manually; exposure time recorded in microseconds
        #
        # *** NOTES ***
        # Notice that the node is checked for availability and writability
        # prior to the setting of the node. In QuickSpin, availability and
        # writability are ensured by checking the access mode.
        #
        # Further, it is ensured that the desired exposure time does not exceed
        # the maximum. Exposure time is counted in microseconds - this can be
        # found out either by retrieving the unit with the GetUnit() method or
        # by checking SpinView.

        if cam.ExposureTime.GetAccessMode() != PySpin.RW:
            print('Unable to set exposure time. Aborting...')
            return False

        # Ensure desired exposure time does not exceed the maximum
        exposure_time_to_set = EXPOSURE_TIME
        exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
        cam.ExposureTime.SetValue(exposure_time_to_set)
        print('Shutter time set to %s us...\n' % exposure_time_to_set)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def reset_exposure(cam):
    """
    This function returns the camera to a normal state by re-enabling automatic exposure.

    :param cam: Camera to reset exposure on.
    :type cam: CameraPtr
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    try:
        result = True

        # Turn automatic exposure back on
        #
        # *** NOTES ***
        # Automatic exposure is turned on in order to return the camera to its
        # default state.

        if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
            print('Unable to enable automatic exposure (node retrieval). Non-fatal error...')
            return False

        cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)

        print('Automatic exposure enabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def configure_gain(cam):

    print('*** CONFIGURING GAIN ***\n')

    try:
        result = True

        if cam.GainAuto.GetAccessMode() != PySpin.RW:
            print('Unable to disable automatic gain. Aborting...')
            return False

        cam.GainAuto.SetValue(PySpin.GainAuto_Off)
        print('Automatic gain disabled...')

    
        if cam.Gain.GetAccessMode() != PySpin.RW:
            print('Unable to set gain . Aborting...')
            return False

        # Ensure desired exposure time does not exceed the maximum
        gain_to_set = GAIN
        cam.Gain.SetValue(gain_to_set)
        print('Gain set to %s ...\n' % gain_to_set)

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result


def reset_gain(cam):
    try:
        result = True

        if cam.GainAuto.GetAccessMode() != PySpin.RW:
            print('Unable to enable gain exposure (node retrieval). Non-fatal error...')
            return False

        cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)

        print('Automatic gain enabled...')

    except PySpin.SpinnakerException as ex:
        print('Error: %s' % ex)
        result = False

    return result 


ill_list = []

if MODE in (0, 1, 3, 5, 6, 7, 8, 9, 10):
   get_ipython().run_line_magic('matplotlib', 'qt')
else:
   get_ipython().run_line_magic('matplotlib', 'inline')

def handle_close(evt):
   """
   This function will close the GUI when close event happens.

   :param evt: Event that occurs when the figure closes.
   :type evt: Event
   """

   global continue_recording
   continue_recording = False


def acquire_and_display_images(cam, nodemap, nodemap_tldevice):
   """
   This function continuously acquires images from a device and display them in a GUI.

   :param cam: Camera to acquire images from.
   :param nodemap: Device nodemap.
   :param nodemap_tldevice: Transport layer device nodemap.
   :type cam: CameraPtr
   :type nodemap: INodeMap
   :type nodemap_tldevice: INodeMap
   :return: True if successful, False otherwise.
   :rtype: bool
   """
   global continue_recording

   sNodemap = cam.GetTLStreamNodeMap()

   # Change bufferhandling mode to NewestOnly
   node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
   if not PySpin.IsAvailable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
       print('Unable to set stream buffer handling mode.. Aborting...')
       return False

   # Retrieve entry node from enumeration node
   node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
   if not PySpin.IsAvailable(node_newestonly) or not PySpin.IsReadable(node_newestonly):
       print('Unable to set stream buffer handling mode.. Aborting...')
       return False

   # Retrieve integer value from entry node
   node_newestonly_mode = node_newestonly.GetValue()

   # Set integer value from entry node as new value of enumeration node
   node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

   print('*** IMAGE ACQUISITION ***\n')
   try:
       node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
       if not PySpin.IsAvailable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
           print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
           return False

       # Retrieve entry node from enumeration node
       node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
       if not PySpin.IsAvailable(node_acquisition_mode_continuous) or not PySpin.IsReadable(
               node_acquisition_mode_continuous):
           print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
           return False

       # Retrieve integer value from entry node
       acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

       # Set integer value from entry node as new value of enumeration node
       node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

       print('Acquisition mode set to continuous...')

       #  Begin acquiring images
       #
       #  *** NOTES ***
       #  What happens when the camera begins acquiring images depends on the
       #  acquisition mode. Single frame captures only a single image, multi
       #  frame catures a set number of images, and continuous captures a
       #  continuous stream of images.
       #
       #  *** LATER ***
       #  Image acquisition must be ended when no more images are needed.
       cam.BeginAcquisition()

       print('Acquiring images...')

       #  Retrieve device serial number for filename
       #
       #  *** NOTES ***
       #  The device serial number is retrieved in order to keep cameras from
       #  overwriting one another. Grabbing image IDs could also accomplish
       #  this.
       device_serial_number = ''
       node_device_serial_number = PySpin.CStringPtr(nodemap_tldevice.GetNode('DeviceSerialNumber'))
       if PySpin.IsAvailable(node_device_serial_number) and PySpin.IsReadable(node_device_serial_number):
           device_serial_number = node_device_serial_number.GetValue()
           print('Device serial number retrieved as %s...' % device_serial_number)

       # Close program
       print('Press enter to close the program..')

       # Figure(1) is default so you can omit this line. Figure(0) will create a new window every time program hits this line
       # Close the GUI when close event happens
       if MODE in (1, 2, 3, 4, 5, 6):
           fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, dpi=200)
           fig.canvas.mpl_connect('close_event', handle_close)
       # Retrieve and display images

       while(continue_recording):
           try:

               #  Retrieve next received image
               #
               #  *** NOTES ***
               #  Capturing an image houses images on the camera buffer. Trying
               #  to capture an image that does not exist will hang the camera.
               #
               #  *** LATER ***
               #  Once an image from the buffer is saved and/or no longer
               #  needed, the image must be released in order to keep the
               #  buffer from filling up.
               
               image_result = cam.GetNextImage(1000)

               #  Ensure image completion
               if image_result.IsIncomplete():
                   print('Image incomplete with image status %d ...' % image_result.GetImageStatus())

               else:                    
                   # Getting the image data as a numpy array
                   image_data = image_result.GetNDArray()  
                   image_data = crop_center(image_data, math.floor(image_data.shape[1]*CROP_FACTOR), 
                                           math.floor(image_data.shape[0]*CROP_FACTOR))
                   rotated_image_data = fix_rotation(image_data)
                   if MODE == 0:
                       plt.imshow(image_data, cmap='gray')
                       plt.show(block=False)
                       calc_mag(image_data, CROP_MAG, BIN_SIZE, PIXEL_SIZE, LINES_MM, PLOT_FITS, PLOT_HIST, PLOT_PHASES)
                       plt.pause(.0001)
                       plt.clf()
                   elif MODE == 1:
                       plt.axvline(image_data.shape[1] / 2, color='red', linewidth=.5)
                       plt.axhline(image_data.shape[0] / 2, color='blue', linewidth=.5)
                       plt.imshow(image_data, cmap='gray')
                       if DOME_POINTS:
                           plt.plot(image_data.shape[1] / 2, image_data.shape[0] / 2, "b.")
                           for point in persistent_points:
                               plt.plot(point[1], point[0], "r+", alpha=.5)
                       plt.show(block=False)
                       plt.pause(.0001)
                       plt.clf()
                   elif MODE == 2:
                       IMAGE_NAME = fr'{IMAGE_NAME_STRING}_{date_time}.png'
                       IMAGE_NAME_BMP = fr'{IMAGE_NAME_STRING}_{date_time}.bmp'
                       plt.imshow(image_data, cmap='gray')
                       if DOME_POINTS:
                           plt.plot(image_data.shape[1] / 2, image_data.shape[0] / 2, "b.")
                           for point in persistent_points:
                               plt.plot(point[1], point[0], "r+")
                           plt.savefig(IMAGE_NAME)
                       else:
                           plt.imsave(IMAGE_NAME, image_data, cmap='gray', format='png')
                           plt.imsave(IMAGE_NAME_BMP, image_data, cmap='gray', format='bmp')
                       plt.pause(10)
                       plt.clf()
                   elif MODE == 3:
                       vert_image = crop_vertical(rotated_image_data, CROP_ROWS, CROP_ROWS)
                       calc_mag(image_data, CROP_MAG, BIN_SIZE, PIXEL_SIZE, LINES_MM, 
                                PLOT_FITS, PLOT_HIST, PLOT_PHASES)
                       plt.axvline(rotated_image_data.shape[1] / 2, color='red')
                       plt.axhline(rotated_image_data.shape[0] / 2, color='blue')
                       plt.imshow(rotated_image_data, cmap='gray', origin='lower')
                       calc_DOF(vert_image, rotated_image_data, THRESHOLD, MAG, PIXEL_SIZE, BIN_SIZE_MED, SAVE_OVERLAY, OVERLAY_NAME)
                       plt.show(block=False)
                       plt.pause(.0001)
                       plt.clf()
                   elif MODE == 4:
                       plt.axvline(image_data.shape[1] / 2, color='red', linewidth=.5)
                       plt.axhline(image_data.shape[0] / 2, color='blue', linewidth=.5)
                       plt.imshow(image_data, cmap='gray', origin='lower')
                       if DOME_POINTS:
                           plt.plot(image_data.shape[1] / 2, image_data.shape[0] / 2, "b.")
                           for point in persistent_points:
                               plt.plot(point[1], point[0], "r+")
                       plt.show(block=False)
                       plt.pause(.0001)
                       plt.clf()
                   elif MODE == 5:
                       plt.imshow(image_data, cmap='gray')
                       ill = calc_center_ill(image_data, RADIUS)
                       ill_list.append(ill)
                       plt.pause(.0001)
                       plt.clf()
                   elif MODE == 6:
                       fig, (ax1, ax2) = plt.subplots(2, 1, dpi=150)
                       mid_row_plot(image_data, 7, ax1)
                       ax2.imshow(image_data, cmap='gray')
                       plt.show(block=False)
                       plt.pause(1.5)
                       plt.close()
                   elif MODE == 7:
                       fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, dpi=100, figsize=(12, 10))
#                         ax4.imshow(image_data, cmap='gray')
                       calc_DOF_alt(image_data, CROP_ROWS, 'sine', BIN_SIZE_MED, ax1, ax2, ax3, ax4, ax5, CROP_H)      
                       plt.show(block=False)
                       plt.pause(.5)
                       plt.close()
                   elif MODE == 8:
                       fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, dpi=150, figsize=(12, 10))
                       rotated_image_data = rotate_90(image_data)
                       ax1.imshow(image_data, cmap='gray')
                       calc_mag_rotated(rotated_image_data, CROP_MAG, BIN_SIZE, PIXEL_SIZE, LINES_MM, 
                                PLOT_FITS, PLOT_HIST, PLOT_PHASES, ax1)
                       in_1, in_2, in_3 = plot_slices(image_data, ax2, ax3, ax4, START_IN, END_IN, True)
                       ax1.axvline(in_1, color='blue')
                       ax1.axvline(in_2, color='red')
                       ax1.axvline(in_3, color='blue')
                       plt.pause(1)
                       plt.close()
                   elif MODE == 9:
                       fig, ((ax1, ax5), (ax2, ax6), (ax3, ax7), (ax4, ax8)) = plt.subplots(4, 2, dpi=150, figsize=(12, 12))
                       in_1, in_2, in_3 = plot_slices(image_data, ax6, ax7, ax8, START_IN, END_IN, False)
                       ax5.imshow(image_data, cmap='gray')
                       rotated_image_data = rotate_90(image_data)
                       full_display(image_data, CROP_ROWS, 'sine', BIN_SIZE_MED, ax1, ax2, ax3, 
                                    ax4, ax6, ax7, ax8, CROP_H, START_IN, END_IN)
                       calc_mag_rotated(rotated_image_data, CROP_MAG, BIN_SIZE, PIXEL_SIZE, LINES_MM, 
                                PLOT_FITS, PLOT_HIST, PLOT_PHASES, ax5)
                       ax5.axvline(in_1, color='blue')
                       ax5.axvline(in_2, color='red')
                       ax5.axvline(in_3, color='blue')
                       plt.pause(.0001)
                       plt.close()
                   elif MODE == 10:
                       fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, dpi=150, figsize=(12, 10))
                       ax1.imshow(image_data, cmap='gray')
                       calc_mag(image_data, CROP_MAG, BIN_SIZE, PIXEL_SIZE, LINES_MM, 
                                PLOT_FITS, PLOT_HIST, PLOT_PHASES)
                       in_1, in_2, in_3, slice1, slice2, slice3 = plot_slices_horiz(image_data, ax2, ax3, ax4, 500, 1500, True)
                       ax1.axhline(in_1, color='blue')
                       ax1.axhline(in_2, color='red')
                       ax1.axhline(in_3, color='blue')
                       amplitude1, offset1 = fit_sine(slice1, CR_K, PLOT_FITS, START_IN, END_IN, ax2)
                       amplitude2, offset2 = fit_sine(slice2, CR_K, PLOT_FITS, START_IN, END_IN, ax3)
                       amplitude3, offset3 = fit_sine(slice3, CR_K, PLOT_FITS, START_IN, END_IN, ax4)
                       cr_1 = amplitude1 / offset1
                       cr_2 = amplitude2 / offset2
                       cr_3 = amplitude3 / offset3
                       cr = np.nanmean(np.array([cr_1, cr_2, cr_3]))
                       ax1.text(1500, 200, cr, color='red') 
                       plt.pause(1)
                       plt.close()
           # If user presses enter, close the program
                   if keyboard.is_pressed('ENTER'):
                       print('Program is closing...')
                       
                       # Close figure
                       plt.close('all')             
                       input('Done! Press Enter to exit...')
                       continue_recording=False                        

               #  Release image
               #
               #  *** NOTES ***
               #  Images retrieved directly from the camera (i.e. non-converted
               #  images) need to be released in order to keep from filling the
               #  buffer.
               image_result.Release()

           except PySpin.SpinnakerException as ex:
               print('Error: %s' % ex)
               return False

       #  End acquisition
       #
       #  *** NOTES ***
       #  Ending acquisition appropriately helps ensure that devices clean up
       #  properly and do not need to be power-cycled to maintain integrity.
       cam.EndAcquisition()

   except PySpin.SpinnakerException as ex:
       print('Error: %s' % ex)
       return False

   return True


def run_single_camera(cam):
   """
   This function acts as the body of the example; please see NodeMapInfo example
   for more in-depth comments on setting up cameras.

   :param cam: Camera to run on.
   :type cam: CameraPtr
   :return: True if successful, False otherwise.
   :rtype: bool
   """
   try:
       result = True

       nodemap_tldevice = cam.GetTLDeviceNodeMap()

       # Initialize camera
       
       
       cam.Init()
       
       nodemap = cam.GetNodeMap()
       if not configure_gain(cam):
           return False
       if not configure_exposure(cam):
           return False
       cam.SensorShutterMode.SetValue(PySpin.SensorShutterMode_Rolling)
       cam.GammaEnable.SetValue(False)
       cam.PixelFormat.SetValue(PySpin.PixelFormat_Mono16)
       # Retrieve GenICam nodemap        
       # Acquire images
       result &= acquire_and_display_images(cam, nodemap, nodemap_tldevice)
       
       # Reset exposure
       result &= reset_exposure(cam)
       result &= reset_gain(cam)
       
       # Deinitialize camera
       cam.DeInit()

   except PySpin.SpinnakerException as ex:
       print('Error: %s' % ex)
       result = False

   return result


def main():
   """
   Example entry point; notice the volume of data that the logging event handler
   prints out on debug despite the fact that very little really happens in this
   example. Because of this, it may be better to have the logger set to lower
   level in order to provide a more concise, focused log.

   :return: True if successful, False otherwise.
   :rtype: bool
   """
   result = True
   # Retrieve singleton reference to system object
   system = PySpin.System.GetInstance()

   # Get current library version
   version = system.GetLibraryVersion()
   print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

   # Retrieve list of cameras from the system
   cam_list = system.GetCameras()

   num_cameras = cam_list.GetSize()

   print('Number of cameras detected: %d' % num_cameras)

   # Finish if there are no cameras
   if num_cameras == 0:

       # Clear camera list before releasing system
       cam_list.Clear()

       # Release system instance
       system.ReleaseInstance()

       print('Not enough cameras!')
       input('Done! Press Enter to exit...')
       return False

   # Run example on each camera
   for i, cam in enumerate(cam_list):
       print('Running example for camera %d...' % i)

       result &= run_single_camera(cam)
       print('Camera %d example complete... \n' % i)

   # Release reference to camera
   # NOTE: Unlike the C++ examples, we cannot rely on pointer objects being automatically
   # cleaned up when going out of scope.
   # The usage of del is preferred to assigning the variable to None.
   del cam

   # Clear camera list before releasing system
   cam_list.Clear()

   # Release system instance
   system.ReleaseInstance()
   if MODE == 5:
       get_ipython().run_line_magic('matplotlib', 'inline')
       np.save("ill_list.npy", np.array(ill_list))
       plt.plot(ill_list)
       plt.savefig(IMAGE_NAME_STRING, format='png')
       plt.show()
   input('Done! Press Enter to exit...')
   return result


if __name__ == '__main__':
   if main():
       sys.exit(0)
   else:
       sys.exit(1)
