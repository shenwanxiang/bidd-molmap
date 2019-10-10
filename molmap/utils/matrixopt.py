from scipy.signal import convolve2d
import numpy as np


def smartpadding(array, target_size, mode='constant', constant_values=0):
    """
    array: 2d array to be padded
    target_size: tuple of target array's shape
    """
    X, Y = array.shape
    M, N = target_size
    top = int(np.ceil((M-X)/2))
    bottom = int(M - X - top)
    right = int(np.ceil((N-Y)/2))
    left = int(N - Y - right)
    array_pad = np.pad(array, pad_width=[(top, bottom),
                                         (left, right)], 
                       mode=mode, 
                       constant_values=constant_values)
    
    return array_pad


def fspecial_gauss(size = 31, sigma = 10):

    """Function to mimic the 'fspecial' gaussian MATLAB function
      size should be odd value
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g


def conv2(array, kernel_size = 31, sigma = 10,  mode='same', fillvalue = 0):
    kernel = fspecial_gauss(kernel_size, sigma)
    return np.rot90(convolve2d(np.rot90(array, 2), np.rot90(kernel, 2), 
                               mode=mode, 
                               fillvalue = fillvalue), 2)




def sub2ind(array_shape, rows, cols):
    ind = rows*array_shape[1] + cols
    ind[ind < 0] = -1
    ind[ind >= array_shape[0]*array_shape[1]] = -1
    return ind.astype(np.int)



def points2array(x_vals, y_vals, target_size = (256, 256)):
    
    min_x_vals = min(x_vals)
    min_y_vals = min(y_vals)

    img = np.zeros(shape = target_size)
    M, N = target_size
    
    x_vals = x_vals - min_x_vals
    mult_x = max(x_vals) / (N - 1)
    x_vals = x_vals / mult_x + 1; # x_vals in [1, IMG_NUM_COLS]
    
    y_vals = y_vals - min_y_vals;
    mult_y = max(y_vals) / (M - 1)
    y_vals = y_vals / mult_y + 1 #y_vals in [1, img_num_rows]

    indices = sub2ind(img.shape, y_vals.round(), x_vals.round())
    
    m, n = img.shape
    img = img.reshape(m*n)
    img[indices] = 1
    img = img.reshape(m,n)

    return img, indices
