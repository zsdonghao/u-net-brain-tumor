#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
import numpy as np
import nibabel as nib
import os

# More to tl.prepro
# def find_contours(x, level=0.8, fully_connected='low', positive_orientation='low'):
#     """ Find iso-valued contours in a 2D array for a given level value, returns list of (n, 2)-ndarrays
#     see `skimage.measure.find_contours <http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.find_contours>`_ .
#
#     Parameters
#     ------------
#     x : 2D ndarray of double. Input data in which to find contours.
#     level : float. Value along which to find contours in the array.
#     fully_connected : str, {‘low’, ‘high’}.  Indicates whether array elements below the given level value are to be considered fully-connected (and hence elements above the value will only be face connected), or vice-versa. (See notes below for details.)
#     positive_orientation : either ‘low’ or ‘high’. Indicates whether the output contours will produce positively-oriented polygons around islands of low- or high-valued elements. If ‘low’ then contours will wind counter-clockwise around elements below the iso-value. Alternately, this means that low-valued elements are always on the left of the contour.
#     """
#     return skimage.measure.find_contours(x, level, fully_connected='low', positive_orientation='low')
#
# def pt2map(list_points=[], size=(100, 100), val=1):
#     """ Inputs a list of points, return a 2D image.
#
#     Parameters
#     --------------
#     list_points : list of [x, y].
#     size : tuple of (w, h) for output size.
#     val : float or int for the contour value.
#     """
#     i_m = np.zeros(size)
#     if list_points == []:
#         return i_m
#     for xx in list_points:
#         for x in xx:
#             # print(x)
#             i_m[int(np.round(x[0]))][int(np.round(x[1]))] = val
#     return i_m
#
# def binary_dilation(x, radius=3):
#     """ Return fast binary morphological dilation of an image.
#     see `skimage.morphology.binary_dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.binary_dilation>`_.
#
#     Parameters
#     -----------
#     x : 2D array image.
#     radius : int for the radius of mask.
#     """
#     from skimage.morphology import disk, binary_dilation
#     mask = disk(radius)
#     x = binary_dilation(image, selem=mask)
#     return x
#
# def dilation(x, radius=3):
#     """ Return greyscale morphological dilation of an image,
#     see `skimage.morphology.dilation <http://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.dilation>`_.
#
#     Parameters
#     -----------
#     x : 2D array image.
#     radius : int for the radius of mask.
#     """
#     from skimage.morphology import disk, dilation
#     mask = disk(radius)
#     x = dilation(x, selem=mask)
#     return x


########
def get_one_contour(x, nw, nh):
    """

    """
    x.shape = (nw, nh) # remove the last dim 3d-> 2d
    lp = find_contours(x, level=0.8, fully_connected='low', positive_orientation='low')
    # if lp != []:
        # lp = np.array(lp)
        # lp = lp[0,:,:]
        # lp[:,[0,1]] = lp[:,[1,0]]
    # if lp != []:
    #     x = pt2map(list_points=lp[0], size=(nw, nh), val=1)  # only 1 contour
    # else:
    x = pt2map(list_points=lp, size=(nw, nh), val=1) # on contour
    x = dilation(x, radius=1)
    x = x[:,:,np.newaxis]  # 2d->3d
    return x

def normalize_img(x):
    """
    Standardization vs. normalization http://www.dataminingblog.com/standardization-vs-normalization/
    """
    # x /= np.max(x)
    # x = (x - np.min(x)) / (np.max(x) - np.min(x))   # [0, 1]
    x = x/ 255. # for python2, need to make sure it is divided by float
    return x

def distort_img(data):
    x, y = data
    x = normalize_img(x)
    ## shape
    x, y = flip_axis_multi([x, y], axis=0, is_random=True)  # up down
    x, y = flip_axis_multi([x, y], axis=1, is_random=True)  # left right
    x, y = elastic_transform_multi([x, y], alpha=255 * 3, sigma=255 * 0.15, is_random=True)
    # x, y = swirl_multi([x, y], strength=4, radius=150, is_random=True,  mode='constant')
    x, y = rotation_multi([x, y], rg=20, is_random=True, fill_mode='constant')
    x, y = shift_multi([x, y], wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    x, y = shear_multi([x, y], 0.05, is_random=True, fill_mode='constant')
    x, y = zoom_multi([x, y], zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
    ## value
    x = brightness(x, gamma=0.05, is_random=True)
    return x, y

def distort_imgs(data, label='whole'):
    # if label == 'label1': # use this if you only want to use T1c and T2
    #     x1, x2, y = data
    #     x1, x2, y = flip_axis_multi([x1, x2, y], axis=1, is_random=True) # left right
    #     # x1, x2, y = elastic_transform_multi([x1, x2, y], alpha=255 * 3, sigma=255 * 0.15, is_random=True)
    #     # # x, y = swirl_multi([x, y], strength=4, radius=150, is_random=True,  mode='constant')
    #     x1, x2, y = rotation_multi([x1, x2, y], rg=20, is_random=True, fill_mode='constant')
    #     x1, x2, y = shift_multi([x1, x2, y], wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    #     #x1, x2, y = shear_multi([x1, x2, y], 0.05, is_random=True, fill_mode='constant')
    #     x1, x2, y = zoom_multi([x1, x2, y], zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
    #     # ## value
    #     # # gamma = np.random.uniform(1-0.05, 1+0.05)
    #     #x1, x2 = brightness_multi([x1, x2], gamma=0.05, is_random=False)
    #
	# # return x1, x2, y
    #     # x1, x2, y = data
    #     # x1, x2, y = flip_axis_multi([x1, x2, y], axis=1, is_random=True) # left right
    #     # x1, x2, y = elastic_transform_multi([x1, x2, y], alpha=255 * 3, sigma=255 * 0.15, is_random=True)
    #     # # x, y = swirl_multi([x, y], strength=4, radius=150, is_random=True,  mode='constant')
    #     # x1, x2, y = rotation_multi([x1, x2, y], rg=20, is_random=True, fill_mode='constant')
    #     # x1, x2, y = shift_multi([x1, x2, y], wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    #     # # x1, x2, y = shear_multi([x1, x2, y], 0.05, is_random=True, fill_mode='constant')
    #     # x1, x2, y = zoom_multi([x1, x2, y], zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
    #     # ## value
    #     # # gamma = np.random.uniform(1-0.05, 1+0.05)
    #     # # x1 = brightness(x1, gamma=gamma, is_random=False)
    #     # # x2 = brightness(x2, gamma=gamma, is_random=False)
    #     return x1, x2, y
    # else:
    x1, x2, x3, x4, y = data
    # x1 = normalize_img(x1)
    # x2 = normalize_img(x2)
    # x3 = normalize_img(x3)
    # x4 = normalize_img(x4)
    ## shape
    # x1, x2, x3, x4, y = flip_axis_multi([x1, x2, x3, x4, y], axis=0, is_random=True) # up down
    x1, x2, x3, x4, y = flip_axis_multi([x1, x2, x3, x4, y], axis=1, is_random=True) # left right
    x1, x2, x3, x4, y = elastic_transform_multi([x1, x2, x3, x4, y], alpha=255 * 3, sigma=255 * 0.15, is_random=True)
    # x, y = swirl_multi([x, y], strength=4, radius=150, is_random=True,  mode='constant')
    x1, x2, x3, x4, y = rotation_multi([x1, x2, x3, x4, y], rg=20, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = shift_multi([x1, x2, x3, x4, y], wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = shear_multi([x1, x2, x3, x4, y], 0.05, is_random=True, fill_mode='constant')
    x1, x2, x3, x4, y = zoom_multi([x1, x2, x3, x4, y], zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
    ## value
    # gamma = np.random.uniform(1-0.05, 1+0.05)
    # x1 = brightness(x1, gamma=gamma, is_random=False)
    # x2 = brightness(x2, gamma=gamma, is_random=False)
    # x3 = brightness(x3, gamma=gamma, is_random=False)
    # x4 = brightness(x4, gamma=gamma, is_random=False)
    return x1, x2, x3, x4, y

## old implementation
# def distort_img(data):
#     x, y = data
#     x = normalize_img(x)
#     ## shape
#     x, y = flip_axis_multi([x, y], axis=0, is_random=True)
#     x, y = flip_axis_multi([x, y], axis=1, is_random=True)
#     x, y = elastic_transform_multi([x, y], alpha=255 * 3, sigma=255 * 0.10, is_random=True)
#     # x, y = swirl_multi([x, y], strength=4, radius=150, is_random=True,  mode='constant')
#     x, y = rotation_multi([x, y], rg=20, is_random=True, fill_mode='constant')
#     x, y = shift_multi([x, y], wrg=0.10, hrg=0.10, is_random=True, fill_mode='constant')
#     x, y = shear_multi([x, y], 0.2, is_random=True, fill_mode='constant')
#     x, y = zoom_multi([x, y], zoom_range=[0.90, 1.10], is_random=True, fill_mode='constant')
#     ## value
#     x = brightness(x, gamma=0.2, is_random=True)
#     return x, y

## Load file
def read_Nifti1Image(file_dir, name):
    """
    http://nipy.org/nibabel/gettingstarted.html
    """
    img_dir = os.path.join(file_dir, name)
    img = nib.load(img_dir)
    # print("  *Name: %s Shape: %s " % (name, img.shape))
    # print(type(img))            # <class 'nibabel.nifti1.Nifti1Image'>
    # print(img.get_data_dtype() == np.dtype(np.int16))   # True
    # return np.array(img, dtype=np.float32)
    return img

def prepare_data(file_dir, file_list, label_list, shape=(), dim_order=(1,0,2), label='whole'):
    """
    X_test, y_test = prepare_data2(file_dir, f1, f1_, shape=shape, dim_order=dim_order)
    """
    print("\n * Preparing X:%s \n\n         y:%s" % (file_list, label_list))
    # data = np.empty(shape=(0,shape[1],shape[2],1))    ######## Akara : slower than list append
    # data2 = np.empty(shape=(0,shape[1],shape[2],1))
    data = []
    data2 = []
    # j = 0
    for f, f2 in zip(file_list, label_list):
        print("%s - %s" % (f, f2))
        ## read original image
        img = read_Nifti1Image(file_dir, f)
        X = img.get_data()
        X = np.transpose(X, dim_order)
        X = X[:,:,:,np.newaxis]
        ## read label image
        img = read_Nifti1Image(file_dir, f2)
        Y = img.get_data()
        Y = np.transpose(Y, dim_order)
        Y = Y[:,:,:,np.newaxis]
        ## if shape correct
        if X.shape == shape:
            for i in range(Y.shape[0]):
                ## if image exists
                if np.max(X[i]) > 0:
                    # print('%d X max:%.3f min:%.3f' % (i, np.max(X[i]), np.min(X[i])))#, np.median(X[i]))
                    ## make label binary
                    if label == 'whole':     # 1 2 3 4
                        Y[i] = (Y[i] > 0.5).astype(int)
                    elif label == 'core':    # 1 3 4
                        mask = (Y[i] != 2).astype(int)
                        Y[i] = (Y[i] > 0.5).astype(int)
                        Y[i] = Y[i] * mask
                    elif label == 'enhance': # 4
                        Y[i] = (Y[i] == 4).astype(int)
                    else:
                        raise Exception("unknow label")
                    # Y[i] = (Y[i] == 4).astype(int)
                    data.append(X[i].astype(np.float32))
                    data2.append(Y[i].astype(np.float32))
        else:
            print("    *shape doesn't match")
        ## plot an example
        # for i in range(0, data.shape[0], 1):
        #     # tl.visualize.frame(X[i,:,:,0], second=0.01, saveable=False, name='slice x:'+str(i),cmap='gray')
        #     tl.visualize.images2d(images=np.asarray([data[i,:,:,:], data2[i,:,:,:]]), second=0.01, saveable=False, name='slice x:'+str(i), dtype=None)
        # exit()
    return np.asarray(data, dtype=np.float32), np.asarray(data2, dtype=np.float32)

def prepare_data_2(file_dir, file_list_Flair, file_T1_list, file_T1c_list, file_T2_list,
            label_list, shape=(), dim_order=(1,0,2), label='whole'):
    """
    X_test, y_test = prepare_data2(file_dir, f1, f1_, shape=shape, dim_order=dim_order)
    """
    print("[prepare_data_2]")
    # print("\n * Preparing X:%s \n\n         y:%s" % (file_list, label_list))
    # data = np.empty(shape=(0,shape[1],shape[2],1))    ######## Akara : slower than list append
    # data2 = np.empty(shape=(0,shape[1],shape[2],1))
    data = []
    data2 = []
    # print(file_list_Flair)
    # exit()
    def _get(f_f, f_t1, f_t1c, f_t2, f_label):
        ## read original image
        img_f = read_Nifti1Image(file_dir, f_f)
        X_f = img_f.get_data()
        X_f = np.transpose(X_f, dim_order)
        X_f = X_f[:,:,:,np.newaxis]
        # print(X_f.shape, X_f.max())
        img_t1 = read_Nifti1Image(file_dir, f_t1)
        X_t1 = img_t1.get_data()
        X_t1 = np.transpose(X_t1, dim_order)
        X_t1 = X_t1[:,:,:,np.newaxis]
        # print(X_t1.shape, X_t1.max())
        img_t1c = read_Nifti1Image(file_dir, f_t1c)
        X_t1c = img_t1c.get_data()
        X_t1c = np.transpose(X_t1c, dim_order)
        X_t1c = X_t1c[:,:,:,np.newaxis]
        # print(X_t1c.shape, X_t1c.max())
        img_t2 = read_Nifti1Image(file_dir, f_t2)
        X_t2 = img_t2.get_data()
        X_t2 = np.transpose(X_t2, dim_order)
        X_t2 = X_t2[:,:,:,np.newaxis]
        # print(X_t2.shape, X_t2.max())

        # X = np.hstack((X_f, X_t1))
        # X = np.array([X_f, X_t1])


        # if label == 'label1': # use this if you only want to use T1c and T2
        #     X = np.concatenate((X_t1c, X_t2), axis=3)
        # else:
        X = np.concatenate((X_f, X_t1, X_t1c, X_t2), axis=3)

        # print(X.shape, X.max())

        img_l = read_Nifti1Image(file_dir, f_label)
        y = img_l.get_data()
        y = np.transpose(y, dim_order)
        y = y[:,:,:,np.newaxis]
        # print(y.shape, y.max())

        return X, y

    # for i in range(len(file_list_Flair)):
    for f_f, f_t1, f_t1c, f_t2, f_label in zip(file_list_Flair, file_T1_list, file_T1c_list, file_T2_list, label_list):
        # f_f, f_t1, f_t1c, f_t2
        print(f_f, f_t1, f_t1c, f_t2, f_label)
        # exit()
        X, Y = _get(f_f, f_t1, f_t1c, f_t2, f_label)
        # exit()
        # ## read label image
        # img = read_Nifti1Image(file_dir, f2)
        # Y = img.get_data()
        # Y = np.transpose(Y, dim_order)
        # Y = Y[:,:,:,np.newaxis]
        ## if shape correct
        # if X.shape == shape:
        for i in range(Y.shape[0]):
            ## if image exists
            if np.max(X[i]) > 0:
                # print('%d X max:%.3f min:%.3f' % (i, np.max(X[i]), np.min(X[i])))#, np.median(X[i]))
                ## make label binary
                if label == 'whole':     # 1 2 3 4
                    Y[i] = (Y[i] > 0.5).astype(int)
                elif label == 'core':    # 1 3 4
                    mask = (Y[i] != 2).astype(int)
                    Y[i] = (Y[i] > 0.5).astype(int)
                    Y[i] = Y[i] * mask
                elif label == 'enhance': # 4
                    Y[i] = (Y[i] == 4).astype(int)
                elif label in ['label1', 'necrosis']: # 1
                    Y[i] = (Y[i] == 1).astype(int)
                elif label == 'label2': # 2
                    Y[i] = (Y[i] == 2).astype(int)
                elif label == 'label3': # 3
                    Y[i] = (Y[i] == 3).astype(int)
                elif label == 'label4': # 4
                    Y[i] = (Y[i] == 4).astype(int)
                elif label == 'label234':    # 2 3 4
                    mask = (Y[i] != 1).astype(int)
                    Y[i] = (Y[i] > 0.5).astype(int)
                    Y[i] = Y[i] * mask
                else:
                    raise Exception("unknow label")
                # Y[i] = (Y[i] == 4).astype(int)
                data.append(X[i].astype(np.float32))
                data2.append(Y[i].astype(np.float32))
        # else:
        #     print("    *shape doesn't match")
        ## plot an example
        # for i in range(0, data.shape[0], 1):
        #     # tl.visualize.frame(X[i,:,:,0], second=0.01, saveable=False, name='slice x:'+str(i),cmap='gray')
        #     tl.visualize.images2d(images=np.asarray([data[i,:,:,:], data2[i,:,:,:]]), second=0.01, saveable=False, name='slice x:'+str(i), dtype=None)

        # exit()
    return np.asarray(data), np.asarray(data2)

# ## Save images (move to tl.visualize)
# def save_images(images, size, image_path):
#     def merge(images, size):
#         h, w = images.shape[1], images.shape[2]
#         img = np.zeros((h * size[0], w * size[1], 3))
#         for idx, image in enumerate(images):
#             i = idx % size[1]
#             j = idx // size[1]
#             img[j*h:j*h+h, i*w:i*w+w, :] = image
#         return img
#
#     def imsave(images, size, path):
#         return scipy.misc.imsave(path, merge(images, size))
#     assert len(images) <= size[0] * size[1], "number of images should be equal or less than size[0] * size[1] {}".format(len(images))
#     return imsave(images, size, image_path)


if __name__ == "__main__":
    pass
    # main()
