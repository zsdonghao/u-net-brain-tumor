#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorlayer as tl
import numpy as np
import time, os
from utils import *
from tensorlayer.prepro import *
import nibabel as nib


def get_data_brats2017(label, data, file_dir):
    """ Returns X_train, y_train, X_test, y_test, nw, nh, nz
    * X (Flair, T1, T1C, T2) --> Y (label)
    * Download the dataset from XXXXXXXXXXXXXXXXXXXXX
    """
    #TODO
    nw = nh = 256
    nz = 4
    return X_train, y_train, X_test, y_test, nw, nh, nz


# def get_data_Flair(label, data): (old)
#     """ Flair --> Y label"""
#     ## get folder dir
#     cwd = os.getcwd()
#
#     ## get file lists from test dataset
#     # file_dir = os.path.join(cwd, 'InitialTrainingBrainTumour')
#     # file_Flair_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_Flair\.(nii.gz)'))
#     # file_T1_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_T1\.(nii.gz)'))
#     # file_T1c_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_T1c\.(nii.gz)'))
#     # file_T2_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_T2\.(nii.gz)'))
#     # file_TumourMask_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_TumourMask\.(nii.gz)'))
#     ## get file lists from BRATSData2015 dataset
#     file_dir = os.path.join(cwd, '/media/gyang/RAIDARRAY/Data/BRASTS_2015_Data/BRATS2015_Training/HG_LG_Training_Normalised')
#     # file_Flair_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_FLAIR_Norm.nii.gz'))[:] # don't use normalized data
#     # file_T1_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T1_Norm.nii.gz'))[:]
#     # file_T1c_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T1c_Norm.nii.gz'))[:]
#     # file_T2_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T2_Norm.nii.gz'))[:]
#     file_Flair_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_FLAIR.nii.gz', printable=False))
#     # file_T1_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T1.nii.gz'))
#     # file_T1c_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T1c.nii.gz'))
#     # file_T2_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T2.nii.gz'))
#     file_TumourMask_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_GT.nii.gz', printable=False))
#
#     # file_Flair_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_LG_FLAIR.nii.gz'))
#     # file_T1_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_LG_T1.nii.gz'))
#     # file_T1c_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_LG_T1c.nii.gz'))
#     # file_T2_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_LG_T2.nii.gz'))
#     # file_TumourMask_list = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_LG_GT.nii.gz'))
#
#     print('*file_Flair_list: %d\n %s' % (len(file_Flair_list), file_Flair_list))
#     # print('*file_T1_list: %d\n %s' % (len(file_T1_list), file_T1_list))
#     # print('*file_T1c_list: %d\n %s' % (len(file_T1c_list), file_T1c_list))
#     # print('*file_T2_list: %d\n %s' % (len(file_T2_list), file_T2_list))
#     print('*file_TumourMask_list: %d\n %s' % (len(file_TumourMask_list), file_TumourMask_list))
#
#     ## visualize the slices of an image
#     dim_order = (2,1,0)
#     img = read_Nifti1Image(file_dir, file_Flair_list[0])
#     target = read_Nifti1Image(file_dir, file_TumourMask_list[0])
#     X = img.get_data()
#     X = np.transpose(X, dim_order)
#     X = X[:,:,:,np.newaxis]
#     Y = target.get_data()
#     Y = np.transpose(Y, dim_order)
#     Y = Y[:,:,:,np.newaxis]
#
#     shape = X.shape
#     print('X.shape',shape)
#     nw = shape[1]
#     nh = shape[2]
#     print('Y max',np.max(Y))  # 1,2,3,4
#     print('X max',np.max(X))
#     # print(X[1,:,:,:].shape)
#     # print(Y[1,:,:,:].shape)
#     print(X.dtype)
#
#     f1 = file_Flair_list[0:44]
#     f2 = file_Flair_list[44:88]
#     f3 = file_Flair_list[88:132]
#     f4 = file_Flair_list[132:176]
#     f5 = file_Flair_list[176:220]
#     f1_ = file_TumourMask_list[0:44]
#     f2_ = file_TumourMask_list[44:88]
#     f3_ = file_TumourMask_list[88:132]
#     f4_ = file_TumourMask_list[132:176]
#     f5_ = file_TumourMask_list[176:220]
#
#     if data == 'f1':
#         f_train_x = f2+f3+f4+f5
#         f_train_y = f2_+f3_+f4_+f5_
#         f_test_x = f1
#         f_test_y = f1_
#     elif data == 'f2':
#         f_train_x = f1+f3+f4+f5
#         f_train_y = f1_+f3_+f4_+f5_
#         f_test_x = f2
#         f_test_y = f2_
#     elif data == 'f3':
#         f_train_x = f1+f2+f4+f5
#         f_train_y = f1_+f2_+f4_+f5_
#         f_test_x = f3
#         f_test_y = f3_
#     elif data == 'f4':
#         f_train_x = f1+f2+f3+f5
#         f_train_y = f1_+f2_+f3_+f5_
#         f_test_x = f4
#         f_test_y = f4_
#     elif data == 'f5':
#         f_train_x = f1+f2+f3+f4
#         f_train_y = f1_+f2_+f3_+f4_
#         f_test_x = f5
#         f_test_y = f5_
#     elif data == 'all':
#         f_train_x = f1+f2+f3+f4+f5
#         f_train_y = f1_+f2_+f3_+f4_+f5_
#         f_test_x = None
#         f_test_y = None
#     elif data == 'debug':
#         f_train_x = f1
#         f_train_y = f1_
#         f_test_x = f2
#         f_test_y = f2_
#
#     X_train, y_train = prepare_data(file_dir, f_train_x, f_train_y, shape=shape, dim_order=dim_order, label=label)
#     if f_test_x != None:
#         X_test, y_test = prepare_data(file_dir, f_test_x, f_test_y, shape=shape, dim_order=dim_order, label=label)
#     else:
#         X_test, y_test = None, None
#     # X_train, y_train = prepare_data(file_dir, f1+f2+f4+f3, f1_+f2_+f4_+f3_, shape=shape, dim_order=dim_order, label=label)
#     # X_test, y_test = prepare_data(file_dir, f5, f5_, shape=shape, dim_order=dim_order, label=label)
#     # X_train, y_train = prepare_data(file_dir, f1, f1_, shape=shape, dim_order=dim_order)
#     # X_test, y_test = prepare_data(file_dir, f2, f2_, shape=shape, dim_order=dim_order)
#     # exit()
#     return X_train, y_train, X_test, y_test, nw, nh, 1

def get_data(label, data):
    """ Flair, T1, T1C, T2 --> Y label"""
    ## Non-Norm HG
    # f_f = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_FLAIR.nii.gz', printable=False))
    # f_t1 = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T1.nii.gz', printable=False))
    # f_t1c = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T1c.nii.gz', printable=False))
    # f_t2 = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T2.nii.gz', printable=False))
    ## 3D Norm HG
    f_f = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_FLAIR_Norm.nii.gz', printable=False))
    f_t1 = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T1_Norm.nii.gz', printable=False))
    f_t1c = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T1c_Norm.nii.gz', printable=False))
    f_t2 = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_T2_Norm.nii.gz', printable=False))
    ## y
    f_label = sorted(tl.files.load_file_list(path=file_dir, regx='[0-9]_HG_GT.nii.gz', printable=False))

    dim_order = (2,1,0)
    shape = None

    # print(f_f[44:220])
    # exit()

    # img = read_Nifti1Image(file_dir, f_f[0])
    # target = read_Nifti1Image(file_dir, f_label[0])
    # X = img.get_data()
    # X = np.transpose(X, dim_order)
    # X = X[:,:,:,np.newaxis]
    # Y = target.get_data()
    # Y = np.transpose(Y, dim_order)
    # Y = Y[:,:,:,np.newaxis]
    #
    # shape = X.shape
    # print('X.shape',shape)
    # nw = shape[1]
    # nh = shape[2]
    # print('Y max',np.max(Y))  # 1,2,3,4
    # print('X max',np.max(X))
    # print(X.dtype)

    if data == 'f1':
        X_train, y_train = prepare_data_2(file_dir,
            f_f[44:220],
            f_t1[44:220],
            f_t1c[44:220],
            f_t2[44:220],
            f_label[44:220], shape=shape, dim_order=dim_order, label=label)
        X_test, y_test = prepare_data_2(file_dir,
            f_f[0:44], f_t1[0:44], f_t1c[0:44], f_t2[0:44], f_label[0:44], shape=shape, dim_order=dim_order, label=label)
    elif data == 'f2':
        X_train, y_train = prepare_data_2(file_dir,
            f_f[0:44]+f_f[88:220],
            f_t1[0:44]+f_t1[88:220],
            f_t1c[0:44]+f_t1c[88:220],
            f_t2[0:44]+f_t2[88:220],
            f_label[0:44]+f_label[88:220],
            shape=shape, dim_order=dim_order, label=label)
        X_test, y_test = prepare_data_2(file_dir,
            f_f[44:88], f_t1[44:88], f_t1c[44:88], f_t2[44:88], f_label[44:88], shape=shape, dim_order=dim_order, label=label)
    elif data == 'f3':
        X_train, y_train = prepare_data_2(file_dir,
            f_f[0:88]+f_f[132:220],
            f_t1[0:88]+f_t1[132:220],
            f_t1c[0:88]+f_t1c[132:220],
            f_t2[0:88]+f_t2[132:220],
            f_label[0:88]+f_label[132:220],
            shape=shape, dim_order=dim_order, label=label)
        X_test, y_test = prepare_data_2(file_dir,
            f_f[88:132], f_t1[88:132], f_t1c[88:132], f_t2[88:132], f_label[88:132], shape=shape, dim_order=dim_order, label=label)
    elif data == 'f4':
        X_train, y_train = prepare_data_2(file_dir,
            f_f[0:132]+f_f[176:220],
            f_t1[0:132]+f_t1[176:220],
            f_t1c[0:132]+f_t1c[176:220],
            f_t2[0:132]+f_t2[176:220],
            f_label[0:132]+f_label[176:220],
            shape=shape, dim_order=dim_order, label=label)
        X_test, y_test = prepare_data_2(file_dir,
            f_f[132:176], f_t1[132:176], f_t1c[132:176], f_t2[132:176], f_label[132:176],
            shape=shape, dim_order=dim_order, label=label)
    elif data == 'f5':
        X_train, y_train = prepare_data_2(file_dir,
            f_f[0:176],
            f_t1[0:176],
            f_t1c[0:176],
            f_t2[0:176],
            f_label[0:176],
            shape=shape, dim_order=dim_order, label=label)
        X_test, y_test = prepare_data_2(file_dir,
            f_f[176:220], f_t1[176:220], f_t1c[176:220], f_t2[176:220], f_label[176:220], shape=shape, dim_order=dim_order, label=label)
    elif data == 'all':
        X_train, y_train = prepare_data_2(file_dir,
            f_f, f_t1, f_t1c, f_t2, f_label, shape=shape, dim_order=dim_order, label=label)
        X_test, y_test = None, None
    elif data == 'debug':
        X_train, y_train = prepare_data_2(file_dir,
            f_f[0:3], f_t1[0:3], f_t1c[0:3], f_t2[0:3], f_label[0:3], shape=shape, dim_order=dim_order, label=label)
        X_test, y_test = prepare_data_2(file_dir,
            f_f[4:5], f_t1[4:5], f_t1c[4:5], f_t2[4:5], f_label[4:5], shape=shape, dim_order=dim_order, label=label)

    # f1 = (f_f_list[0:44], file_T1_list[0:44], file_T1c_list[0:44], file_T2_list[0:44])
    # f2 = (f_f_list[44:88], file_T1_list[44:88], file_T1c_list[44:88], file_T2_list[44:88])
    # f3 = (f_f_list[88:132], file_T1_list[88:132], file_T1c_list[88:132], file_T2_list[88:132])
    # f4 = (f_f_list[132:176], file_T1_list[132:176], file_T1c_list[132:176], file_T2_list[132:176])
    # f5 = (f_f_list[176:220], file_T1_list[176:220], file_T1c_list[176:220], file_T2_list[176:220])

    print(X_train.shape, X_train.min(), X_train.max(), y_train.shape, y_train.min(), y_train.max())
    nw = X_train.shape[1]
    nh = X_train.shape[2]
    nz = X_train.shape[3]
    # exit()
    return X_train, y_train, X_test, y_test, nw, nh, nz

#
if __name__ == "__main__":
    X_train, y_train, X_test, y_test, nw, nh, nz = get_data(label='whole', data='debug')
