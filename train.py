#! /usr/bin/python
# -*- coding: utf8 -*-

"""
This is a simple "U-Net Training Example" which inputs brain f-MRI including
all Flair, T1, T1c and T2 images to predicts the different tumors
(whole tumor, core tumor or enhance tumor)

This example implemented on Brats 2017 dataset using :
* Sørensen–Dice coefficient for loss function
* Elastic transform, flip left and right, rotation, shift, shearing and zoom for data augumentation
* Sørensen–Dice coefficient for loss function
* Flair, T1, T1c and T2 images --> whole tumor, core tumor or enhance tumor

Usage
------
If you wnat to test on the 5th fold and predict the whole tumor :

>>> python train.py --label=whole --data=f5

Licence
--------
Apache 2.0
"""

import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import numpy as np
import os
from model import *
from utils import *

def main(label_type, data_type):
    ## Create folder to save trained models and result images
    save_dir = "checkpoint"
    tl.files.exists_or_mkdir(save_dir)
    tl.files.exists_or_mkdir("samples/images_{}_{}".format(label_type, data_type))

    ###======================== LOAD DATA ======================================###
    ## You can load your own data here.
    import prepare_data
    cwd = os.getcwd()
    file_dir = os.path.join(cwd, 'data/Brats17TrainingData/HGG')
        # X_train, y_train, X_test, y_test, nw, nh, nz = prepare_data.get_data(label_type, data_type) REMOVE
    X_train, y_train, X_test, y_test, nw, nh, nz = prepare_data.get_data_brats2017(label_type, data_type, file_dir)

    with tf.device('/cpu:0'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'): #<- remove this if you train on CPU or other GPU
            ###======================== DEFIINE MODEL ===============================###
            batch_size = 10
            ## image range from [0, 1]. nz is 4 as we input all Flair, T1, T1c and T2,
            t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')
            ## labels are either 0 or 1
            t_seg = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_segment')
            ## Train inference
            net = u_net(t_image, is_train=True, reuse=False, pad='SAME', n_out=1)
            ## Test inference
            net_test = u_net(t_image, is_train=False, reuse=True, pad='SAME', n_out=1)
            net.print_layers()

            ###======================== DEFINE LOSS =================================###
            ## Train losses
                # out_seg = tf.expand_dims(net.outputs[:,:,:,0], axis=3)  REMOVE
            out_seg = net.outputs
            dice_loss = 1 - tl.cost.dice_coe(out_seg, t_seg, epsilon=1e-10)
            iou_loss = 1 - tl.cost.iou_coe(out_seg, t_seg)
            dice_hard = tl.cost.dice_hard_coe(out_seg, t_seg)

            loss = dice_loss # training loss

            ## Test losses
                # test_out_seg = tf.expand_dims(net_test.outputs[:,:,:,0], axis=3)  REMOVE
            test_out_seg = net_test.outputs
            test_dice_loss = 1 - tl.cost.dice_coe(test_out_seg, t_seg, epsilon=1e-10)
            test_iou_loss = 1 - tl.cost.iou_coe(test_out_seg, t_seg)
            test_dice_hard = tl.cost.dice_hard_coe(test_out_seg, t_seg)

        ###======================== DEFINE TRAIN OPTS ===========================###
        lr = 0.00001
        lr_decay = 0.5
        decay_every = 10
        beta1 = 0.9
        n_epoch = 200
        print_freq_step = 100

        t_vars = tl.layers.get_variables_with_name('u_net', True, True)
        with tf.device('/gpu:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr, trainable=False)
            train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars)

        ###======================== TRAINING ====================================###
        tl.layers.initialize_global_variables(sess)

        ## Load previous model if exists
        tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}_{}.npz'.format(label_type, data_type), network=net)

        for epoch in range(0, n_epoch+1):
            epoch_time = time.time()
            ## Update decay learning rate at the beginning of every epoch
            if epoch !=0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(lr_v, lr * new_lr_decay))
                log = " ** new learning rate: %f" % (lr * new_lr_decay)
                print(log)
            elif epoch == 0:
                log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
                print(log)

            ## Train loop
            total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
            for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train, batch_size=batch_size, shuffle=True):
                images, labels = batch

                step_time = time.time()
                ## Data augumentation for a batch of Flair, T1, T1c, T2 images
                ## and label images synchronously. You can define your own
                ## method here.
                data = threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis],
                        images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis],
                        images[:,:,:,3, np.newaxis], labels)],
                        fn=distort_imgs, label=label_type)
                # print(data.shape)
                # data = data.transpose((1,0,2,3,4))    REMOVE
                # print(data.shape)
                b_images = data[:,0:4,:,:,:]  # (10, 4, 240, 240, 1)
                b_images = b_images.transpose((0,2,3,1,4))
                b_images.shape = (batch_size, nw, nh, nz)
                b_labels = data[:,4,:,:,:]

                ## Update network
                _, _dice, _iou, _diceh, out = sess.run([train_op,
                        dice_loss, iou_loss, dice_hard, net.outputs],
                        {t_image: b_images, t_seg: b_labels})
                total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
                n_batch += 1

                if n_batch % print_freq_step == 0:
                    print("Epoch %d step %d [0: (1-dice): %f (1-iou): %f hard-dice: %f] took %fs (with distortion)" %
                            (epoch, n_batch, _dice, _iou, _diceh, time.time()-step_time))

                # if np.isnan(_dice):
                #     exit(" ** NaN loss found during training, stop training" % str(err))  REMOVE
                # if np.isnan(out).any():
                #     exit(" ** NaN found in output images during training, stop training")

            print(" ** Epoch [%d/%d] train [0-(1-dice): %f (1-iou): %f hard-dice: %f] took %fs (with distortion)" %
                    (epoch, n_epoch, total_dice/n_batch, total_iou/n_batch, total_dice_hard/n_batch, time.time()-epoch_time))

            ## Save training images of Flair, T1, T1c, T2, label and prediction in order
            tl.visualize.save_images(np.asarray([b_images[0][:,:,0,np.newaxis],
                b_images[0][:,:,1,np.newaxis], b_images[0][:,:,2,np.newaxis],
                b_images[0][:,:,3,np.newaxis], b_labels[0], out[0,:,:,0,np.newaxis]]), size=(1, 6),
                image_path="samples/images_{}_{}_{}/train_{}.png".format(mode, label_type, data_type, epoch))

            ##======================== Evaluation ========================##
            total_dice, total_iou, total_dice_hard, n_batch = 0, 0, 0, 0
            for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test, batch_size=batch_size, shuffle=False):
                b_images, b_labels = batch
                # b_images = threading_data(b_images, fn=normalize_img) # [0, 255]->[0, 1] # if we already normalized the images, remove this line
                _dice, _iou, _diceh, out = sess.run([test_dice_loss,
                        test_iou_loss, test_dice_hard, net_test.outputs],
                        {t_image: b_images, t_seg: b_labels})
                total_dice += _dice; total_iou += _iou; total_dice_hard += _diceh
                n_batch += 1

            print(" **  test [0-(1-dice): %f (1-iou): %f hard-dice: %f] took %fs (no distortion)" %
                    (total_dice/n_batch, total_iou/n_batch, total_dice_hard/n_batch,
                    time.time()-epoch_time))
            print("    for {} tumor , {} fold".format(label_type, data_type))

            ## Save testing images of Flair, T1, T1c, T2, label and prediction in order
            tl.visualize.save_images(np.asarray([b_images[0][:,:,0,np.newaxis],
                b_images[0][:,:,1,np.newaxis], b_images[0][:,:,2,np.newaxis],
                b_images[0][:,:,3,np.newaxis], b_labels[0], out[0,:,:,0,np.newaxis]]), size=(1, 6),
                image_path="samples/images_{}_{}/test_{}.png".format(label_type, data_type, epoch))

            ##======================== SAVE MODEL ========================##
            tl.files.save_npz(net.all_params, name=save_dir+'/u_net_{}_{}.npz'.format(label_type, data_type), sess=sess)
            print("[*] Save checkpoints SUCCESS!")

if __name__ == "__main__":
    ###======================== SELECT DATASET ==============================###
    ## you can use your own dataset with this code
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--label', type=str, default='whole', help='whole(1,2,3,4), \
        core(1,3,4), enhance(4), label(1~4), label234 [What kind of tumor do you want to predict]')
                    # whole: 1 2 3 4;  core: 1 3 4;  enhance: 4
    parser.add_argument('--data', type=str, default='f5', help='f1, f2, f3, f4, f5, \
        all, debug [Which fold is the test set]')
                    # f1~5: which fold as testing set; all: mean all data are training set
    args = parser.parse_args()

    ###======================== START =======================================###
    main(args.label, args.data)
