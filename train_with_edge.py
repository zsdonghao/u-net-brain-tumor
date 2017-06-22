#! /usr/bin/python
# -*- coding: utf8 -*-


import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import numpy as np
import os
from model import *
from utils import *

def main_train(mode, label_type, data_type):
    tl.files.exists_or_mkdir("checkpoint")
    tl.files.exists_or_mkdir("samples_edge/train_{}_{}_{}".format(mode, label_type, data_type))
    save_dir = "checkpoint"

    import prepare_data
    if mode == 'flair_only':
        X_train, y_train, X_test, y_test, nw, nh, nz = prepare_data.get_data_Flair(label_type, data_type)
    elif mode == 'all':
        X_train, y_train, X_test, y_test, nw, nh, nz = prepare_data.get_data(label_type, data_type)
    else:
        raise Exception("Unsupport mode: {}".format(mode))
    # print(X_train.shape, X_test.shape, nw, nh, nz)
    # exit()

    ## try different data argumentation parameter
    # x = elastic_transform(X_train[80], alpha=255 * 3, sigma=255 * 0.10, is_random=True)
    # tl.visualize.save_images(np.asarray([X_train[80], x]), size=(1, 2), image_path="samples_edge/train_{}_{}/_.png".format(label_type, data_type))
    # x = elastic_transform(X_train[80], alpha=255 * 3, sigma=255 * 0.15, is_random=True)
    # tl.visualize.save_images(np.asarray([X_train[80], x]), size=(1, 2), image_path="samples_edge/train_{}_{}/_2.png".format(label_type, data_type))
    # x = elastic_transform(X_train[80], alpha=255 * 3, sigma=255 * 0.20, is_random=True)
    # tl.visualize.save_images(np.asarray([X_train[80], x]), size=(1, 2), image_path="samples_edge/train_{}_{}/_3.png".format(label_type, data_type))
    # exit()

    with tf.device('/cpu:0'):
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        with tf.device('/gpu:0'):
        ###======================== DEFIINE MODEL ===============================###
            batch_size = 10

            t_image = tf.placeholder('float32', [batch_size, nw, nh, nz], name='input_image')  # [0, 1]
            t_seg = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_segment') # [0, 1]
            t_edge = tf.placeholder('float32', [batch_size, nw, nh, 1], name='target_edge')   # [0, 1]

            net = u_net(t_image, is_train=True, reuse=False, pad='SAME')
            net_test = u_net(t_image, is_train=False, reuse=True, pad='SAME')

            #net = u_net_bn(t_image, is_train=True, reuse=False, batch_size=batch_size, pad='SAME')
            #net_test = u_net_bn(t_image, is_train=False, reuse=True, batch_size=batch_size,  pad='SAME')

            net.print_layers()

        ###======================== DEFINE loss =================================###
            # train
            out_seg = tf.expand_dims(net.outputs[:,:,:,0], axis=3)
            dice_loss0 = 1 - tl.cost.dice_coe(out_seg, t_seg, epsilon=1e-10)
            iou_loss0 = 1 - tl.cost.iou_coe(out_seg, t_seg)
            dice_hard0 = tl.cost.dice_hard_coe(out_seg, t_seg)

            out_edge = tf.expand_dims(net.outputs[:,:,:,1], axis=3)
            dice_loss1 = 1 - tl.cost.dice_coe(out_edge, t_edge, epsilon=1e-10)
            iou_loss1 = 1 - tl.cost.iou_coe(out_edge, t_edge)
            dice_hard1 = tl.cost.dice_hard_coe(out_edge, t_edge)

            loss = dice_loss0 #+ dice_loss1 * 0.2
            # loss = (dice_loss0 ** 2 + dice_loss1 ** 2) / (dice_loss0 + dice_loss1 + 0.01)

            # test
            test_out_seg = tf.expand_dims(net_test.outputs[:,:,:,0], axis=3)
            test_dice_loss0 = 1 - tl.cost.dice_coe(test_out_seg, t_seg, epsilon=1e-10)
            test_iou_loss0 = 1 - tl.cost.iou_coe(test_out_seg, t_seg)
            test_dice_hard0 = tl.cost.dice_hard_coe(test_out_seg, t_seg)

            test_out_edge = tf.expand_dims(net_test.outputs[:,:,:,1], axis=3)
            test_dice_loss1 = 1 - tl.cost.dice_coe(test_out_edge, t_edge, epsilon=1e-10)
            test_iou_loss1 = 1 - tl.cost.iou_coe(test_out_edge, t_edge)
            test_dice_hard1 = tl.cost.dice_hard_coe(test_out_edge, t_edge)

        ###======================== DEFINE TRAIN OPTS ===========================###
        lr = 0.00001 #/ 2
        lr_decay = 0.5
        decay_every = 10#50
        beta1 = 0.9
        n_epoch = 200
        print_freq_step = 100

        t_vars = tl.layers.get_variables_with_name('u_net', True, True)
        with tf.device('/gpu:0'):
            with tf.variable_scope('learning_rate'):
                lr_v = tf.Variable(lr, trainable=False)
            train_op = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss, var_list=t_vars )

        ###======================== TRAINING ====================================###
        tl.layers.initialize_global_variables(sess)

        tl.files.load_and_assign_npz(sess=sess, name=save_dir+'/u_net_{}_{}_{}.npz'.format(mode, label_type, data_type), network=net)

        for epoch in range(0, n_epoch+1):
            epoch_time = time.time()
            if epoch !=0 and (epoch % decay_every == 0):
                new_lr_decay = lr_decay ** (epoch // decay_every)
                sess.run(tf.assign(lr_v, lr * new_lr_decay))
                log = " ** new learning rate: %f" % (lr * new_lr_decay)
                print(log)
            elif epoch == 0:
                log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
                print(log)

            n_batch = 0
            train_dice0, train_iou0, train_dice_hard0 = 0, 0, 0
            train_dice1, train_iou1, train_dice_hard1 = 0, 0, 0
            for batch in tl.iterate.minibatches(inputs=X_train, targets=y_train, batch_size=batch_size, shuffle=True):
                images, labels = batch

                step_time = time.time()
                if mode == 'flair_only':
                    data = threading_data([_ for _ in zip(images, labels)], fn=distort_img)
                    b_images, b_labels = data.transpose((1,0,2,3,4))
                    # print(b_images.shape, b_images[0].shape, b_labels.shape)# (10, 240, 240, 1) (240, 240, 1) (10, 240, 240, 1)
                    # exit()
                elif mode == 'all': # b_images.shape[-1] == 4
                    # print(images.shape, images[:,:,:,0, np.newaxis].shape) # (10, 240, 240, 4) (240, 240, 4)
                    # x1, x2, x3, x4, y = distort_imgs((images[0,:,:,0, np.newaxis], images[0,:,:,1, np.newaxis], images[0,:,:,2, np.newaxis], images[0,:,:,3, np.newaxis], labels[0]))
                    # print(x1.shape, x2.shape, x3.shape, x4.shape, y.shape) #(240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1) (240, 240, 1)

                    if label_type == 'label1':
                        # T1C, T2
                        data = threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis], images[:,:,:,1, np.newaxis], labels)], fn=distort_imgs, label=label_type)
                    else:
                        # Flair, T1, T1C, T2
                        data = threading_data([_ for _ in zip(images[:,:,:,0, np.newaxis], images[:,:,:,1, np.newaxis], images[:,:,:,2, np.newaxis], images[:,:,:,3, np.newaxis], labels)], fn=distort_imgs, label=label_type)
                    # print(data.shape)
                    # data = data.transpose((1,0,2,3,4))
                    # print(data.shape)

                    if label_type == 'label1':
                        b_images = data[:,0:2,:,:,:]
                    else:
                        b_images = data[:,0:4,:,:,:]  # (10, 4, 240, 240, 1)

                    b_images = b_images.transpose((0,2,3,1,4))
                    b_images.shape = (batch_size, nw, nh, nz)

                    if label_type == 'label1':
                        b_labels = data[:,-1,:,:,:]
                    else:
                        b_labels = data[:,4,:,:,:]

                b_edges = threading_data(b_labels, fn=get_one_contour, nw=nw, nh=nh)

                # print(b_images.shape, b_images[0][:,:,0].shape, b_labels.shape)
                # b_edges = threading_data(b_labels, fn=get_one_contour, nw=nw, nh=nh)
                # # tl.visualize.save_images(np.asarray([b_images[0][:,:,0, np.newaxis], b_labels[0], b_edges[0]]), size=(1, 3),
                # #     image_path="samples_edge/train_{}_{}_{}/_debug_{}.png".format(mode, label_type, data_type, epoch))
                # tl.visualize.save_images(np.asarray([b_images[0][:,:,0,np.newaxis], b_images[0][:,:,1,np.newaxis], b_images[0][:,:,2,np.newaxis], b_images[0][:,:,3,np.newaxis],
                #     b_labels[0], b_edges[0]]), size=(1, 6),
                #     image_path="samples_edge/train_{}_{}_{}/_debug_{}.png".format(mode, label_type, data_type, epoch))
                # tl.visualize.save_images(np.asarray([b_images[0][:,:,0,np.newaxis], b_images[0][:,:,1,np.newaxis],
                #     b_labels[0], b_edges[0]]), size=(1, 4),
                #     image_path="samples_edge/train_{}_{}_{}/_debug_{}.png".format(mode, label_type, data_type, epoch))
                # print(b_labels[0].min(), b_labels[0].max())
                # exit()

                _, dice0, iou0, diceh0, dice1, iou1, diceh1, out = sess.run([train_op, dice_loss0, iou_loss0, dice_hard0, dice_loss1, iou_loss1, dice_hard1, net.outputs],
                                        {t_image: b_images, t_seg: b_labels, t_edge: b_edges})
                train_dice0 += dice0; train_iou0 += iou0; train_dice_hard0 += diceh0
                train_dice1 += dice1; train_iou1 += iou1; train_dice_hard1 += diceh1
                n_batch += 1

                if n_batch % print_freq_step == 0:
                    print("Epoch %d step %d [0: (1-dice): %f (1-iou): %f hard-dice: %f] [1:(1-dice): %f (1-iou): %f hard-dice: %f] took %fs" %
                            (epoch, n_batch, dice0, iou0, diceh0, dice1, iou1, diceh1, time.time()-step_time))
                # tl.visualize.save_images(np.asarray([b_images[0], b_labels[0], b_edges[0]]), size=(1, 3), image_path="samples_edge/train/train_%d.png" % epoch)
                # print(out[0,:,:,0, np.newaxis].shape, b_images[0].shape, b_images[0].min(), b_images[0].max(), b_labels[0].shape, b_labels.max())
                # tl.visualize.save_images(np.asarray([b_images[0], b_labels[0], b_edges[0], out[0,:,:,0, np.newaxis], out[0,:,:,1, np.newaxis]]), size=(1, 5), image_path="samples_edge/train/_.png")
                # exit()
                # exit()
                # if np.isnan(dice0):
                #     exit(" ** NaN loss found during training, stop training" % str(err))
                # if np.isnan(out).any():
                #     exit(" ** NaN found in output images during training, stop training")

            print(" ** Epoch [%d/%d] train [0-(1-dice): %f (1-iou): %f hard-dice: %f] [1-(1-dice): %f (1-iou): %f hard-dice: %f] took %fs" %
                    (epoch, n_epoch,
                    train_dice0/n_batch, train_iou0/n_batch, train_dice_hard0/n_batch,
                    train_dice1/n_batch, train_iou1/n_batch, train_dice_hard1/n_batch,
                    time.time()-epoch_time))
            # tl.visualize.save_images(np.asarray([b_images[0], b_labels[0], b_edges[0], out[0,:,:,0], out[0,:,:,1]]), size=(1, 3), image_path="samples_edge/train/train_{}_{}/train_{}.png".format(label_type, data_type, epoch))
            if mode == 'flair_only':
                tl.visualize.save_images(np.asarray([b_images[0], b_labels[0], b_edges[0],
                    out[0,:,:,0, np.newaxis], out[0,:,:,1, np.newaxis]]), size=(1, 5),
                    image_path="samples_edge/train_{}_{}_{}/train_{}.png".format(mode, label_type, data_type, epoch))
            elif mode == 'all':
                if label_type == 'label1':
                    tl.visualize.save_images(np.asarray([b_images[0][:,:,0,np.newaxis], b_images[0][:,:,1,np.newaxis],
                        b_labels[0], b_edges[0], out[0,:,:,0, np.newaxis], out[0,:,:,1, np.newaxis]]), size=(1, 6),
                        image_path="samples_edge/train_{}_{}_{}/train_{}.png".format(mode, label_type, data_type, epoch))
                else:
                    tl.visualize.save_images(np.asarray([b_images[0][:,:,0,np.newaxis], b_images[0][:,:,1,np.newaxis], b_images[0][:,:,2,np.newaxis], b_images[0][:,:,3,np.newaxis],
                        b_labels[0], b_edges[0], out[0,:,:,0, np.newaxis], out[0,:,:,1, np.newaxis]]), size=(1, 8),
                        image_path="samples_edge/train_{}_{}_{}/train_{}.png".format(mode, label_type, data_type, epoch))

            if X_test is not None:
                n_batch = 0
                test_dice0, test_iou0, test_dice_hard0 = 0, 0, 0
                test_dice1, test_iou1, test_dice_hard1 = 0, 0, 0
                for batch in tl.iterate.minibatches(inputs=X_test, targets=y_test, batch_size=batch_size, shuffle=True):
                    b_images, b_labels = batch
                    # b_images = threading_data(b_images, fn=normalize_img) # [0, 255]->[0, 1]
                    b_edges = threading_data(b_labels, fn=get_one_contour, nw=nw, nh=nh)
                    dice0, iou0, diceh0, dice1, iou1, diceh1, out = sess.run([dice_loss0, iou_loss0, dice_hard0, dice_loss1, iou_loss1, dice_hard1, net_test.outputs],
                                            {t_image: b_images, t_seg: b_labels, t_edge: b_edges})
                    test_dice0 += dice0; test_iou0 += iou0; test_dice_hard0 += diceh0
                    test_dice1 += dice1; test_iou1 += iou1; test_dice_hard1 += diceh1
                    n_batch += 1

                print(" **                test [0-(1-dice): %f (1-iou): %f hard-dice: %f] [1-(1-dice): %f (1-iou): %f hard-dice: %f] took %fs" %
                        (test_dice0/n_batch, test_iou0/n_batch, test_dice_hard0/n_batch,
                        test_dice1/n_batch, test_iou1/n_batch, test_dice_hard1/n_batch,
                        time.time()-epoch_time))
                print("                   {}  {}  {}".format(mode, label_type, data_type))
                # tl.visualize.save_images(np.asarray([b_images[0], b_labels[0], b_edges[0],
                #         out[0,:,:,0, np.newaxis], out[0,:,:,1, np.newaxis]]), size=(1, 5),
                #         image_path="samples_edge/train_{}_{}_{}/test_{}.png".format(mode, label_type, data_type, epoch))
                if mode == 'flair_only':
                    tl.visualize.save_images(np.asarray([b_images[0], b_labels[0], b_edges[0],
                        out[0,:,:,0, np.newaxis], out[0,:,:,1, np.newaxis]]), size=(1, 5),
                        image_path="samples_edge/train_{}_{}_{}/test_{}.png".format(mode, label_type, data_type, epoch))
                elif mode == 'all':
                    if label_type == 'label1':
                        tl.visualize.save_images(np.asarray([b_images[0][:,:,0,np.newaxis], b_images[0][:,:,1,np.newaxis],
                            b_labels[0], b_edges[0], out[0,:,:,0, np.newaxis], out[0,:,:,1, np.newaxis]]), size=(1, 6),
                            image_path="samples_edge/train_{}_{}_{}/test_{}.png".format(mode, label_type, data_type, epoch))
                    else:
                        tl.visualize.save_images(np.asarray([b_images[0][:,:,0,np.newaxis], b_images[0][:,:,1,np.newaxis], b_images[0][:,:,2,np.newaxis], b_images[0][:,:,3,np.newaxis],
                            b_labels[0], b_edges[0], out[0,:,:,0, np.newaxis], out[0,:,:,1, np.newaxis]]), size=(1, 8),
                            image_path="samples_edge/train_{}_{}_{}/test_{}.png".format(mode, label_type, data_type, epoch))

            # if (epoch != 0) and (epoch % 5) == 0:
            tl.files.save_npz(net.all_params, name=save_dir+'/u_net_{}_{}_{}.npz'.format(mode, label_type, data_type), sess=sess)
            print("[*] Save checkpoints SUCCESS!")






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='all', help='flair_only, all (flair+T1+T1c+T2)')
    parser.add_argument('--label', type=str, default='whole', help='whole(1,2,3,4), core(1,3,4), enhance(4), label(1~4), label234')
                    # whole: 1 2 3 4;  core: 1 3 4;  enhance: 4
    parser.add_argument('--data', type=str, default='f5', help='f1, f2, f3, f4, f5, all, debug')
                    # f1~5: which fold as testing set; all: mean all data are training set

    args = parser.parse_args()

    main_train(args.mode, args.label, args.data) # use flair input to predict all situation
