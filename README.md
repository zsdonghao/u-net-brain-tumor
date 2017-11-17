# U-Net Brain Tumor Segmentation 

This repo show you how to train a U-Net for brain tumor segmentation. By default, you need to download the training set of [BRATS 2017](http://braintumorsegmentation.org) dataset, which have 210 HGG and 75 LGG volumes, and put the data folder along with all scripts.

```bash
data
  -- Brats17TrainingData
  -- train_dev_all
model.py
train.py
...
```

### About the data
Note that according to the license, user have to apply the dataset from BRAST, please do **NOT** contact me for the dataset. Many thanks.

<div align="center">
    <img src="https://github.com/zsdonghao/u-net-brain-tumor/blob/master/example/brain_tumor_data.png" width="80%" height="50%"/>
    <br>  
    <em align="center">Fig 1: Brain Image</em>  
</div>

* Each volume have 4 scanning images: FLAIR、T1、T1c and T2.
* Each volume have 4 segmentation labels:

```
Label 0: background
Label 1: necrotic and non-enhancing tumor
Label 2: edema 
Label 4: enhancing tumor
```

The `prepare_data_with_valid.py` split the training set into 2 folds for training and validating. By default, it will use only half of the data for the sake of training speed, if you want to use all data, just change `DATA_SIZE = 'half'` to `all`.

### About the method

- Network and Loss: In this experiment, as we use [dice loss](http://tensorlayer.readthedocs.io/en/latest/modules/cost.html#dice-coefficient) to train a network, one network only predict one labels (Label 1,2 or 4). We evaluate the performance using [hard dice](http://tensorlayer.readthedocs.io/en/latest/modules/cost.html#hard-dice-coefficient) and [IOU](http://tensorlayer.readthedocs.io/en/latest/modules/cost.html#iou-coefficient).

- Data augmenation: Includes random left and right flip, rotation, shifting, shearing, zooming and the most important one -- [Elastic trasnformation](http://tensorlayer.readthedocs.io/en/latest/modules/prepro.html#elastic-transform) which is borrowed from ["Automatic Brain Tumor Detection and Segmentation Using U-Net Based Fully Convolutional Networks"](https://arxiv.org/pdf/1705.03820.pdf).

<div align="center">
    <img src="https://github.com/zsdonghao/u-net-brain-tumor/blob/master/example/brain_tumor_aug.png" width="80%" height="50%"/>
    <br>  
    <em align="center">Fig 2: Data augmentation</em>  
</div>

### Start training

We train HGG and LGG together, as one network only have one task, set the `task` to `all`, `necrotic`, `edema` or `enhance`, "all" means learn to segment all tumors.

```
python train.py --task=all
```

Note that, if the loss stick on 1 at the beginning, it means the network doesn't converge to near-perfect accuracy, please try restart it.
