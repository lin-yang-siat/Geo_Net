from sklearn.model_selection import KFold
from torch import nn
from torch.cuda.amp import autocast
from batchgenerators.utilities.file_and_folder_operations import *
import os

from monai.transforms import (
    AsDiscreted,
    EnsureChannelFirstd,
    # AddChanneld
    Compose,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    KeepLargestConnectedComponentd,
    Spacingd,
    ToTensord,
    RandAffined,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    RandRotate90d,
    EnsureTyped,
    Invertd,
    KeepLargestConnectedComponentd,
    SaveImaged,
    Activationsd
)

import numpy as np
from collections import OrderedDict
import glob
import monai.transforms.croppad.dictionary#.RandCropByPosNegLabeld
import monai.transforms.croppad.array
def data_loader(args):
    root_pet_dir = args.root_pet
    root_ct_dir = args.root_ct
    root_prior_P_dir = args.root_prior_P
    root_prior_S_dir = args.root_prior_S
    dataset = args.dataset

    if dataset == 'pet':
        print('Start to load data from directory: {}'.format(root_pet_dir))
        if args.mode == 'train':
            train_samples = {}
            valid_samples = {}
            ## Input training data
            train_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTr', '*_0000.nii.gz')))
            train_label = sorted(glob.glob(os.path.join(root_pet_dir, 'labelsTr', '*.nii.gz')))
            train_samples['images'] = train_img
            train_samples['labels'] = train_label
            ## Input validation data
            valid_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesVal', '*_0000.nii.gz')))
            valid_label = sorted(glob.glob(os.path.join(root_pet_dir, 'labelsVal', '*.nii.gz')))
            valid_samples['images'] = valid_img
            valid_samples['labels'] = valid_label
            print('Finished loading all training samples from dataset: {}!'.format(dataset))
            print(root_pet_dir)
            return train_samples, valid_samples
        elif args.mode == 'test':
            test_samples = {}
            ## Input inference data
            test_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTs', '*_0000.nii.gz')))
            test_label = sorted(glob.glob(os.path.join(root_pet_dir, 'labelsTs', '*.nii.gz')))
            test_samples['images'] = test_img
            test_samples['labels'] = test_label
            test_samples['path'] = root_pet_dir
            print('Finished loading all inference samples from dataset: {}!'.format(dataset))
            return test_samples, test_samples#,test_samples

    elif dataset == 'pet_ct':
        print('Start to load data from directory: {}, {}'.format(root_pet_dir, root_ct_dir))
        if args.mode == 'train':
            train_samples = {}
            valid_samples = {}
            ## Input training data
            train_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTr', '*_0000.nii.gz')))
            train_ct_img = sorted(glob.glob(os.path.join(root_ct_dir, 'imagesTr', '*_0001.nii.gz')))
            train_label = sorted(glob.glob(os.path.join(root_ct_dir, 'labelsTr', '*.nii.gz')))
            train_samples['images_pet'] = train_pet_img
            train_samples['images_ct'] = train_ct_img
            train_samples['labels'] = train_label
            ## Input validation data
            valid_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesVal', '*_0000.nii.gz')))
            valid_ct_img = sorted(glob.glob(os.path.join(root_ct_dir, 'imagesVal', '*_0001.nii.gz')))
            valid_label = sorted(glob.glob(os.path.join(root_ct_dir, 'labelsVal', '*.nii.gz')))
            valid_samples['images_pet'] = valid_pet_img
            valid_samples['images_ct'] = valid_ct_img
            valid_samples['labels'] = valid_label
            print('Finished loading all training samples from dataset: {}!'.format(dataset))
            return train_samples, valid_samples
        elif args.mode == 'test':
            test_samples = {}
            ## Input inference data
            test_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTs', '*_0000.nii.gz')))
            test_ct_img = sorted(glob.glob(os.path.join(root_ct_dir, 'imagesTs', '*_0001.nii.gz')))
            test_samples['images_pet'] = test_pet_img
            test_samples['images_ct'] = test_ct_img
            print('Finished loading all inference samples from dataset: {}!'.format(dataset))
            return test_samples
    elif dataset == 'pet_prior':
        print('Start to load data from directory: {}, {}'.format(root_pet_dir, root_prior_P_dir, root_prior_S_dir))
        if args.mode == 'train':
            train_samples = {}
            valid_samples = {}
            ## Input training data
            train_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTr', '*_0000.nii.gz')))
            train_P_img = sorted(glob.glob(os.path.join(root_prior_P_dir, 'PriorTr', '*_0001.nii.gz')))
            train_S_1_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTr', '*_0002.nii.gz')))
            train_S_2_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTr', '*_0003.nii.gz')))
            train_S_3_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTr', '*_0004.nii.gz')))
            train_S_4_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTr', '*_0005.nii.gz')))
            train_S_5_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTr', '*_0006.nii.gz')))
            train_S_6_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTr', '*_0007.nii.gz')))
            train_S_7_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTr', '*_0008.nii.gz')))
            train_S_8_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTr', '*_0009.nii.gz')))

            train_label = sorted(glob.glob(os.path.join(root_pet_dir, 'labelsTr', '*.nii.gz')))

            train_samples['images_pet'] = train_pet_img
            train_samples['Prior_P'] = train_P_img
            train_samples['Prior_S_1'] = train_S_1_img
            train_samples['Prior_S_2'] = train_S_2_img
            train_samples['Prior_S_3'] = train_S_3_img
            train_samples['Prior_S_4'] = train_S_4_img
            train_samples['Prior_S_5'] = train_S_5_img
            train_samples['Prior_S_6'] = train_S_6_img
            train_samples['Prior_S_7'] = train_S_7_img
            train_samples['Prior_S_8'] = train_S_8_img

            train_samples['labels'] = train_label
            ## Input validation data
            valid_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesVal', '*_0000.nii.gz')))
            valid_P_img = sorted(glob.glob(os.path.join(root_prior_P_dir, 'PriorVal', '*_0001.nii.gz')))
            valid_S1_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesVal', '*_0002.nii.gz')))
            valid_S2_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesVal', '*_0003.nii.gz')))
            valid_S3_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesVal', '*_0004.nii.gz')))
            valid_S4_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesVal', '*_0005.nii.gz')))
            valid_S5_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesVal', '*_0006.nii.gz')))
            valid_S6_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesVal', '*_0007.nii.gz')))
            valid_S7_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesVal', '*_0008.nii.gz')))
            valid_S8_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesVal', '*_0009.nii.gz')))



            valid_label = sorted(glob.glob(os.path.join(root_pet_dir, 'labelsVal', '*.nii.gz')))
            valid_samples['images_pet'] = valid_pet_img
            valid_samples['Prior_P'] = valid_P_img
            valid_samples['Prior_S_1'] = valid_S1_img
            valid_samples['Prior_S_2'] = valid_S2_img
            valid_samples['Prior_S_3'] = valid_S3_img
            valid_samples['Prior_S_4'] = valid_S4_img
            valid_samples['Prior_S_5'] = valid_S5_img
            valid_samples['Prior_S_6'] = valid_S6_img
            valid_samples['Prior_S_7'] = valid_S7_img
            valid_samples['Prior_S_8'] = valid_S8_img

            valid_samples['labels'] = valid_label
            print('Finished loading all training samples from dataset: {}!'.format(dataset))
            return train_samples, valid_samples
        elif args.mode == 'test':
            test_samples = {}
            ## Input inference data
            # test_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTs', '*.nii.gz')))

            test_pet_img = sorted(glob.glob(os.path.join(root_pet_dir, 'imagesTs', '*_0000.nii.gz')))
            test_P_img = sorted(glob.glob(os.path.join(root_prior_P_dir, 'PriorTs', '*_0001.nii.gz')))
            test_S1_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTs', '*_0002.nii.gz')))
            test_S2_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTs', '*_0003.nii.gz')))
            test_S3_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTs', '*_0004.nii.gz')))
            test_S4_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTs', '*_0005.nii.gz')))
            test_S5_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTs', '*_0006.nii.gz')))
            test_S6_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTs', '*_0007.nii.gz')))
            test_S7_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTs', '*_0008.nii.gz')))
            test_S8_img = sorted(glob.glob(os.path.join(root_prior_S_dir, 'SequencesTs', '*_0009.nii.gz')))

            test_samples['images_pet'] = test_pet_img
            test_samples['Prior_P'] = test_P_img
            test_samples['Prior_S_1'] = test_S1_img
            test_samples['Prior_S_2'] = test_S2_img
            test_samples['Prior_S_3'] = test_S3_img
            test_samples['Prior_S_4'] = test_S4_img
            test_samples['Prior_S_5'] = test_S5_img
            test_samples['Prior_S_6'] = test_S6_img
            test_samples['Prior_S_7'] = test_S7_img
            test_samples['Prior_S_8'] = test_S8_img
            print('Finished loading all inference samples from dataset: {}!'.format(dataset))
            return test_samples

def data_transforms(args):
    dataset = args.dataset
    if args.mode == 'train':
        crop_samples = args.crop_sample
    else:
        crop_samples = None

    if dataset == 'pet':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(48, 48, 48),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(48, 48, 48),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'ct':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(48, 48, 48),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=1.0, spatial_size=(48, 48, 48),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                ToTensord(keys=["image", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Orientationd(keys=["image"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image"], source_key="image"),
                ToTensord(keys=["image"]),
            ]
        )

    elif dataset == 'pet_ct':
        train_transforms = Compose(
            [
                LoadImaged(keys=["image_pet", "image_ct", "label"]),
                EnsureChannelFirstd(keys=["image_pet", "image_ct", "label"]),
                Orientationd(keys=["image_pet", "image_ct", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image_pet"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                ScaleIntensityRanged(
                    keys=["image_ct"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image_pet", "image_ct", "label"], source_key="image_pet"),
                RandCropByPosNegLabeld(
                    keys=["image_pet", "image_ct", "label"],
                    label_key="label",
                    spatial_size=(48, 48, 48),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="image_pet",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["image_pet", "image_ct"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['image_pet', 'image_ct', 'label'],
                    mode=('bilinear', 'bilinear', 'nearest'),
                    prob=1.0, spatial_size=(48, 48, 48),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["image_pet", "image_ct", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["image_pet", "image_ct", "label"]),
                EnsureChannelFirstd(keys=["image_pet", "image_ct", "label"]),
                Orientationd(keys=["image_pet", "image_ct", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image_pet"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                ScaleIntensityRanged(
                    keys=["image_ct"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image_pet", "image_ct", "label"], source_key="image_pet"),
                ToTensord(keys=["image_pet", "image_ct", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["image_pet", "image_ct"]),
                EnsureChannelFirstd(keys=["image_pet", "image_ct"]),
                Orientationd(keys=["image_pet", "image_ct"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["image_pet"], a_min=0, a_max=137919,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                ScaleIntensityRanged(
                    keys=["image_ct"], a_min=0, a_max=255,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["image_pet", "image_ct"], source_key="image_pet"),
                ToTensord(keys=["image_pet", "image_ct"]),
            ]
        )
    elif dataset == 'pet_prior':
        train_transforms = Compose(
            [
                LoadImaged(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"]),
                EnsureChannelFirstd(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"]),
                Orientationd(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["images_pet"], a_min=0, a_max=50,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                ScaleIntensityRanged(
                    keys=["Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8"], a_min=0, a_max=1,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"], source_key="images_pet"),
                RandCropByPosNegLabeld(
                    keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"],
                    label_key="label",
                    spatial_size=(48, 48, 48),
                    pos=1,
                    neg=1,
                    num_samples=crop_samples,
                    image_key="images_pet",
                    image_threshold=0,
                ),
                RandShiftIntensityd(
                    keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8"],
                    offsets=0.10,
                    prob=0.50,
                ),
                RandAffined(
                    keys=['images_pet', "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", 'label'],
                    mode=('bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'bilinear', 'nearest'),
                    prob=1.0, spatial_size=(48, 48, 48),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)),
                ToTensord(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"]),
            ]
        )

        val_transforms = Compose(
            [
                LoadImaged(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"]),
                EnsureChannelFirstd(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"]),
                Orientationd(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["images_pet"], a_min=0, a_max=50,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                ScaleIntensityRanged(
                    keys=["Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                        , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8"], a_min=0, a_max=1,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"],
                                source_key="images_pet"),
                ToTensord(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"]),
            ]
        )

        test_transforms = Compose(
            [
                LoadImaged(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"]),
                EnsureChannelFirstd(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"]),
                Orientationd(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"], axcodes="RAS"),
                ScaleIntensityRanged(
                    keys=["images_pet"], a_min=0, a_max=50,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                ScaleIntensityRanged(
                    keys=["Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                        , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8"], a_min=0, a_max=1,
                    b_min=0.0, b_max=1.0, clip=True,
                ),
                CropForegroundd(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"],
                                source_key="images_pet"),
                ToTensord(keys=["images_pet", "Prior_P", "Prior_S_1", "Prior_S_2", "Prior_S_3"
                    , "Prior_S_4", "Prior_S_5", "Prior_S_6", "Prior_S_7", "Prior_S_8", "label"]),
            ]
        )
    if args.mode == 'train':
        print('Cropping {} sub-volumes for training!'.format(str(crop_samples)))
        print('Performed Data Augmentations for all samples!')
        return train_transforms, val_transforms

    elif args.mode == 'test':
        print('Performed transformations for all samples!')
        return test_transforms


def infer_post_transforms(args):
    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Activationsd(keys="pred", softmax=True),
        AsDiscreted(keys="pred", argmax=True),

        SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=os.path.join(args.output, args.network),
                   output_postfix="", output_ext=".nii.gz", resample=True),
    ])

    return post_transforms



