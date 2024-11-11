from monai.utils import first, set_determinism


from monai.networks.nets import UNETR, SwinUNETR
from networks.Geo_net.Geo_net import Geo_Net
from medzoo.Unet3D import UNet3D
from medzoo.SkipDenseNet3D import SkipDenseNet3D
from medzoo.Vnet import VNet
from medzoo.ResNet3D_VAE import ResNet3dVAE
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, decollate_batch
from monai.transforms import AsDiscrete
from monai.metrics import DiceMetric

import torch
from load_datasets_transforms import data_loader, data_transforms, infer_post_transforms
import multiprocessing
import numpy as np
import pandas as pd

import os
import argparse

from metrics_class import calculate_metrics, calculate_distance

parser = argparse.ArgumentParser(description='An Automatic 3D PET Tumor Segmentation Framework Assisted by Geodesic Sequences')
## Input data hyperparameters

parser.add_argument('--root_pet', type=str, default=r'D:\Data\hecktor\Train', required=False, help='Root folder of all your images and labels')
parser.add_argument('--root_ct', type=str, default=r'D:\Data\hecktor\Train', required=False, help='Root folder of all your images and labels')
parser.add_argument('--root_prior_P', type=str, default=r'D:\Data\hecktor\Train', required=False, help='Root folder of all your images and labels')
parser.add_argument('--root_prior_S', type=str, default=r'D:\Data\hecktor\Train', required=False, help='Root folder of all your images and labels')
parser.add_argument('--output', type=str, default='./output/hecktor/Geo_Net', required=False, help='Output folder for both tensorboard and the best model')
parser.add_argument('--dataset', type=str, default='pet_prior', required=False, help='Datasets: {pet, ct, pet_ct, pet_prior}, Fyi: You can add your dataset here')

## Input model & training hyperparameters
parser.add_argument('--network', type=str, default='Geo_Net', required=False, help='Network models: {3DUXNET, nnFormer, UNETR, SwinUNETR, NestedUNet, UNet3D, ResUNet, VNet}')
parser.add_argument('--trained_weights', default='./output/Geo_Net/best_metric_model.pth', help='Path of pretrained/fine-tuned weights')
parser.add_argument('--mode', type=str, default='train', help='Training or testing mode')
parser.add_argument('--sw_batch_size', type=int, default=1, help='Sliding window batch size for inference')
parser.add_argument('--overlap', type=float, default=0.5, help='Sub-volume overlapped percentage')
parser.add_argument('--out_classes', type=int, default=2, help='out lasses of brain dataset segmentation')
parser.add_argument('--crop_sample', type=int, default='2', help='Number of cropped sub-volumes for each subject')

## Efficiency hyperparameters
parser.add_argument('--gpu', type=str, default='0', help='your GPU number')
parser.add_argument('--cache_rate', type=float, default=0.1, help='Cache rate to cache your dataset into GPUs')
parser.add_argument('--num_workers', type=int, default=2, help='Number of workers')

args = parser.parse_args()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    _, valid_samples = data_loader(args)

    if args.dataset == 'pet_ct':
        val_files = [
            {"image_pet": image_pet_name, "image_ct": image_ct_name, "label": label_name}
            for image_pet_name, image_ct_name, label_name in zip(valid_samples['images_pet'], valid_samples['images_ct'], valid_samples['labels'])
        ]
    elif args.dataset == 'pet_prior':
        val_files = [
            {"images_pet": image_pet_name, "Prior_P": Prior_P_name, 'Prior_S_1':Prior_S_1_name, 'Prior_S_2':Prior_S_2_name, 'Prior_S_3':Prior_S_3_name, 'Prior_S_4':Prior_S_4_name, 'Prior_S_5':Prior_S_5_name
                , 'Prior_S_6':Prior_S_6_name, 'Prior_S_7':Prior_S_7_name, 'Prior_S_8':Prior_S_8_name,"label": label_name}
            for image_pet_name, Prior_P_name, Prior_S_1_name, Prior_S_2_name, Prior_S_3_name, Prior_S_4_name, Prior_S_5_name, Prior_S_6_name, Prior_S_7_name, Prior_S_8_name, label_name in
            zip(valid_samples['images_pet'], valid_samples['Prior_P'], valid_samples['Prior_S_1'], valid_samples['Prior_S_2'], valid_samples['Prior_S_3'], valid_samples['Prior_S_4']
                , valid_samples['Prior_S_5'], valid_samples['Prior_S_6'], valid_samples['Prior_S_7'], valid_samples['Prior_S_8'], valid_samples['labels'])
        ]
    else:
        val_files = [
            {"image": image_name, "label": label_name}
            for image_name, label_name in zip(valid_samples['images'], valid_samples['labels'])
        ]

    set_determinism(seed=0)

    _, val_transforms = data_transforms(args)
    post_transforms = infer_post_transforms(args)

    ## Inference Pytorch Data Loader and Caching
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms, cache_rate=args.cache_rate, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=args.num_workers)

    ## Load Networks
    device = torch.device("cuda:0")
    if args.network == 'Geo_Net':
        model = Geo_Net(
            input_channels=1,
            output_channels=args.out_classes,
            feature_channels=12,
            seq_len=8
        ).to(device)
    elif args.network == 'SwinUNETR':
        model = SwinUNETR(
            img_size=(144, 144, 144),
            in_channels=1,
            out_channels=args.out_classes,
            feature_size=24,
            use_checkpoint=False,
        ).to(device)
    elif args.network == 'UNet3D':
        model = UNet3D(in_channels=1, n_classes=args.out_classes).to(device)

    print('Chosen Network Architecture: {}'.format(args.network))

    model.load_state_dict(torch.load(args.trained_weights))
    model.eval()

    post_label = AsDiscrete(to_onehot=args.out_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=args.out_classes)
    num_samples = len(val_loader)
    num_classes = args.out_classes
    dice_vals =  np.zeros((num_samples, num_classes))
    ravd_vals = np.zeros((num_samples, num_classes))
    with torch.no_grad():
        for i, val_data in enumerate(val_loader):
            if args.dataset == 'pet_prior':
                x, val_labels = (val_data["images_pet"].cuda(), val_data["label"].cuda())
                x_P, x_S_1, x_S_2, x_S_3, x_S_4, x_S_5, x_S_6, x_S_7, x_S_8 = (
                    val_data["Prior_P"].cuda(), val_data["Prior_S_1"].cuda(), val_data["Prior_S_2"].cuda(),
                    val_data["Prior_S_3"].cuda(), val_data["Prior_S_4"].cuda(), val_data["Prior_S_5"].cuda(),
                    val_data["Prior_S_6"].cuda(), val_data["Prior_S_7"].cuda(), val_data["Prior_S_8"].cuda())
                val_pet_inputs = torch.cat((x, x_P, x_S_1, x_S_2, x_S_3, x_S_4, x_S_5, x_S_6, x_S_7, x_S_8), dim=1)
            elif args.dataset == 'pet':
                val_pet_inputs, val_labels = (val_data["image"].cuda(), val_data["label"].cuda())
            roi_size = (144, 144, 144)
            val_outputs = sliding_window_inference(val_pet_inputs, roi_size, args.sw_batch_size, model, overlap=args.overlap)
            #-------存储
            val_data['pred'] = val_outputs
            val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            #------计算dice
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [
                post_label(val_label_tensor) for val_label_tensor in val_labels_list
            ]
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [
                post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list
            ]
            for j in range(args.out_classes):
                val_output = val_output_convert[0][j]
                val_label = val_labels_convert[0][j]
                #dice
                dice_value = calculate_metrics(val_label, val_output, 'dice')
                dice_vals[i,j] = dice_value
                #ravd
                ravd_value = calculate_metrics(val_label, val_output, 'ravd')
                ravd_vals[i, j] = ravd_value


    mean_dice_val = np.mean(dice_vals)
    mean_ravd_val = np.mean(ravd_vals)
    print("mean_dice_val:",mean_dice_val)
    print("mean_ravd_val:",mean_ravd_val)


              
if __name__ == '__main__':
    multiprocessing.freeze_support()
    torch.cuda.empty_cache()
    main()
