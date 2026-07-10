# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle
import logging

import numpy as np
import torch
from torch.utils.data.sampler import Sampler
import SimpleITK as sitk
from scipy.ndimage import _ni_support
from medpy import metric
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion,generate_binary_structure
from scipy import ndimage
import scipy
import networks
from medpy import metric
def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformely accross pseudolabels.
        Args:
            N (int): size of returned iterator.
            images_lists: dict of key (target), value (list of data with this target)
    """

    def __init__(self, N, images_lists):
        self.N = N
        self.images_lists = images_lists
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        size_per_pseudolabel = int(self.N / len(self.images_lists)) + 1
        res = np.zeros(size_per_pseudolabel * len(self.images_lists))

        for i in range(len(self.images_lists)):
            indexes = np.random.choice(
                self.images_lists[i],
                size_per_pseudolabel,
                replace=(len(self.images_lists[i]) <= size_per_pseudolabel)
            )
            res[i * size_per_pseudolabel: (i + 1) * size_per_pseudolabel] = indexes

        np.random.shuffle(res)
        return res[:self.N].astype('int')

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return self.N


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger():
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def _get_od_oc_masks(y, threshold=0.5):
    """Return binary full-disc (OD) and cup (OC) masks.

    The fundus labels use two mutually exclusive channels: disc rim and cup.
    The anatomical optic disc therefore includes both channels.
    """
    y = np.asarray(y)
    if y.ndim < 3 or y.shape[1] < 2:
        raise ValueError("Expected masks with shape [N, 2, ...], got {}".format(y.shape))

    rim = y[:, 0, ...] > threshold
    cup = y[:, 1, ...] > threshold
    return np.logical_or(rim, cup), cup


def _eval_dice(gt_y, pred_y, detail=False):
    if np.shape(gt_y) != np.shape(pred_y):
        raise ValueError(
            "Ground truth and prediction shapes differ: {} vs {}".format(
                np.shape(gt_y), np.shape(pred_y)
            )
        )

    class_names = ("disk", "cup")
    gt_masks = _get_od_oc_masks(gt_y)
    pred_masks = _get_od_oc_masks(pred_y)
    dice = []

    for cls, (gt, pred) in enumerate(zip(gt_masks, pred_masks)):
        denominator = np.count_nonzero(gt) + np.count_nonzero(pred)
        if denominator == 0:
            dice_this = 1.0
        else:
            intersection = np.count_nonzero(np.logical_and(gt, pred))
            dice_this = 2.0 * intersection / denominator
        dice.append(dice_this)

        if detail is True:
            logging.info("class %s, dice is %f", class_names[cls], dice_this)
    return dice

def _connectivity_region_analysis(mask):
    mask = np.asarray(mask)
    label_im, nb_labels = ndimage.label(mask)
    if nb_labels == 0:
        return np.zeros_like(mask, dtype=np.uint8)

    component_ids = np.arange(1, nb_labels + 1)
    sizes = ndimage.sum(mask, label_im, component_ids)
    largest_component = component_ids[np.argmax(sizes)]
    return (label_im == largest_component).astype(np.uint8)

def _eval_average_surface_distances(reference, result, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    return metric.binary.asd(result, reference)


def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()
            
    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)
    
    # test for emptiness
    if 0 == np.count_nonzero(result): 
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference): 
        raise RuntimeError('The second supplied array does not contain any binary object.')    
            
    # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
    
    # compute average surface distance        
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    
    return sds

def asd(result, reference, voxelspacing=None, connectivity=1):
  
    sds = __surface_distances(result, reference, voxelspacing, connectivity)
    asd = sds.mean()
    return asd

def calculate_hausdorff(lP,lT):
    return scipy.spatial.distance.directed_hausdorff(lP, lT)
    # return asd(lP, lT, spacing)

def _eval_haus(pred_y, gt_y, detail=False):
    '''
    :param pred: whole brain prediction
    :param gt: whole
    :param detail:
    :return: a list, indicating Dice of each class for one case
    '''
    class_names = ("disk", "cup")
    gt_masks = _get_od_oc_masks(gt_y)
    pred_masks = _get_od_oc_masks(pred_y)
    haus = []

    for cls, (gt, pred) in enumerate(zip(gt_masks, pred_masks)):
        gt = gt[0]
        pred = pred[0]
# def calculate_metric_percase(pred, gt):
#     dice = metric.binary.dc(pred, gt)
#     jc = metric.binary.jc(pred, gt)
#     hd = metric.binary.hd95(pred, gt)
#     asd = metric.binary.asd(pred, gt)

        # hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
        # hausdorff_distance_filter.Execute(gt_i, pred_i)
        # print (gt.shape)
        gt_nonzero = np.count_nonzero(gt)
        pred_nonzero = np.count_nonzero(pred)
        if gt_nonzero == 0 and pred_nonzero == 0:
            haus_cls = 0.0
        elif gt_nonzero == 0 or pred_nonzero == 0:
            haus_cls = np.inf
        else:
            haus_cls = metric.binary.hd95(gt, pred)

        haus.append(haus_cls)

        if detail is True:
            logging.info("class %s, haus is %f", class_names[cls], haus_cls)
    # logging.info("4 class average haus is {:4f}".format(np.mean(haus)))

    return haus


def parse_fn_haus(data_path):
    '''
    :param image_path: path to a folder of a patient
    :return: normalized entire image with its corresponding label
    In an image, the air region is 0, so we only calculate the mean and std within the brain area
    For any image-level normalization, do it here
    '''
    path = data_path.split(",")
    image_path = path[0]
    label_path = path[1]
    itk_image = sitk.ReadImage(image_path)#os.path.join(image_path, 'T1_unbiased_brain_rigid_to_mni.nii.gz'))
    itk_mask = sitk.ReadImage(label_path)#os.path.join(image_path, 'T1_brain_seg_rigid_to_mni.nii.gz'))
    spacing = itk_mask.GetSpacing()

    image = sitk.GetArrayFromImage(itk_image)
    mask = sitk.GetArrayFromImage(itk_mask)

    mask[mask==2] = 1

    return image.transpose([0, 1,2]), mask.transpose([0, 1,2]), spacing
