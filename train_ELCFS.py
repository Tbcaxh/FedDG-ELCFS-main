# -*- coding: utf-8 -*-
import os
import sys
import csv
import inspect
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
import collections
from collections import OrderedDict
from glob import glob

import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from pytorch_metric_learning import losses

from networks.unet2d import Unet2D
from utils.losses import dice_loss
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis, parse_fn_haus
from utils.memory_routing import (
    DualMemoryRouter,
    build_memory_payload,
    clone_adapter_state,
    compute_content_memory,
    compute_content_memory_resnet,
    compute_style_memory,
    ema_memory_update,
    embedding_to_content_query,
    load_adapter_state,
    low_freq_image_vector_torch,
    weighted_adapter_state,
)
from utils.conflict_consensus import (
    anatomy_auxiliary_losses,
    apply_prefixed_delta,
    clone_prefixed_parameters,
    dynamic_head_consensus,
    parse_prefixes,
    select_named_parameters,
    sparse_anatomy_backward,
)
from dataloaders.fundus_dataloader import TorchDataset, ToTensor, FundusDataset

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--max_epoch', type=int,  default=200, help='maximum epoch number to train')
parser.add_argument('--client_num', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--batch_size', type=int, default=5, help='batch_size per gpu')
parser.add_argument('--clip_value', type=float,  default=100, help='maximum epoch number to train')
parser.add_argument('--meta_step_size', type=float,  default=1e-3, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='6', help='GPU to use')
parser.add_argument('--display_freq', type=int, default=5, help='batch_size per gpu')
# parser 区新增
parser.add_argument('--data_root', type=str,
                    default='/home/users/chenchen/projects/FedDG-ELCFS-main/dataset',
                    help='dataset root path (contains client0..client3)')

parser.add_argument('--unseen_site', type=int, default=0, help='batch_size per gpu')
parser.add_argument('--method', type=str, default='ELCFS', choices=['ELCFS', 'FedAvg', 'CFS'],
                    help='training method: ELCFS (meta+contrastive+CFS), '
                         'FedAvg (no CFS), CFS (CFS augmentation only, no meta/contrastive)')
parser.add_argument('--freq_strategy', type=str, default='mean',
                    choices=['random', 'mean', 'mean_perturb', 'ema_mean', 'ema_mean_perturb'],
                    help='target amplitude strategy for CFS')
parser.add_argument('--freq_perturb_alpha', type=float, default=0.5,
                    help='residual perturb strength for mean_perturb CFS')
parser.add_argument('--freq_ema_momentum', type=float, default=0.9,
                    help='EMA momentum for federated frequency prototypes')
parser.add_argument('--freq_ema_batch_size', type=int, default=16,
                    help='number of amplitude spectra sampled per client to update EMA prototypes; <=0 means all')
parser.add_argument('--freq_peer_strategy', type=str, default='all',
                    choices=['all', 'nearest', 'farthest'],
                    help='which source clients are used as target amplitude peers')
parser.add_argument('--freq_peer_top_k', type=int, default=1,
                    help='number of frequency peers selected for nearest/farthest')
parser.add_argument('--freq_distance_l', type=float, default=0.01,
                    help='low-frequency window ratio used for client prototype distance')
parser.add_argument('--cfs_consistency_weight', type=float, default=0.0,
                    help='prediction consistency weight between raw and CFS-perturbed views')
parser.add_argument('--cfs_consistency_warmup_start', type=int, default=0,
                    help='epoch before which CFS prediction consistency is disabled')
parser.add_argument('--cfs_consistency_warmup_epochs', type=int, default=0,
                    help='linear warmup length for CFS prediction consistency')
parser.add_argument('--grcfs_aggregation', type=int, default=0,
                    help='enable FedAvg-anchored cross-client CFS reliability aggregation')
parser.add_argument('--grcfs_probe_mode', type=str, default='cross',
                    choices=['self', 'cross'],
                    help='self keeps the diagnostic self-domain probe; cross uses source-to-source probing')
parser.add_argument('--grcfs_probe_batches', type=int, default=2,
                    help='number of source-train probe batches per client for GR-CFS statistics')
parser.add_argument('--grcfs_warmup', type=int, default=5,
                    help='number of initial epochs using ordinary FedAvg before GR-CFS weighting')
parser.add_argument('--grcfs_anchor_gamma', type=float, default=0.3,
                    help='blend from FedAvg weights to reliability weights: beta=(1-gamma)FedAvg+gamma*GR-CFS')
parser.add_argument('--grcfs_gamma_ramp_epochs', type=int, default=5,
                    help='epochs used to ramp GR-CFS anchor gamma after warmup; 0 disables ramp')
parser.add_argument('--grcfs_cross_targets_per_round', type=int, default=0,
                    help='number of target source domains sampled per cross-probe round; 0 uses all')
parser.add_argument('--grcfs_cross_demand_mode', type=str, default='client_score',
                    choices=['client_score', 'global_uncertainty', 'relative_gap'],
                    help='cross GR-CFS target weighting: client_score uniformly averages cross-domain reliability; global_uncertainty and relative_gap are diagnostic demand variants')
parser.add_argument('--grcfs_temperature', type=float, default=0.5,
                    help='softmax temperature for GR-CFS reliability weights')
parser.add_argument('--grcfs_demand_temperature', type=float, default=1.0,
                    help='softmax temperature for cross-domain demand weights')
parser.add_argument('--grcfs_ema_momentum', type=float, default=0.8,
                    help='EMA momentum for dynamic aggregation weights; 0 disables smoothing')
parser.add_argument('--grcfs_lambda_gap', type=float, default=1.0,
                    help='scale for normalized reliability score before aggregation softmax')
parser.add_argument('--grcfs_lambda_cross_loss', type=float, default=1.0,
                    help='penalty for normalized cross-domain Dice loss in cross GR-CFS reliability')
parser.add_argument('--grcfs_lambda_inconsistency', type=float, default=1.0,
                    help='penalty for normalized raw-CFS prediction inconsistency')
parser.add_argument('--grcfs_lambda_anatomy', type=float, default=1.0,
                    help='reward for normalized anatomy structure score')
parser.add_argument('--grcfs_lambda_uncertainty', type=float, default=0.5,
                    help='penalty for normalized prediction entropy')
parser.add_argument('--grcfs_lambda_global_loss', type=float, default=0.25,
                    help='weight for global loss when forming cross-domain demand')
parser.add_argument('--grcfs_lambda_global_inconsistency', type=float, default=1.0,
                    help='weight for global raw-CFS inconsistency when forming cross-domain demand')
parser.add_argument('--grcfs_lambda_global_uncertainty', type=float, default=1.0,
                    help='weight for global prediction entropy when forming cross-domain demand')
parser.add_argument('--grcfs_cup_penalty_weight', type=float, default=0.1,
                    help='penalty weight for predicted cup mass outside predicted disc')
parser.add_argument('--grcfs_weight_floor_ratio', type=float, default=0.5,
                    help='lower clamp as a ratio of FedAvg source weight; <=0 disables lower clamp')
parser.add_argument('--grcfs_weight_ceiling_ratio', type=float, default=1.5,
                    help='upper clamp as a ratio of FedAvg source weight; <=0 disables upper clamp')
parser.add_argument('--use_adapter', type=int, default=1,
                    help='use client-specific residual adapters')
parser.add_argument('--adapter_bottleneck', type=int, default=8,
                    help='bottleneck channels for residual adapters')
parser.add_argument('--adapter_scale', type=float, default=1.0,
                    help='residual adapter output scale')
parser.add_argument('--adapter_type', type=str, default='residual',
                    choices=['residual', 'spatial_gate'],
                    help='adapter block type for ablation')
parser.add_argument('--gate_bias', type=float, default=-3.0,
                    help='spatial_gate gate_conv2 bias initialization (sigmoid(bias) = initial gate value)')
parser.add_argument('--eval_adapter', type=str, default='routing',
                    choices=['neutral', 'average', 'routing'],
                    help='test-time adapter strategy for unseen site: neutral (zero residual), '
                         'average (equal-weight source adapters), routing (dual-memory graph routing)')
parser.add_argument('--memory_routing', type=int, default=1,
                    help='(deprecated, use --eval_adapter routing) enable dual memory routing')
parser.add_argument('--memory_low_freq_ratio', type=float, default=0.01,
                    help='low-frequency window ratio used by spectral memory')
parser.add_argument('--content_memory_max_batches', type=int, default=8,
                    help='max batches per client for content prototype memory; <=0 uses all batches')
parser.add_argument('--content_memory_update_interval', type=int, default=5,
                    help='rebuild content memory every N epochs; <=0 builds once and then reuses it')
parser.add_argument('--content_memory_ema_momentum', type=float, default=0.9,
                    help='EMA momentum for content memory updates')
parser.add_argument('--memory_style_weight', type=float, default=0.5,
                    help='weight of spectral/style memory when fusing with content memory')
parser.add_argument('--graph_neighbors', type=int, default=1,
                    help='nearest neighbors per memory node')
parser.add_argument('--graph_temperature', type=float, default=0.1,
                    help='temperature for memory graph edge affinity')
parser.add_argument('--query_temperature', type=float, default=0.1,
                    help='temperature for query-to-memory affinity')
parser.add_argument('--seed_topk', type=int, default=2,
                    help='number of seed memory nodes retrieved by the query')
parser.add_argument('--expert_topk', type=int, default=2,
                    help='number of routed adapter experts used for prediction')
parser.add_argument('--graph_steps', type=int, default=3,
                    help='random-walk diffusion steps on the retrieved memory subgraph')
parser.add_argument('--graph_beta', type=float, default=0.5,
                    help='restart rate for memory graph random walk')
parser.add_argument('--anatomy_aux_weight', type=float, default=0.0,
                    help='weight of view-averaged OD/OC auxiliary Dice loss; 0 keeps F2 unchanged')
parser.add_argument('--anatomy_od_aux_weight', type=float, default=None,
                    help='explicit OD auxiliary Dice weight; defaults to 0.5 * anatomy_aux_weight')
parser.add_argument('--anatomy_oc_aux_weight', type=float, default=None,
                    help='explicit OC auxiliary Dice weight; defaults to 0.5 * anatomy_aux_weight')
parser.add_argument('--anatomy_aux_warmup_start', type=int, default=0,
                    help='epoch before which anatomy auxiliary losses are disabled')
parser.add_argument('--anatomy_aux_warmup_epochs', type=int, default=0,
                    help='linear warmup length for anatomy auxiliary weights after warmup_start')
parser.add_argument('--anatomy_sparse_gate', type=int, default=0,
                    help='softly project conflicting OD/OC gradients on selected head parameters')
parser.add_argument('--anatomy_align_main', type=int, default=0,
                    help='project anatomy head gradients away from conflicts with F2 head gradients')
parser.add_argument('--anatomy_conflict_threshold', type=float, default=0.05,
                    help='negative cosine margin required before anatomy correction starts')
parser.add_argument('--anatomy_gate_temperature', type=float, default=0.1,
                    help='soft anatomy conflict gate temperature')
parser.add_argument('--anatomy_head_prefixes', type=str, default='convu1,seg1',
                    help='comma-separated parameter prefixes corrected by anatomy gating')
parser.add_argument('--server_head_consensus', type=int, default=0,
                    help='enable dynamic conflict-aware aggregation on selected head parameters')
parser.add_argument('--server_head_prefixes', type=str, default='convu1,seg1',
                    help='comma-separated parameter prefixes used by server consensus')
parser.add_argument('--server_consensus_beta', type=float, default=0.2,
                    help='maximum blend from FedAvg head delta to consensus head delta')
parser.add_argument('--server_consensus_target', type=str, default='all',
                    choices=['all', 'od', 'oc'],
                    help='target entries corrected by server consensus')
parser.add_argument('--server_consensus_align_fedavg', type=int, default=0,
                    help='project server correction away from conflicts with FedAvg delta')
parser.add_argument('--server_consensus_norm_cap', type=float, default=0.0,
                    help='max server correction norm as a ratio of FedAvg target norm; 0 disables')
parser.add_argument('--server_conflict_ema_momentum', type=float, default=0.9,
                    help='EMA momentum for server negative-interaction energy')
parser.add_argument('--server_conflict_threshold', type=float, default=0.0,
                    help='EMA conflict score below which server consensus remains inactive')
parser.add_argument('--server_gate_temperature', type=float, default=0.01,
                    help='soft server conflict activation temperature')
parser.add_argument('--server_consensus_warmup', type=int, default=10,
                    help='number of initial rounds using ordinary FedAvg')
args = parser.parse_args()

if args.anatomy_aux_weight < 0:
    parser.error('--anatomy_aux_weight must be non-negative')
if args.cfs_consistency_weight < 0:
    parser.error('--cfs_consistency_weight must be non-negative')
if args.cfs_consistency_warmup_start < 0 or args.cfs_consistency_warmup_epochs < 0:
    parser.error('CFS consistency warmup values must be non-negative')
if args.grcfs_aggregation:
    if args.method != 'CFS':
        parser.error('--grcfs_aggregation currently requires --method CFS')
    if args.use_adapter:
        parser.error('--grcfs_aggregation is designed for the F2 no-adapter baseline; set --use_adapter 0')
    if args.grcfs_probe_batches <= 0:
        parser.error('--grcfs_probe_batches must be positive')
    if args.grcfs_warmup < 0:
        parser.error('--grcfs_warmup must be non-negative')
    if args.grcfs_gamma_ramp_epochs < 0:
        parser.error('--grcfs_gamma_ramp_epochs must be non-negative')
    if args.grcfs_cross_targets_per_round < 0:
        parser.error('--grcfs_cross_targets_per_round must be non-negative')
    if not 0.0 <= args.grcfs_anchor_gamma <= 1.0:
        parser.error('--grcfs_anchor_gamma must be in [0, 1]')
    if args.grcfs_temperature <= 0:
        parser.error('--grcfs_temperature must be positive')
    if args.grcfs_demand_temperature <= 0:
        parser.error('--grcfs_demand_temperature must be positive')
    if not 0.0 <= args.grcfs_ema_momentum < 1.0:
        parser.error('--grcfs_ema_momentum must be in [0, 1)')
    if args.grcfs_lambda_cross_loss < 0:
        parser.error('--grcfs_lambda_cross_loss must be non-negative')
    if args.grcfs_lambda_global_loss < 0:
        parser.error('--grcfs_lambda_global_loss must be non-negative')
    if args.grcfs_lambda_global_inconsistency < 0:
        parser.error('--grcfs_lambda_global_inconsistency must be non-negative')
    if args.grcfs_lambda_global_uncertainty < 0:
        parser.error('--grcfs_lambda_global_uncertainty must be non-negative')
    if args.grcfs_cup_penalty_weight < 0:
        parser.error('--grcfs_cup_penalty_weight must be non-negative')
    if args.grcfs_weight_floor_ratio < 0 or args.grcfs_weight_ceiling_ratio < 0:
        parser.error('GR-CFS clamp ratios must be non-negative')
    if (
            args.grcfs_weight_floor_ratio > 0
            and args.grcfs_weight_ceiling_ratio > 0
            and args.grcfs_weight_ceiling_ratio < args.grcfs_weight_floor_ratio):
        parser.error('--grcfs_weight_ceiling_ratio must be >= --grcfs_weight_floor_ratio')
if args.anatomy_od_aux_weight is not None and args.anatomy_od_aux_weight < 0:
    parser.error('--anatomy_od_aux_weight must be non-negative')
if args.anatomy_oc_aux_weight is not None and args.anatomy_oc_aux_weight < 0:
    parser.error('--anatomy_oc_aux_weight must be non-negative')
if args.anatomy_od_aux_weight is None:
    args.anatomy_od_aux_weight = 0.5 * float(args.anatomy_aux_weight)
if args.anatomy_oc_aux_weight is None:
    args.anatomy_oc_aux_weight = 0.5 * float(args.anatomy_aux_weight)
if args.anatomy_aux_warmup_start < 0 or args.anatomy_aux_warmup_epochs < 0:
    parser.error('anatomy warmup values must be non-negative')
use_anatomy_aux = (
    float(args.anatomy_od_aux_weight) > 0.0
    or float(args.anatomy_oc_aux_weight) > 0.0
)
if args.anatomy_sparse_gate and not use_anatomy_aux:
    parser.error('--anatomy_sparse_gate requires positive OD or OC auxiliary weight')
if (use_anatomy_aux or args.anatomy_sparse_gate) and args.method != 'CFS':
    parser.error('anatomy auxiliary experiments currently require --method CFS')
if args.anatomy_gate_temperature <= 0 or args.server_gate_temperature <= 0:
    parser.error('gate temperatures must be positive')
if not 0.0 <= args.server_consensus_beta <= 1.0:
    parser.error('--server_consensus_beta must be in [0, 1]')
if args.server_consensus_norm_cap < 0:
    parser.error('--server_consensus_norm_cap must be non-negative')
if not 0.0 <= args.server_conflict_ema_momentum < 1.0:
    parser.error('--server_conflict_ema_momentum must be in [0, 1)')

anatomy_head_prefixes = parse_prefixes(args.anatomy_head_prefixes)
server_head_prefixes = parse_prefixes(args.server_head_prefixes)
effective_eval_adapter = args.eval_adapter if args.use_adapter else 'neutral'
use_routing_eval = effective_eval_adapter == 'routing'


snapshot_path = "../output/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
meta_step_size = args.meta_step_size
clip_value = args.clip_value
base_lr = args.base_lr
client_num = args.client_num
max_epoch = args.max_epoch
display_freq = args.display_freq

# 原来是 ['client1', 'client2', 'client3', 'client4']
client_name = [f'client{i}' for i in range(client_num)]
client_data_list = []
for client_idx in range(client_num):
    client_data_list.append(
        glob(os.path.join(args.data_root, client_name[client_idx], 'data_npy', '*.npy'))
    )
    print(len(client_data_list[client_idx]))

slice_num = np.array([max(1, len(files)) for files in client_data_list], dtype=np.float32)
volume_size = [384, 384, 3]
unseen_site_idx = args.unseen_site
source_site_idx = list(range(client_num))
source_site_idx.remove(unseen_site_idx)
client_weight = np.zeros(client_num, dtype=np.float32)
source_weight = slice_num[source_site_idx] / np.sum(slice_num[source_site_idx])
client_weight[source_site_idx] = source_weight
print(client_weight)
num_classes = 3

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def update_global_model(net_clients, client_weight, skip_prefixes=()):
    """
    修复#12: FedAvg 聚合, 动态支持任意数量 client, 不再硬编码4个.
    """
    all_params = [collections.OrderedDict(net.named_parameters()) for net in net_clients]
    for name in all_params[0]:
        if any(name.startswith(prefix) for prefix in skip_prefixes):
            continue
        new_para = torch.zeros_like(all_params[0][name].data)
        for c_idx in range(len(net_clients)):
            new_para.add_(all_params[c_idx][name].data, alpha=float(client_weight[c_idx]))
        for c_idx in range(len(net_clients)):
            all_params[c_idx][name].data.copy_(new_para)

def update_frequency_ema_prototypes(freq_proto_clients, freq_dataset, momentum, sample_size):
    updated_proto_clients = []
    for client_idx, old_proto in enumerate(freq_proto_clients):
        batch_mean = freq_dataset.sample_client_freq_mean(client_idx, sample_size)
        if batch_mean is None:
            updated_proto_clients.append(old_proto)
            continue
        if old_proto is None:
            new_proto = batch_mean
        else:
            new_proto = momentum * old_proto + (1.0 - momentum) * batch_mean
        updated_proto_clients.append(new_proto.astype(np.float32))
    return updated_proto_clients

def extract_contour_embedding(contour_list, embeddings):

    average_embeddings_list = []
    for contour in contour_list:
        contour_embeddings = contour * embeddings
        average_embeddings = torch.sum(contour_embeddings, (-1,-2))/torch.sum(contour, (-1,-2))
    # print (contour.shape)
    # print (embeddings.shape)
    # print (contour_embeddings.shape)
    # print (average_embeddings.shape)
        average_embeddings_list.append(average_embeddings)
    return average_embeddings_list


def _soft_dice_mean(score, target, eps=1e-5):
    target = target.float()
    score = score.float()
    if score.dim() == 4:
        reduce_dims = (2, 3)
    elif score.dim() == 3:
        reduce_dims = (1, 2)
    else:
        reduce_dims = tuple(range(1, score.dim()))
    intersection = torch.sum(score * target, dim=reduce_dims)
    denominator = torch.sum(score, dim=reduce_dims) + torch.sum(target, dim=reduce_dims)
    return ((2.0 * intersection + eps) / (denominator + eps)).mean()


def _binary_entropy_mean(prediction, eps=1e-6):
    prediction = torch.clamp(prediction.float(), eps, 1.0 - eps)
    entropy = -(
        prediction * torch.log(prediction)
        + (1.0 - prediction) * torch.log(1.0 - prediction)
    ) / np.log(2.0)
    return entropy.mean()


def _cfs_prediction_inconsistency(predictions):
    if len(predictions) <= 1:
        return torch.zeros((), device=predictions[0].device)
    raw_prediction = predictions[0]
    distances = [
        1.0 - _soft_dice_mean(raw_prediction, transformed_prediction)
        for transformed_prediction in predictions[1:]
    ]
    return sum(distances) / float(len(distances))


def _anatomy_structure_score(predictions, target, cup_penalty_weight=0.1):
    target_od = torch.clamp(target[:, 0, ...] + target[:, 1, ...], 0.0, 1.0)
    target_oc = target[:, 1, ...]
    scores = []
    violations = []
    for prediction in predictions:
        pred_od = prediction[:, 0, ...] + prediction[:, 1, ...] - prediction[:, 0, ...] * prediction[:, 1, ...]
        pred_oc = prediction[:, 1, ...]
        anatomy_score = (
            0.5 * _soft_dice_mean(pred_od, target_od)
            + 0.5 * _soft_dice_mean(pred_oc, target_oc)
        )
        violation = torch.mean(pred_oc * torch.clamp(1.0 - pred_od, min=0.0))
        scores.append(anatomy_score - float(cup_penalty_weight) * violation)
        violations.append(violation)
    return (
        sum(scores) / float(len(scores)),
        sum(violations) / float(len(violations)),
    )


def _probe_model_loss_and_predictions(model, views, label):
    predictions = []
    losses = []
    for view in views:
        _, prediction, _ = model(view)
        predictions.append(prediction)
        losses.append(dice_loss(prediction, label))
    return sum(losses) / float(len(losses)), predictions


@torch.no_grad()
def collect_grcfs_probe_batches(dataloader, args):
    probe_batches = []
    for batch_index, sampled_batch in enumerate(dataloader):
        if batch_index >= int(args.grcfs_probe_batches):
            break
        probe_batches.append(sampled_batch)
    return probe_batches


@torch.no_grad()
def estimate_grcfs_probe_metrics_on_batches(model, probe_batches, args, device):
    was_training = model.training
    model.eval()

    losses = []
    inconsistencies = []
    anatomy_scores = []
    uncertainties = []
    cup_violations = []

    for sampled_batch in probe_batches:
        volume_batch = sampled_batch['image']
        label_batch = sampled_batch['label'].float().to(device)
        raw_view = volume_batch[:, :3, ...].float().to(device)
        views = [raw_view]
        n_transformed = (volume_batch.shape[1] - 3) // 3
        for t_idx in range(n_transformed):
            start_ch = 3 + t_idx * 3
            views.append(volume_batch[:, start_ch:start_ch + 3, ...].float().to(device))

        loss, predictions = _probe_model_loss_and_predictions(model, views, label_batch)
        inconsistency = _cfs_prediction_inconsistency(predictions)
        anatomy_score, cup_violation = _anatomy_structure_score(
            predictions,
            label_batch,
            cup_penalty_weight=args.grcfs_cup_penalty_weight,
        )
        uncertainty = sum(_binary_entropy_mean(prediction) for prediction in predictions)
        uncertainty = uncertainty / float(len(predictions))

        losses.append(float(loss.detach().cpu()))
        inconsistencies.append(float(inconsistency.detach().cpu()))
        anatomy_scores.append(float(anatomy_score.detach().cpu()))
        uncertainties.append(float(uncertainty.detach().cpu()))
        cup_violations.append(float(cup_violation.detach().cpu()))

    if was_training:
        model.train()

    if not losses:
        return None

    return {
        'loss': float(np.mean(losses)),
        'inconsistency': float(np.mean(inconsistencies)),
        'anatomy': float(np.mean(anatomy_scores)),
        'uncertainty': float(np.mean(uncertainties)),
        'cup_violation': float(np.mean(cup_violations)),
        'probe_batches': len(losses),
    }


@torch.no_grad()
def estimate_grcfs_probe_metrics(model, dataloader, args, device):
    probe_batches = collect_grcfs_probe_batches(dataloader, args)
    return estimate_grcfs_probe_metrics_on_batches(
        model,
        probe_batches,
        args,
        device,
    )


@torch.no_grad()
def estimate_grcfs_client_metrics(local_model, global_model, dataloader, args, device):
    probe_batches = collect_grcfs_probe_batches(dataloader, args)
    global_metrics = estimate_grcfs_probe_metrics_on_batches(
        global_model,
        probe_batches,
        args,
        device,
    )
    local_metrics = estimate_grcfs_probe_metrics_on_batches(
        local_model,
        probe_batches,
        args,
        device,
    )
    if global_metrics is None or local_metrics is None:
        return None

    global_loss = global_metrics['loss']
    local_loss = local_metrics['loss']
    return {
        'global_loss': global_loss,
        'local_loss': local_loss,
        'gap': max(0.0, global_loss - local_loss),
        'inconsistency': local_metrics['inconsistency'],
        'anatomy': local_metrics['anatomy'],
        'uncertainty': local_metrics['uncertainty'],
        'cup_violation': local_metrics['cup_violation'],
        'probe_batches': local_metrics['probe_batches'],
    }


def _standardize(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    std = float(values.std())
    if std < eps:
        return np.zeros_like(values)
    return (values - float(values.mean())) / (std + eps)


def _softmax_np(values, temperature):
    values = np.asarray(values, dtype=np.float64) / max(float(temperature), 1e-12)
    values = values - float(values.max())
    exp_values = np.exp(values)
    return exp_values / np.sum(exp_values)


def _effective_grcfs_gamma(args, epoch_num):
    if epoch_num < int(args.grcfs_warmup):
        return 0.0
    if int(args.grcfs_gamma_ramp_epochs) <= 0:
        return float(args.grcfs_anchor_gamma)
    progress = (
        float(epoch_num - int(args.grcfs_warmup) + 1)
        / float(args.grcfs_gamma_ramp_epochs)
    )
    return float(args.grcfs_anchor_gamma) * min(1.0, max(0.0, progress))


def _select_cross_probe_targets(source_clients, epoch_num, targets_per_round):
    source_clients = list(source_clients)
    if len(source_clients) <= 2:
        return source_clients
    if targets_per_round <= 0 or targets_per_round >= len(source_clients):
        return source_clients
    target_count = max(2, min(int(targets_per_round), len(source_clients)))
    start = int(epoch_num) % len(source_clients)
    return [
        source_clients[(start + offset) % len(source_clients)]
        for offset in range(target_count)
    ]


def _finalize_grcfs_weights(reliability_weights, base_client_weight, source_clients,
                            previous_source_weights, args, effective_gamma):
    source_clients = list(source_clients)
    fedavg_weights = np.asarray(
        [float(base_client_weight[client_id]) for client_id in source_clients],
        dtype=np.float64,
    )
    fedavg_weights = fedavg_weights / max(float(fedavg_weights.sum()), 1e-12)
    reliability_weights = np.asarray(reliability_weights, dtype=np.float64)
    reliability_weights = reliability_weights / max(float(reliability_weights.sum()), 1e-12)

    anchored_weights = (
        (1.0 - float(effective_gamma)) * fedavg_weights
        + float(effective_gamma) * reliability_weights
    )
    lower = None
    upper = None
    if args.grcfs_weight_floor_ratio > 0:
        lower = fedavg_weights * float(args.grcfs_weight_floor_ratio)
    if args.grcfs_weight_ceiling_ratio > 0:
        upper = fedavg_weights * float(args.grcfs_weight_ceiling_ratio)
    if lower is not None or upper is not None:
        if lower is not None:
            anchored_weights = np.maximum(anchored_weights, lower)
        if upper is not None:
            anchored_weights = np.minimum(anchored_weights, upper)
        anchored_weights = anchored_weights / max(float(anchored_weights.sum()), 1e-12)

    source_weights = anchored_weights
    if previous_source_weights is not None and len(previous_source_weights) == len(source_clients):
        if float(effective_gamma) > 0.0:
            momentum = float(args.grcfs_ema_momentum)
            source_weights = momentum * np.asarray(previous_source_weights, dtype=np.float64)
            source_weights = source_weights + (1.0 - momentum) * anchored_weights
            source_weights = source_weights / max(float(source_weights.sum()), 1e-12)

    full_weights = np.zeros_like(base_client_weight, dtype=np.float32)
    for index, client_id in enumerate(source_clients):
        full_weights[client_id] = float(source_weights[index])
    return full_weights, source_weights, fedavg_weights


def build_grcfs_aggregation_weights(client_metrics, base_client_weight,
                                    source_clients, previous_source_weights, args,
                                    effective_gamma=None):
    source_clients = list(source_clients)
    if effective_gamma is None:
        effective_gamma = float(args.grcfs_anchor_gamma)

    gaps = np.asarray([client_metrics[client_id]['gap'] for client_id in source_clients], dtype=np.float64)
    inconsistencies = np.asarray(
        [client_metrics[client_id]['inconsistency'] for client_id in source_clients],
        dtype=np.float64,
    )
    anatomies = np.asarray([client_metrics[client_id]['anatomy'] for client_id in source_clients], dtype=np.float64)
    uncertainties = np.asarray(
        [client_metrics[client_id]['uncertainty'] for client_id in source_clients],
        dtype=np.float64,
    )

    reliability = (
        np.exp(-inconsistencies)
        * np.maximum(anatomies, 0.0)
        * np.exp(-uncertainties)
    )
    gap_reliability = gaps * reliability
    scores = (
        float(args.grcfs_lambda_gap) * _standardize(gap_reliability)
        - float(args.grcfs_lambda_inconsistency) * _standardize(inconsistencies)
        + float(args.grcfs_lambda_anatomy) * _standardize(anatomies)
        - float(args.grcfs_lambda_uncertainty) * _standardize(uncertainties)
    )
    reliability_weights = _softmax_np(scores, args.grcfs_temperature)
    full_weights, source_weights, fedavg_weights = _finalize_grcfs_weights(
        reliability_weights,
        base_client_weight,
        source_clients,
        previous_source_weights,
        args,
        effective_gamma,
    )

    summary = OrderedDict()
    summary['probe_mode'] = 'self'
    for index, client_id in enumerate(source_clients):
        metrics = client_metrics[client_id]
        summary['weight_client{}'.format(client_id)] = float(source_weights[index])
        summary['fedavg_client{}'.format(client_id)] = float(fedavg_weights[index])
        summary['reliability_weight_client{}'.format(client_id)] = float(reliability_weights[index])
        summary['score_client{}'.format(client_id)] = float(scores[index])
        summary['gap_client{}'.format(client_id)] = float(metrics['gap'])
        summary['global_loss_client{}'.format(client_id)] = float(metrics['global_loss'])
        summary['local_loss_client{}'.format(client_id)] = float(metrics['local_loss'])
        summary['inconsistency_client{}'.format(client_id)] = float(metrics['inconsistency'])
        summary['anatomy_client{}'.format(client_id)] = float(metrics['anatomy'])
        summary['uncertainty_client{}'.format(client_id)] = float(metrics['uncertainty'])
        summary['cup_violation_client{}'.format(client_id)] = float(metrics['cup_violation'])
        summary['reliability_client{}'.format(client_id)] = float(reliability[index])
        summary['gap_reliability_client{}'.format(client_id)] = float(gap_reliability[index])
    summary['weight_entropy'] = float(-np.sum(source_weights * np.log(source_weights + 1e-12)))
    summary['anchor_gamma'] = float(effective_gamma)
    return full_weights, source_weights, summary


def build_cross_grcfs_aggregation_weights(pair_metrics, global_metrics, base_client_weight,
                                          source_clients, target_clients,
                                          previous_source_weights, args,
                                          effective_gamma):
    source_clients = list(source_clients)
    target_clients = list(target_clients)
    fedavg_weights = np.asarray(
        [float(base_client_weight[client_id]) for client_id in source_clients],
        dtype=np.float64,
    )
    fedavg_weights = fedavg_weights / max(float(fedavg_weights.sum()), 1e-12)

    pair_keys = [
        (model_id, target_id)
        for target_id in target_clients
        for model_id in source_clients
        if model_id != target_id and (model_id, target_id) in pair_metrics
    ]
    if not pair_keys:
        full_weights, source_weights, _ = _finalize_grcfs_weights(
            fedavg_weights,
            base_client_weight,
            source_clients,
            previous_source_weights,
            args,
            0.0,
        )
        summary = OrderedDict([
            ('probe_mode', 'cross'),
            ('cross_status', 'no_pairs'),
            ('anchor_gamma', 0.0),
        ])
        for index, client_id in enumerate(source_clients):
            summary['weight_client{}'.format(client_id)] = float(source_weights[index])
            summary['fedavg_client{}'.format(client_id)] = float(fedavg_weights[index])
            summary['reliability_weight_client{}'.format(client_id)] = float(fedavg_weights[index])
            summary['score_client{}'.format(client_id)] = 0.0
        return full_weights, source_weights, summary

    pair_loss_z_by_key = {}
    pair_inconsistency_z_by_key = {}
    pair_anatomy_z_by_key = {}
    pair_uncertainty_z_by_key = {}
    pair_reliability_by_key = {}
    for target_id in target_clients:
        target_pair_keys = [key for key in pair_keys if key[1] == target_id]
        if not target_pair_keys:
            continue
        target_losses = np.asarray(
            [pair_metrics[key]['loss'] for key in target_pair_keys],
            dtype=np.float64,
        )
        target_inconsistencies = np.asarray(
            [pair_metrics[key]['inconsistency'] for key in target_pair_keys],
            dtype=np.float64,
        )
        target_anatomies = np.asarray(
            [pair_metrics[key]['anatomy'] for key in target_pair_keys],
            dtype=np.float64,
        )
        target_uncertainties = np.asarray(
            [pair_metrics[key]['uncertainty'] for key in target_pair_keys],
            dtype=np.float64,
        )
        target_loss_z = _standardize(target_losses)
        target_inconsistency_z = _standardize(target_inconsistencies)
        target_anatomy_z = _standardize(target_anatomies)
        target_uncertainty_z = _standardize(target_uncertainties)
        target_reliability = (
            - float(args.grcfs_lambda_cross_loss) * target_loss_z
            - float(args.grcfs_lambda_inconsistency) * target_inconsistency_z
            + float(args.grcfs_lambda_anatomy) * target_anatomy_z
            - float(args.grcfs_lambda_uncertainty) * target_uncertainty_z
        )
        for index, key in enumerate(target_pair_keys):
            pair_loss_z_by_key[key] = float(target_loss_z[index])
            pair_inconsistency_z_by_key[key] = float(target_inconsistency_z[index])
            pair_anatomy_z_by_key[key] = float(target_anatomy_z[index])
            pair_uncertainty_z_by_key[key] = float(target_uncertainty_z[index])
            pair_reliability_by_key[key] = float(target_reliability[index])

    target_demands = OrderedDict((target_id, 0.0) for target_id in target_clients)
    target_demand_scores = OrderedDict(
        (target_id, float('nan')) for target_id in target_clients
    )
    demand_weights = {}
    demand_total = 0.0
    status = 'zero_demand'
    if args.grcfs_cross_demand_mode == 'client_score':
        valid_targets = [
            target_id
            for target_id in target_clients
            if any(key[1] == target_id for key in pair_keys)
        ]
        if valid_targets:
            uniform_weight = 1.0 / float(len(valid_targets))
            for target_id in valid_targets:
                target_demands[target_id] = uniform_weight
                target_demand_scores[target_id] = 0.0
                demand_weights[target_id] = uniform_weight
            demand_total = 1.0
            status = 'active'
    elif args.grcfs_cross_demand_mode == 'relative_gap':
        for target_id in target_clients:
            losses_for_target = [
                pair_metrics[(model_id, target_id)]['loss']
                for model_id in source_clients
                if model_id != target_id and (model_id, target_id) in pair_metrics
            ]
            if not losses_for_target or target_id not in global_metrics:
                continue
            target_demands[target_id] = max(
                0.0,
                float(global_metrics[target_id]['loss']) - float(np.mean(losses_for_target)),
            )
        demand_total = float(sum(target_demands.values()))
        if demand_total > 1e-12:
            demand_weights = {
                target_id: target_demands[target_id] / demand_total
                for target_id in target_demands
            }
            status = 'active'
    else:
        valid_targets = [
            target_id
            for target_id in target_clients
            if (
                target_id in global_metrics
                and any(
                    model_id != target_id and (model_id, target_id) in pair_metrics
                    for model_id in source_clients
                )
            )
        ]
        if valid_targets:
            global_losses = np.asarray(
                [global_metrics[target_id]['loss'] for target_id in valid_targets],
                dtype=np.float64,
            )
            global_inconsistencies = np.asarray(
                [global_metrics[target_id]['inconsistency'] for target_id in valid_targets],
                dtype=np.float64,
            )
            global_uncertainties = np.asarray(
                [global_metrics[target_id]['uncertainty'] for target_id in valid_targets],
                dtype=np.float64,
            )
            global_demand_scores = (
                float(args.grcfs_lambda_global_loss) * _standardize(global_losses)
                + float(args.grcfs_lambda_global_inconsistency) * _standardize(global_inconsistencies)
                + float(args.grcfs_lambda_global_uncertainty) * _standardize(global_uncertainties)
            )
            demand_values = _softmax_np(
                global_demand_scores,
                args.grcfs_demand_temperature,
            )
            for index, target_id in enumerate(valid_targets):
                target_demand_scores[target_id] = float(global_demand_scores[index])
                target_demands[target_id] = float(demand_values[index])
                demand_weights[target_id] = float(demand_values[index])
            demand_total = float(sum(demand_weights.values()))
            status = 'active'

    raw_scores = np.zeros(len(source_clients), dtype=np.float64)
    score_norms = np.zeros(len(source_clients), dtype=np.float64)
    if status == 'active' and demand_total > 1e-12:
        index_by_client = {client_id: index for index, client_id in enumerate(source_clients)}
        for model_id, target_id in pair_keys:
            model_index = index_by_client[model_id]
            demand_weight = float(demand_weights.get(target_id, 0.0))
            raw_scores[model_index] += demand_weight * pair_reliability_by_key[(model_id, target_id)]
            score_norms[model_index] += demand_weight
        for index in range(len(raw_scores)):
            if score_norms[index] > 1e-12:
                raw_scores[index] /= score_norms[index]
        if float(raw_scores.std()) > 1e-12:
            scores = float(args.grcfs_lambda_gap) * _standardize(raw_scores)
            reliability_weights = _softmax_np(scores, args.grcfs_temperature)
            status = 'active'
        else:
            scores = np.zeros(len(source_clients), dtype=np.float64)
            reliability_weights = fedavg_weights.copy()
            status = 'flat_score'
    else:
        scores = np.zeros(len(source_clients), dtype=np.float64)
        reliability_weights = fedavg_weights.copy()
        status = 'zero_demand'

    full_weights, source_weights, fedavg_weights = _finalize_grcfs_weights(
        reliability_weights,
        base_client_weight,
        source_clients,
        previous_source_weights,
        args,
        effective_gamma if status == 'active' else 0.0,
    )

    summary = OrderedDict()
    summary['probe_mode'] = 'cross'
    summary['cross_status'] = status
    summary['cross_demand_mode'] = args.grcfs_cross_demand_mode
    summary['target_clients'] = '|'.join(str(client_id) for client_id in target_clients)
    summary['pair_count'] = int(len(pair_keys))
    summary['demand_total'] = float(demand_total)
    summary['anchor_gamma'] = float(effective_gamma if status == 'active' else 0.0)
    for index, client_id in enumerate(source_clients):
        client_pair_keys = [key for key in pair_keys if key[0] == client_id]
        summary['weight_client{}'.format(client_id)] = float(source_weights[index])
        summary['fedavg_client{}'.format(client_id)] = float(fedavg_weights[index])
        summary['reliability_weight_client{}'.format(client_id)] = float(reliability_weights[index])
        summary['score_client{}'.format(client_id)] = float(scores[index])
        summary['raw_score_client{}'.format(client_id)] = float(raw_scores[index])
        if client_pair_keys:
            summary['cross_loss_client{}'.format(client_id)] = float(np.mean([
                pair_metrics[key]['loss'] for key in client_pair_keys
            ]))
            summary['cross_anatomy_client{}'.format(client_id)] = float(np.mean([
                pair_metrics[key]['anatomy'] for key in client_pair_keys
            ]))
            summary['cross_inconsistency_client{}'.format(client_id)] = float(np.mean([
                pair_metrics[key]['inconsistency'] for key in client_pair_keys
            ]))
            summary['cross_uncertainty_client{}'.format(client_id)] = float(np.mean([
                pair_metrics[key]['uncertainty'] for key in client_pair_keys
            ]))
    for target_id in source_clients:
        summary['demand_client{}'.format(target_id)] = float(target_demands.get(target_id, float('nan')))
        summary['demand_weight_client{}'.format(target_id)] = float(demand_weights.get(target_id, 0.0))
        summary['demand_score_client{}'.format(target_id)] = float(target_demand_scores.get(target_id, float('nan')))
        if target_id in global_metrics:
            summary['global_loss_client{}'.format(target_id)] = float(global_metrics[target_id]['loss'])
            summary['global_inconsistency_client{}'.format(target_id)] = float(global_metrics[target_id]['inconsistency'])
            summary['global_anatomy_client{}'.format(target_id)] = float(global_metrics[target_id]['anatomy'])
            summary['global_uncertainty_client{}'.format(target_id)] = float(global_metrics[target_id]['uncertainty'])
        else:
            summary['global_loss_client{}'.format(target_id)] = float('nan')
            summary['global_inconsistency_client{}'.format(target_id)] = float('nan')
            summary['global_anatomy_client{}'.format(target_id)] = float('nan')
            summary['global_uncertainty_client{}'.format(target_id)] = float('nan')
    for target_id in source_clients:
        for model_id in source_clients:
            if model_id == target_id:
                continue
            key = (model_id, target_id)
            prefix = 'pair_m{}_t{}'.format(model_id, target_id)
            if key in pair_metrics:
                summary[prefix + '_loss'] = float(pair_metrics[key]['loss'])
                summary[prefix + '_loss_z'] = pair_loss_z_by_key[key]
                summary[prefix + '_anatomy'] = float(pair_metrics[key]['anatomy'])
                summary[prefix + '_anatomy_z'] = pair_anatomy_z_by_key[key]
                summary[prefix + '_reliability'] = pair_reliability_by_key[key]
                summary[prefix + '_inconsistency'] = float(pair_metrics[key]['inconsistency'])
                summary[prefix + '_inconsistency_z'] = pair_inconsistency_z_by_key[key]
                summary[prefix + '_uncertainty'] = float(pair_metrics[key]['uncertainty'])
                summary[prefix + '_uncertainty_z'] = pair_uncertainty_z_by_key[key]
            else:
                summary[prefix + '_loss'] = float('nan')
                summary[prefix + '_loss_z'] = float('nan')
                summary[prefix + '_anatomy'] = float('nan')
                summary[prefix + '_anatomy_z'] = float('nan')
                summary[prefix + '_reliability'] = float('nan')
                summary[prefix + '_inconsistency'] = float('nan')
                summary[prefix + '_inconsistency_z'] = float('nan')
                summary[prefix + '_uncertainty'] = float('nan')
                summary[prefix + '_uncertainty_z'] = float('nan')
    summary['weight_entropy'] = float(-np.sum(source_weights * np.log(source_weights + 1e-12)))
    return full_weights, source_weights, summary

def test(site_index, test_net, eval_adapter='routing', router=None,
         adapter_states=None, neutral_adapter_state=None, avg_adapter_state=None,
         content_encoder=None):

    test_data_list = client_data_list[site_index]
    device = next(test_net.parameters()).device

    # Pre-load static adapter for 'neutral' or 'average' modes (one-shot, outside loop)
    if eval_adapter == 'neutral' and neutral_adapter_state is not None:
        load_adapter_state(test_net, neutral_adapter_state, device=device)
    elif eval_adapter == 'average' and avg_adapter_state is not None:
        load_adapter_state(test_net, avg_adapter_state, device=device)

    dice_array = []
    with torch.no_grad():
        for fid, filename in enumerate(test_data_list):
            data = np.load(filename)
            image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
            mask = np.expand_dims(data[..., 3:].transpose(2, 0, 1), axis=0)
            image = torch.from_numpy(image).float().to(device)

            if eval_adapter == 'routing':
                # Reset to neutral first, then route
                if neutral_adapter_state is not None:
                    load_adapter_state(test_net, neutral_adapter_state, device=device)
                if router is not None and adapter_states:
                    style_query = low_freq_image_vector_torch(image, args.memory_low_freq_ratio)
                    if content_encoder is not None:
                        content_query = content_encoder(image)
                    else:
                        _, _, query_embedding = test_net(image, use_adapter=False)
                        content_query = embedding_to_content_query(query_embedding)
                    route_weights = router.route(style_query, content_query)
                    routed_adapter = weighted_adapter_state(adapter_states, route_weights, device=device)
                    if routed_adapter is not None:
                        load_adapter_state(test_net, routed_adapter, device=device)

            logit, pred, _ = test_net(image)
            pred_y = pred.cpu().detach().numpy()
            pred_y[pred_y>0.75] = 1
            pred_y[pred_y<0.75] = 0

            pred_y_0 = pred_y[:, 0:1, ...]
            pred_y_1 = pred_y[:, 1:, ...]
            processed_pred_y_0 = _connectivity_region_analysis(pred_y_0)
            processed_pred_y_1 = _connectivity_region_analysis(pred_y_1)
            processed_pred_y = np.concatenate([processed_pred_y_0, processed_pred_y_1], axis=1)
            dice_subject = _eval_dice(mask, processed_pred_y)
            # haus_subject = _eval_haus(mask, processed_pred_y)
            dice_array.append(dice_subject)
            # haus_array.append(haus_subject)
    if neutral_adapter_state is not None:
        load_adapter_state(test_net, neutral_adapter_state, device=device)
    dice_array = np.array(dice_array)
    dice_avg = np.mean(dice_array, axis=0).tolist()
    logging.info("OD dice_avg %.4f OC dice_avg %.4f" % (dice_avg[0], dice_avg[1]))
    return dice_avg, dice_array, 0, [0,0]

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + '/model'):
        os.makedirs(snapshot_path + '/model')
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree(
    '.',
    snapshot_path + '/code',
    ignore=shutil.ignore_patterns('.git', '__pycache__', 'output', 'dataset', 'dataset2', 'logs', 'tmp')
)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    if args.eval_adapter != effective_eval_adapter:
        logging.warning(
            'Using effective eval_adapter=%s because use_adapter=%d; requested eval_adapter=%s is ignored.',
            effective_eval_adapter,
            args.use_adapter,
            args.eval_adapter,
        )
    logging.info(
        'Effective anatomy setup: od_weight=%.6f oc_weight=%.6f warmup_start=%d warmup_epochs=%d sparse_gate=%d align_main=%d',
        args.anatomy_od_aux_weight,
        args.anatomy_oc_aux_weight,
        args.anatomy_aux_warmup_start,
        args.anatomy_aux_warmup_epochs,
        args.anatomy_sparse_gate,
        args.anatomy_align_main,
    )
    logging.info(
        'Effective server consensus: target=%s beta=%.6f align_fedavg=%d norm_cap=%.6f',
        args.server_consensus_target,
        args.server_consensus_beta,
        args.server_consensus_align_fedavg,
        args.server_consensus_norm_cap,
    )
    logging.info(
        'Effective CFS consistency: weight=%.6f warmup_start=%d warmup_epochs=%d',
        args.cfs_consistency_weight,
        args.cfs_consistency_warmup_start,
        args.cfs_consistency_warmup_epochs,
    )
    logging.info(
        'Effective GR-CFS aggregation: enabled=%d mode=%s demand_mode=%s probe_batches=%d warmup=%d gamma=%.4f ramp=%d cross_targets=%d tau=%.4f demand_tau=%.4f ema=%.4f lambdas=(gap %.4f cross_loss %.4f incons %.4f anatomy %.4f uncertainty %.4f global_loss %.4f global_incons %.4f global_uncert %.4f) clamp=(%.4f, %.4f)',
        args.grcfs_aggregation,
        args.grcfs_probe_mode,
        args.grcfs_cross_demand_mode,
        args.grcfs_probe_batches,
        args.grcfs_warmup,
        args.grcfs_anchor_gamma,
        args.grcfs_gamma_ramp_epochs,
        args.grcfs_cross_targets_per_round,
        args.grcfs_temperature,
        args.grcfs_demand_temperature,
        args.grcfs_ema_momentum,
        args.grcfs_lambda_gap,
        args.grcfs_lambda_cross_loss,
        args.grcfs_lambda_inconsistency,
        args.grcfs_lambda_anatomy,
        args.grcfs_lambda_uncertainty,
        args.grcfs_lambda_global_loss,
        args.grcfs_lambda_global_inconsistency,
        args.grcfs_lambda_global_uncertainty,
        args.grcfs_weight_floor_ratio,
        args.grcfs_weight_ceiling_ratio,
    )


    # define dataset, model, optimizer for each client
    # 修复#10: unseen client 只创建 model (用于聚合接收), 不创建训练 dataloader 和 optimizer
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    dataloader_clients = {}
    net_clients = []
    optimizer_clients = {}
    dataset_clients = {}
    requested_model_config = {
        'use_adapter': bool(args.use_adapter),
        'adapter_bottleneck': args.adapter_bottleneck,
        'adapter_scale': args.adapter_scale,
        'adapter_type': args.adapter_type,
        'gate_bias': args.gate_bias,
    }
    unet_signature = inspect.signature(Unet2D.__init__)
    accepts_model_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in unet_signature.parameters.values()
    )
    if accepts_model_kwargs:
        model_config = requested_model_config
        ignored_model_keys = []
    else:
        supported_model_keys = set(unet_signature.parameters) - {'self'}
        model_config = {
            key: value
            for key, value in requested_model_config.items()
            if key in supported_model_keys
        }
        ignored_model_keys = sorted(
            set(requested_model_config) - set(model_config)
        )
    if ignored_model_keys:
        logging.warning(
            'Ignoring unsupported Unet2D constructor keys: %s',
            ', '.join(ignored_model_keys),
        )
    for client_idx in range(client_num):
        net = Unet2D(**model_config)
        net = net.cuda()
        net_clients.append(net)

        if client_idx == unseen_site_idx:
            # unseen client 只需要模型用于聚合, 不需要 dataloader 和 optimizer
            continue

        if args.method == 'FedAvg':
            freq_site_idx = []
        elif args.method in ('ELCFS', 'CFS'):
            freq_site_idx = source_site_idx.copy()
            freq_site_idx.remove(client_idx)
        else:
            freq_site_idx = []

        dataset = FundusDataset(
            client_idx=client_idx,
            freq_site_idx=freq_site_idx,
            split='train',
            data_root=args.data_root,
            transform=None,
            freq_strategy=args.freq_strategy,
            freq_perturb_alpha=args.freq_perturb_alpha,
            freq_peer_strategy=args.freq_peer_strategy,
            freq_peer_top_k=args.freq_peer_top_k,
            freq_distance_l=args.freq_distance_l
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
        dataset_clients[client_idx] = dataset
        dataloader_clients[client_idx] = dataloader
        optimizer_clients[client_idx] = optimizer

    neutral_adapter_state = clone_adapter_state(net_clients[unseen_site_idx])
    anatomy_parameters_clients = {}
    if args.anatomy_sparse_gate:
        anatomy_parameters_clients = {
            client_idx: select_named_parameters(
                net_clients[client_idx], anatomy_head_prefixes
            )
            for client_idx in source_site_idx
        }
        logging.info(
            "Sparse anatomy gate parameters: prefixes=%s tensors=%d values=%d",
            anatomy_head_prefixes,
            len(anatomy_parameters_clients[source_site_idx[0]]),
            sum(
                parameter.numel()
                for _, parameter in anatomy_parameters_clients[source_site_idx[0]]
            ),
        )

    for name, param in  net_clients[0].named_parameters():
        print (name)

    temperature = 0.05
    cont_loss_func = losses.NTXentLoss(temperature)  if args.method == 'ELCFS' else None

    # start federated learning
    writer = SummaryWriter(snapshot_path+'/log')
    server_conflict_ema = None
    grcfs_weight_ema = None
    server_diagnostic_path = os.path.join(snapshot_path, 'server_consensus.csv')
    grcfs_diagnostic_path = os.path.join(snapshot_path, 'grcfs_aggregation.csv')
    use_ema_freq = args.freq_strategy in ['ema_mean', 'ema_mean_perturb']
    freq_proto_clients = None
    freq_proto_dataset = None
    if use_ema_freq and len(dataset_clients) > 0:
        freq_proto_dataset = next(iter(dataset_clients.values()))
        freq_proto_clients = freq_proto_dataset.get_initial_freq_prototypes()
        for dataset in dataset_clients.values():
            dataset.set_freq_prototypes(freq_proto_clients)
        logging.info(
            "Initialized EMA frequency prototypes with static client means; momentum %.4f batch_size %d",
            args.freq_ema_momentum,
            args.freq_ema_batch_size
        )

    # best trackers
    best_od = -1.0
    best_od_epoch = -1

    best_oc = -1.0
    best_oc_epoch = -1

    best_avg = -1.0
    best_avg_epoch = -1
    best_avg_od = -1.0
    best_avg_oc = -1.0

    lr_ = base_lr

    # ---------- one-shot memory construction (before training) ----------
    # Style memory: built once from static .npy files (mean_perturb strategy).
    # Content memory: built once with frozen ResNet-18, which preserves
    # domain-discriminative features without being eroded by ELCFS domain-
    # invariance training.
    style_memory_cache = None
    content_memory_cache = None
    content_encoder = None
    if use_routing_eval:
        device = next(net_clients[unseen_site_idx].parameters()).device
        style_memory_cache = compute_style_memory(
            args.data_root,
            client_name,
            source_site_idx,
            low_freq_ratio=args.memory_low_freq_ratio,
            freq_proto_clients=freq_proto_clients,
        )
        content_memory_cache, content_encoder = compute_content_memory_resnet(
            args.data_root,
            client_name,
            source_site_idx,
            device=device,
            max_images=args.content_memory_max_batches * args.batch_size,
        )
        logging.info(
            "One-shot memory ready: style=%s content=%s",
            style_memory_cache['client_ids'] if style_memory_cache else None,
            content_memory_cache['client_ids'] if content_memory_cache else None,
        )

    for epoch_num in tqdm(range(max_epoch), ncols=70):
        anatomy_epoch_metrics = []
        server_reference = None
        server_start_epoch = max(1, int(args.server_consensus_warmup))
        if args.server_head_consensus and epoch_num >= server_start_epoch:
            # Epoch 0 starts from independently initialized client models in the
            # legacy trainer. From epoch 1 onward all clients share the previous
            # FedAvg model, so their local deltas have a valid common origin.
            server_reference = clone_prefixed_parameters(
                net_clients[unseen_site_idx], server_head_prefixes
            )
        if use_ema_freq and freq_proto_dataset is not None:
            freq_proto_clients = update_frequency_ema_prototypes(
                freq_proto_clients,
                freq_proto_dataset,
                args.freq_ema_momentum,
                args.freq_ema_batch_size
            )
            for dataset in dataset_clients.values():
                dataset.set_freq_prototypes(freq_proto_clients)
            logging.info("Updated EMA frequency prototypes at epoch %d", epoch_num)

        for client_idx in source_site_idx:
            dataloader_current = dataloader_clients[client_idx]
            net_current = net_clients[client_idx]
            net_current.train()
            optimizer_current = optimizer_clients[client_idx]
            time1 = time.time()
            iter_num = 0

            for i_batch, sampled_batch in enumerate(dataloader_current):
                # common tensors
                volume_batch = sampled_batch['image']
                label_batch = sampled_batch['label'].cuda()
                volume_batch_raw_np = volume_batch[:, :3, ...]
                volume_batch_raw = volume_batch_raw_np.cuda()

                if args.method == 'FedAvg':
                    # ---------------- FedAvg local training ----------------
                    logits, pred, _ = net_current(volume_batch_raw)
                    total_loss = dice_loss(pred, label_batch)

                    optimizer_current.zero_grad()
                    total_loss.backward()
                    optimizer_current.step()

                    iter_num += 1
                    if iter_num % display_freq == 0:
                        writer.add_scalar('lr', lr_, iter_num)
                        writer.add_scalar('loss/total', total_loss, iter_num)
                        logging.info(
                            'Epoch: [%d] client [%d] iteration [%d / %d] : fedavg loss : %f' %
                            (epoch_num, client_idx, iter_num, len(dataloader_current), total_loss.item())
                        )

                    if iter_num % 20 == 0:
                        image = np.array(volume_batch_raw_np[0, 0:3, :, :], dtype='uint8')
                        writer.add_image('train/RawImage', image, iter_num)
                        pred_vis = torch.sigmoid(logits)
                        image = pred_vis[0, 0:1, ...].data.cpu().numpy()
                        writer.add_image('train/RawDiskMask', image, iter_num)
                        image = pred_vis[0, 1:, ...].data.cpu().numpy()
                        writer.add_image('train/RawCupMask', image, iter_num)

                elif args.method == 'CFS':
                    # ---------------- CFS-only training ----------------
                    # Supervised learning on original + K-1 CFS images,
                    # no meta-learning, no contrastive loss.
                    n_transformed = (volume_batch.shape[1] - 3) // 3

                    logits, pred, _ = net_current(volume_batch_raw)
                    predictions = [pred]
                    f2_loss = dice_loss(pred, label_batch)

                    for t_idx in range(n_transformed):
                        start_ch = 3 + t_idx * 3
                        trs_img = volume_batch[:, start_ch:start_ch + 3, ...].cuda()
                        _, pred_t, _ = net_current(trs_img)
                        predictions.append(pred_t)
                        f2_loss = f2_loss + dice_loss(pred_t, label_batch)
                    f2_loss = f2_loss / float(n_transformed + 1)

                    od_loss = None
                    oc_loss = None
                    anatomy_loss = None
                    consistency_loss = None
                    base_loss = f2_loss
                    total_loss = base_loss
                    consistency_scale = 0.0
                    if args.cfs_consistency_weight > 0 and len(predictions) > 1:
                        if epoch_num >= int(args.cfs_consistency_warmup_start):
                            if int(args.cfs_consistency_warmup_epochs) > 0:
                                consistency_scale = min(
                                    1.0,
                                    float(
                                        epoch_num - int(args.cfs_consistency_warmup_start) + 1
                                    ) / float(args.cfs_consistency_warmup_epochs)
                                )
                            else:
                                consistency_scale = 1.0
                    effective_consistency_weight = (
                        float(args.cfs_consistency_weight) * consistency_scale
                    )
                    if effective_consistency_weight > 0.0:
                        raw_prediction = predictions[0].detach()
                        consistency_loss = sum(
                            F.mse_loss(prediction, raw_prediction)
                            for prediction in predictions[1:]
                        ) / float(len(predictions) - 1)
                        base_loss = (
                            f2_loss
                            + effective_consistency_weight * consistency_loss
                        )
                        total_loss = base_loss
                    anatomy_scale = 0.0
                    if use_anatomy_aux:
                        if epoch_num >= int(args.anatomy_aux_warmup_start):
                            if int(args.anatomy_aux_warmup_epochs) > 0:
                                anatomy_scale = min(
                                    1.0,
                                    float(
                                        epoch_num - int(args.anatomy_aux_warmup_start) + 1
                                    ) / float(args.anatomy_aux_warmup_epochs)
                                )
                            else:
                                anatomy_scale = 1.0
                    effective_od_weight = float(args.anatomy_od_aux_weight) * anatomy_scale
                    effective_oc_weight = float(args.anatomy_oc_aux_weight) * anatomy_scale
                    use_effective_anatomy = (
                        effective_od_weight > 0.0 or effective_oc_weight > 0.0
                    )
                    if use_effective_anatomy:
                        od_loss, oc_loss = anatomy_auxiliary_losses(
                            predictions, label_batch
                        )
                        anatomy_loss = (
                            effective_od_weight * od_loss
                            + effective_oc_weight * oc_loss
                        )
                        total_loss = (
                            base_loss
                            + anatomy_loss
                        )

                    optimizer_current.zero_grad(set_to_none=True)
                    if args.anatomy_sparse_gate and use_effective_anatomy:
                        anatomy_metrics = sparse_anatomy_backward(
                            base_loss=base_loss,
                            od_loss=od_loss,
                            oc_loss=oc_loss,
                            named_parameters=anatomy_parameters_clients[client_idx],
                            od_weight=effective_od_weight,
                            oc_weight=effective_oc_weight,
                            align_main=bool(args.anatomy_align_main),
                            conflict_threshold=args.anatomy_conflict_threshold,
                            gate_temperature=args.anatomy_gate_temperature,
                        )
                        anatomy_epoch_metrics.append(anatomy_metrics)
                    else:
                        total_loss.backward()
                    optimizer_current.step()

                    iter_num += 1
                    if iter_num % display_freq == 0:
                        writer.add_scalar('lr', lr_, iter_num)
                        writer.add_scalar('loss/total', total_loss, iter_num)
                        writer.add_scalar('loss/f2', f2_loss, iter_num)
                        if consistency_loss is not None:
                            writer.add_scalar('loss/cfs_consistency', consistency_loss, iter_num)
                            writer.add_scalar(
                                'cfs_consistency/weight_scale',
                                consistency_scale,
                                iter_num,
                            )
                            writer.add_scalar(
                                'cfs_consistency/weight_effective',
                                effective_consistency_weight,
                                iter_num,
                            )
                        if anatomy_loss is not None:
                            writer.add_scalar('loss/anatomy', anatomy_loss, iter_num)
                            writer.add_scalar('loss/od_aux', od_loss, iter_num)
                            writer.add_scalar('loss/oc_aux', oc_loss, iter_num)
                            writer.add_scalar('anatomy/weight_scale', anatomy_scale, iter_num)
                            writer.add_scalar('anatomy/od_weight_effective', effective_od_weight, iter_num)
                            writer.add_scalar('anatomy/oc_weight_effective', effective_oc_weight, iter_num)
                        if args.anatomy_sparse_gate and use_effective_anatomy:
                            writer.add_scalar(
                                'anatomy/cosine', anatomy_metrics['cosine'], iter_num
                            )
                            writer.add_scalar(
                                'anatomy/gate', anatomy_metrics['gate'], iter_num
                            )
                        logging.info(
                            'Epoch: [%d] client [%d] iteration [%d / %d] : '
                            'f2 loss : %f anatomy loss : %s total loss : %f' %
                            (
                                epoch_num,
                                client_idx,
                                iter_num,
                                len(dataloader_current),
                                f2_loss.item(),
                                'none' if anatomy_loss is None else '{:.6f}'.format(
                                    anatomy_loss.item()
                                ),
                                total_loss.item(),
                            )
                        )

                    if iter_num % 20 == 0:
                        image = np.array(volume_batch_raw_np[0, 0:3, :, :], dtype='uint8')
                        writer.add_image('train/RawImage', image, iter_num)
                        pred_vis = torch.sigmoid(logits)
                        image = pred_vis[0, 0:1, ...].data.cpu().numpy()
                        writer.add_image('train/RawDiskMask', image, iter_num)
                        image = pred_vis[0, 1:, ...].data.cpu().numpy()
                        writer.add_image('train/RawCupMask', image, iter_num)

                else:
                    # ---------------- ELCFS original training ----------------
                    disc_contour = sampled_batch['disc_contour'].cuda()
                    disc_bg = sampled_batch['disc_bg'].cuda()
                    cup_contour = sampled_batch['cup_contour'].cuda()
                    cup_bg = sampled_batch['cup_bg'].cuda()

                    # K-1 transformed images
                    n_transformed = (volume_batch.shape[1] - 3) // 3
                    volume_batch_trs_list = []
                    volume_batch_trs_np_list = []
                    for t_idx in range(n_transformed):
                        start_ch = 3 + t_idx * 3
                        trs_np = volume_batch[:, start_ch:start_ch + 3, ...]
                        volume_batch_trs_np_list.append(trs_np)
                        volume_batch_trs_list.append(trs_np.cuda())

                    # Inner loop
                    logits_inner, pred_inner, embedding_inner = net_current(volume_batch_raw)
                    loss_inner = dice_loss(pred_inner, label_batch)
                    grads = torch.autograd.grad(loss_inner, net_current.parameters(), retain_graph=True)

                    fast_weights = OrderedDict(
                        (name, param - torch.mul(meta_step_size, torch.clamp(grad, -clip_value, clip_value)))
                        for ((name, param), grad) in zip(net_current.named_parameters(), grads)
                    )

                    # Outer loop
                    embedding_outer_list = []
                    loss_outer_dice = 0.0
                    for t_idx in range(n_transformed):
                        logits_out, pred_out, emb_out = net_current(volume_batch_trs_list[t_idx], fast_weights)
                        loss_outer_dice += dice_loss(pred_out, label_batch)
                        embedding_outer_list.append(emb_out)
                    loss_outer_dice = loss_outer_dice / max(n_transformed, 1)

                    # Contrastive loss
                    inner_disc_ct_em, inner_disc_bg_em, inner_cup_ct_em, inner_cup_bg_em = \
                        extract_contour_embedding([disc_contour, disc_bg, cup_contour, cup_bg], embedding_inner)

                    all_disc_ct = [inner_disc_ct_em]
                    all_disc_bg = [inner_disc_bg_em]
                    all_cup_ct = [inner_cup_ct_em]
                    all_cup_bg = [inner_cup_bg_em]
                    for emb_out in embedding_outer_list:
                        out_disc_ct, out_disc_bg, out_cup_ct, out_cup_bg = \
                            extract_contour_embedding([disc_contour, disc_bg, cup_contour, cup_bg], emb_out)
                        all_disc_ct.append(out_disc_ct)
                        all_disc_bg.append(out_disc_bg)
                        all_cup_ct.append(out_cup_ct)
                        all_cup_bg.append(out_cup_bg)

                    disc_ct_em = torch.cat(all_disc_ct, 0)
                    disc_bg_em = torch.cat(all_disc_bg, 0)
                    cup_ct_em = torch.cat(all_cup_ct, 0)
                    cup_bg_em = torch.cat(all_cup_bg, 0)
                    disc_em = torch.cat((disc_ct_em, disc_bg_em), 0)
                    cup_em = torch.cat((cup_ct_em, cup_bg_em), 0)

                    label = np.concatenate([np.ones(disc_ct_em.shape[0]), np.zeros(disc_bg_em.shape[0])])
                    label = torch.from_numpy(label).long().to(disc_em.device)

                    disc_cont_loss = cont_loss_func(disc_em, label)
                    cup_cont_loss = cont_loss_func(cup_em, label)
                    cont_loss = disc_cont_loss + cup_cont_loss
                    loss_outer = loss_outer_dice + cont_loss * 0.1
                    total_loss = loss_inner + loss_outer

                    optimizer_current.zero_grad()
                    total_loss.backward()
                    optimizer_current.step()

                    iter_num += 1
                    if iter_num % display_freq == 0:
                        writer.add_scalar('lr', lr_, iter_num)
                        writer.add_scalar('loss/inner', loss_inner, iter_num)
                        writer.add_scalar('loss/outer', loss_outer, iter_num)
                        writer.add_scalar('loss/total', total_loss, iter_num)
                        logging.info(
                            'Epoch: [%d] client [%d] iteration [%d / %d] : inner loss : %f outer dice loss : %f outer cont loss : %f outer loss : %f total loss : %f' %
                            (epoch_num, client_idx, iter_num, len(dataloader_current),
                             loss_inner.item(), loss_outer_dice.item(), cont_loss.item(), loss_outer.item(),
                             total_loss.item())
                        )

                    if iter_num % 20 == 0:
                        image = np.array(volume_batch_raw_np[0, 0:3, :, :], dtype='uint8')
                        writer.add_image('train/RawImage', image, iter_num)

                        for t_idx, trs_np in enumerate(volume_batch_trs_np_list):
                            image = np.array(trs_np[0, 0:3, :, :], dtype='uint8')
                            writer.add_image(f'train/TrsImage_{t_idx}', image, iter_num)

                        pred_inner_vis = torch.sigmoid(logits_inner)
                        image = pred_inner_vis[0, 0:1, ...].data.cpu().numpy()
                        writer.add_image('train/RawDiskMask', image, iter_num)
                        image = pred_inner_vis[0, 1:, ...].data.cpu().numpy()
                        writer.add_image('train/RawCupMask', image, iter_num)

                        image = np.array(disc_contour[0, 0:1, :, :].data.cpu().numpy())
                        writer.add_image('train/disc_contour', image, iter_num)

                        image = np.array(disc_bg[0, 0:1, :, :].data.cpu().numpy())
                        writer.add_image('train/disc_bg', image, iter_num)

                        image = np.array(cup_contour[0, 0:1, :, :].data.cpu().numpy())
                        writer.add_image('train/cup_contour', image, iter_num)

                        image = np.array(cup_bg[0, 0:1, :, :].data.cpu().numpy())
                        writer.add_image('train/cup_bg', image, iter_num)

            torch.cuda.empty_cache()

        if anatomy_epoch_metrics:
            anatomy_cosines = np.asarray(
                [item['cosine'] for item in anatomy_epoch_metrics],
                dtype=np.float64,
            )
            anatomy_gates = np.asarray(
                [item['gate'] for item in anatomy_epoch_metrics],
                dtype=np.float64,
            )
            anatomy_main_alignments = np.asarray(
                [item.get('main_alignment', 0.0) for item in anatomy_epoch_metrics],
                dtype=np.float64,
            )
            anatomy_main_gates = np.asarray(
                [item.get('main_gate', 1.0) for item in anatomy_epoch_metrics],
                dtype=np.float64,
            )
            anatomy_conflict_fraction = float(np.mean([
                item['conflict'] for item in anatomy_epoch_metrics
            ]))
            writer.add_scalar(
                'anatomy_epoch/cosine_mean', anatomy_cosines.mean(), epoch_num
            )
            writer.add_scalar(
                'anatomy_epoch/cosine_min', anatomy_cosines.min(), epoch_num
            )
            writer.add_scalar(
                'anatomy_epoch/conflict_fraction',
                anatomy_conflict_fraction,
                epoch_num,
            )
            writer.add_scalar(
                'anatomy_epoch/gate_mean', anatomy_gates.mean(), epoch_num
            )
            writer.add_scalar(
                'anatomy_epoch/main_alignment_mean',
                anatomy_main_alignments.mean(),
                epoch_num,
            )
            writer.add_scalar(
                'anatomy_epoch/main_gate_mean',
                anatomy_main_gates.mean(),
                epoch_num,
            )
            logging.info(
                'Anatomy gate epoch %d: cosine_mean=%.6f cosine_min=%.6f '
                'conflict_fraction=%.6f gate_mean=%.6f main_align=%.6f main_gate=%.6f',
                epoch_num,
                anatomy_cosines.mean(),
                anatomy_cosines.min(),
                anatomy_conflict_fraction,
                anatomy_gates.mean(),
                anatomy_main_alignments.mean(),
                anatomy_main_gates.mean(),
            )

        client_adapter_states = None
        if args.use_adapter:
            client_adapter_states = {
                client_idx: clone_adapter_state(net_clients[client_idx])
                for client_idx in source_site_idx
            }

        active_client_weight = client_weight.copy()
        grcfs_summary = None
        if args.grcfs_aggregation:
            if epoch_num >= int(args.grcfs_warmup):
                effective_grcfs_gamma = _effective_grcfs_gamma(args, epoch_num)
                if args.grcfs_probe_mode == 'self':
                    grcfs_client_metrics = OrderedDict()
                    for client_idx in source_site_idx:
                        device = next(net_clients[client_idx].parameters()).device
                        metrics = estimate_grcfs_client_metrics(
                            local_model=net_clients[client_idx],
                            global_model=net_clients[unseen_site_idx],
                            dataloader=dataloader_clients[client_idx],
                            args=args,
                            device=device,
                        )
                        if metrics is None:
                            logging.warning(
                                'GR-CFS self-probe produced no metrics for client %d at epoch %d; falling back to FedAvg weights.',
                                client_idx,
                                epoch_num,
                            )
                            grcfs_client_metrics = None
                            break
                        grcfs_client_metrics[client_idx] = metrics

                    if grcfs_client_metrics is not None:
                        active_client_weight, grcfs_weight_ema, grcfs_summary = \
                            build_grcfs_aggregation_weights(
                                grcfs_client_metrics,
                                client_weight,
                                source_site_idx,
                                grcfs_weight_ema,
                                args,
                                effective_gamma=effective_grcfs_gamma,
                            )
                else:
                    target_clients = _select_cross_probe_targets(
                        source_site_idx,
                        epoch_num,
                        args.grcfs_cross_targets_per_round,
                    )
                    global_metrics = OrderedDict()
                    pair_metrics = OrderedDict()
                    probe_batches_by_target = OrderedDict()
                    for target_idx in target_clients:
                        probe_batches = collect_grcfs_probe_batches(
                            dataloader_clients[target_idx],
                            args,
                        )
                        if not probe_batches:
                            logging.warning(
                                'GR-CFS collected no probe batches for target client %d at epoch %d; falling back to FedAvg weights.',
                                target_idx,
                                epoch_num,
                            )
                            probe_batches_by_target = None
                            break
                        probe_batches_by_target[target_idx] = probe_batches
                    need_global_probe = args.grcfs_cross_demand_mode != 'client_score'
                    if probe_batches_by_target is not None and need_global_probe:
                        global_device = next(net_clients[unseen_site_idx].parameters()).device
                        for target_idx in target_clients:
                            metrics = estimate_grcfs_probe_metrics_on_batches(
                                net_clients[unseen_site_idx],
                                probe_batches_by_target[target_idx],
                                args,
                                global_device,
                            )
                            if metrics is None:
                                logging.warning(
                                    'GR-CFS global cross-probe produced no metrics for target client %d at epoch %d; falling back to FedAvg weights.',
                                    target_idx,
                                    epoch_num,
                                )
                                global_metrics = None
                                break
                            global_metrics[target_idx] = metrics
                    elif probe_batches_by_target is None:
                        global_metrics = None
                    else:
                        global_metrics = OrderedDict()

                    if probe_batches_by_target is not None and global_metrics is not None:
                        for target_idx in target_clients:
                            for model_idx in source_site_idx:
                                if model_idx == target_idx:
                                    continue
                                device = next(net_clients[model_idx].parameters()).device
                                metrics = estimate_grcfs_probe_metrics_on_batches(
                                    net_clients[model_idx],
                                    probe_batches_by_target[target_idx],
                                    args,
                                    device,
                                )
                                if metrics is None:
                                    logging.warning(
                                        'GR-CFS cross-probe produced no metrics for model client %d on target client %d at epoch %d; falling back to FedAvg weights.',
                                        model_idx,
                                        target_idx,
                                        epoch_num,
                                    )
                                    pair_metrics = None
                                    break
                                pair_metrics[(model_idx, target_idx)] = metrics
                            if pair_metrics is None:
                                break

                    if global_metrics is not None and pair_metrics is not None:
                        active_client_weight, grcfs_weight_ema, grcfs_summary = \
                            build_cross_grcfs_aggregation_weights(
                                pair_metrics,
                                global_metrics,
                                client_weight,
                                source_site_idx,
                                target_clients,
                                grcfs_weight_ema,
                                args,
                                effective_grcfs_gamma,
                            )

                if grcfs_summary is not None:
                    grcfs_row = OrderedDict([('epoch', epoch_num)])
                    grcfs_row.update(grcfs_summary)
                    write_header = not os.path.exists(grcfs_diagnostic_path)
                    with open(grcfs_diagnostic_path, 'a', newline='') as handle:
                        writer_csv = csv.DictWriter(
                            handle,
                            fieldnames=list(grcfs_row.keys()),
                        )
                        if write_header:
                            writer_csv.writeheader()
                        writer_csv.writerow(grcfs_row)

                    for key, value in grcfs_summary.items():
                        if isinstance(value, (int, float, np.integer, np.floating)):
                            value = float(value)
                            if np.isfinite(value):
                                writer.add_scalar('grcfs/{}'.format(key), value, epoch_num)
                    logging.info(
                        'GR-CFS epoch %d mode=%s status=%s gamma=%.6f targets=%s weights=%s scores=%s',
                        epoch_num,
                        grcfs_summary.get('probe_mode', args.grcfs_probe_mode),
                        grcfs_summary.get('cross_status', 'active'),
                        grcfs_summary.get('anchor_gamma', effective_grcfs_gamma),
                        grcfs_summary.get('target_clients', 'self'),
                        {
                            client_idx: grcfs_summary['weight_client{}'.format(client_idx)]
                            for client_idx in source_site_idx
                        },
                        {
                            client_idx: grcfs_summary.get('score_client{}'.format(client_idx), 0.0)
                            for client_idx in source_site_idx
                        },
                    )
            else:
                logging.info(
                    'GR-CFS warmup epoch %d/%d: using ordinary FedAvg weights.',
                    epoch_num,
                    args.grcfs_warmup,
                )

        server_output_delta = None
        server_metrics = None
        if server_reference is not None:
            server_output_delta, server_conflict_ema, server_metrics = \
                dynamic_head_consensus(
                    client_models={
                        client_idx: net_clients[client_idx]
                        for client_idx in source_site_idx
                    },
                    client_weights={
                        client_idx: float(active_client_weight[client_idx])
                        for client_idx in source_site_idx
                    },
                    reference=server_reference,
                    previous_ema=server_conflict_ema,
                    ema_momentum=args.server_conflict_ema_momentum,
                    conflict_threshold=args.server_conflict_threshold,
                    gate_temperature=args.server_gate_temperature,
                    consensus_beta=args.server_consensus_beta,
                    target=args.server_consensus_target,
                    align_fedavg=bool(args.server_consensus_align_fedavg),
                    norm_cap=args.server_consensus_norm_cap,
                )

        ## model aggregation
        skip_prefixes = ('adapter.',) if args.use_adapter else ()
        update_global_model(net_clients, active_client_weight, skip_prefixes=skip_prefixes)
        if server_output_delta is not None:
            apply_prefixed_delta(
                net_clients,
                server_reference,
                server_output_delta,
            )
            writer.add_scalar(
                'server/conflict_score', server_metrics['conflict_score'], epoch_num
            )
            writer.add_scalar(
                'server/conflict_ema', server_metrics['conflict_ema'], epoch_num
            )
            writer.add_scalar(
                'server/activation', server_metrics['activation'], epoch_num
            )
            writer.add_scalar(
                'server/effective_beta', server_metrics['effective_beta'], epoch_num
            )
            writer.add_scalar(
                'server/pairwise_min', server_metrics['pairwise_min'], epoch_num
            )
            writer.add_scalar(
                'server/cancellation_ratio',
                server_metrics['cancellation_ratio'],
                epoch_num,
            )
            writer.add_scalar(
                'server/alignment_gate',
                server_metrics['alignment_gate'],
                epoch_num,
            )
            writer.add_scalar(
                'server/correction_alignment',
                server_metrics['correction_alignment'],
                epoch_num,
            )
            writer.add_scalar(
                'server/norm_cap_ratio',
                server_metrics['norm_cap_ratio'],
                epoch_num,
            )
            server_row = {
                'epoch': epoch_num,
                'target': server_metrics['target'],
                'pairwise_min': server_metrics['pairwise_min'],
                'pairwise_mean': server_metrics['pairwise_mean'],
                'pairwise_variance': server_metrics['pairwise_variance'],
                'negative_pair_fraction': server_metrics['negative_pair_fraction'],
                'conflict_score': server_metrics['conflict_score'],
                'conflict_ema': server_metrics['conflict_ema'],
                'activation': server_metrics['activation'],
                'effective_beta': server_metrics['effective_beta'],
                'alignment_gate': server_metrics['alignment_gate'],
                'correction_alignment': server_metrics['correction_alignment'],
                'norm_cap_ratio': server_metrics['norm_cap_ratio'],
                'cancellation_ratio': server_metrics['cancellation_ratio'],
            }
            server_row.update({
                'cosine_{}'.format(name): value
                for name, value in server_metrics['pairwise_cosine'].items()
            })
            server_row.update({
                'delta_norm_client{}'.format(client_id): value
                for client_id, value in server_metrics['delta_norms'].items()
            })
            write_header = not os.path.exists(server_diagnostic_path)
            with open(server_diagnostic_path, 'a', newline='') as handle:
                writer_csv = csv.DictWriter(
                    handle, fieldnames=list(server_row.keys())
                )
                if write_header:
                    writer_csv.writeheader()
                writer_csv.writerow(server_row)
            logging.info(
                'Server consensus epoch %d: score=%.6f ema=%.6f '
                'activation=%.6f beta=%.6f pair_min=%.6f cancel=%.6f '
                'align_gate=%.6f norm_cap=%.6f target=%s',
                epoch_num,
                server_metrics['conflict_score'],
                server_metrics['conflict_ema'],
                server_metrics['activation'],
                server_metrics['effective_beta'],
                server_metrics['pairwise_min'],
                server_metrics['cancellation_ratio'],
                server_metrics['alignment_gate'],
                server_metrics['norm_cap_ratio'],
                server_metrics['target'],
            )
        if args.use_adapter:
            load_adapter_state(net_clients[unseen_site_idx], neutral_adapter_state)

        router = None
        avg_adapter_state = None
        if use_routing_eval:
            # Rebuild style memory only when EMA frequency prototypes are used
            if use_ema_freq:
                device = next(net_clients[unseen_site_idx].parameters()).device
                style_memory_cache = compute_style_memory(
                    args.data_root,
                    client_name,
                    source_site_idx,
                    low_freq_ratio=args.memory_low_freq_ratio,
                    freq_proto_clients=freq_proto_clients,
                )

            memory_payload = build_memory_payload(style_memory_cache, content_memory_cache, args)
            if memory_payload is not None:
                router = DualMemoryRouter.from_checkpoint({'memory': memory_payload}, device='cpu')

        if effective_eval_adapter == 'average' and args.use_adapter and client_adapter_states:
            avg_weights = {client_id: 1.0 for client_id in client_adapter_states}
            avg_adapter_state = weighted_adapter_state(client_adapter_states, avg_weights)

        ## evaluation
        with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:
            dice_list = []
            haus_list = []
            print("epoch {} testing , site {}".format(epoch_num, unseen_site_idx), file=f)
            dice, dice_array, haus, haus_array = test(
                unseen_site_idx,
                net_clients[unseen_site_idx],
                eval_adapter=effective_eval_adapter,
                router=router,
                adapter_states=client_adapter_states,
                avg_adapter_state=avg_adapter_state,
                neutral_adapter_state=neutral_adapter_state,
                content_encoder=content_encoder,
            )
            od = float(dice[0])
            oc = float(dice[1])
            avg = (od + oc) / 2.0

            if od > best_od:
                best_od = od
                best_od_epoch = epoch_num

            if oc > best_oc:
                best_oc = oc
                best_oc_epoch = epoch_num

            if avg > best_avg:
                best_avg = avg
                best_avg_epoch = epoch_num
                best_avg_od = od
                best_avg_oc = oc

            print(("   OD dice is: {}, std is {}".format(dice[0], np.std(dice_array[:, 0]))), file=f)
            print(("   OC dice is: {}, std is {}".format(dice[1], np.std(dice_array[:, 1]))), file=f)

        ## save model
        save_mode_path = os.path.join(snapshot_path + 'model', 'epoch_' + str(epoch_num) + '.pth')
        if args.use_adapter:
            checkpoint_payload = {
                'state_dict': net_clients[unseen_site_idx].state_dict(),
                'client_adapter_states': client_adapter_states or {},
                'source_site_idx': list(source_site_idx),
                'client_names': list(client_name),
                'model_config': {
                    'use_adapter': bool(args.use_adapter),
                    'adapter_bottleneck': int(args.adapter_bottleneck),
                    'adapter_scale': float(args.adapter_scale),
                    'adapter_type': args.adapter_type,
                    'gate_bias': float(args.gate_bias),
                }
            }
            if use_routing_eval and memory_payload is not None:
                checkpoint_payload['memory'] = memory_payload
            torch.save(checkpoint_payload, save_mode_path)
        else:
            torch.save(net_clients[unseen_site_idx].state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    logging.info("==== Best Results Over All Epochs ====")
    logging.info("Best OD dice: %.6f at epoch %d", best_od, best_od_epoch)
    logging.info("Best OC dice: %.6f at epoch %d", best_oc, best_oc_epoch)
    logging.info("Best AVG dice: %.6f at epoch %d", best_avg, best_avg_epoch)
    logging.info(
        "Best AVG paired metrics: OD %.6f OC %.6f at epoch %d",
        best_avg_od,
        best_avg_oc,
        best_avg_epoch,
    )

    with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:
        print("\n==== Best Results Over All Epochs ====", file=f)
        print("Best OD dice: {:.6f} at epoch {}".format(best_od, best_od_epoch), file=f)
        print("Best OC dice: {:.6f} at epoch {}".format(best_oc, best_oc_epoch), file=f)
        print("Best AVG dice: {:.6f} at epoch {}".format(best_avg, best_avg_epoch), file=f)
        print(
            "Best AVG paired metrics: OD {:.6f} OC {:.6f} at epoch {}".format(
                best_avg_od, best_avg_oc, best_avg_epoch
            ),
            file=f,
        )


    writer.close()
