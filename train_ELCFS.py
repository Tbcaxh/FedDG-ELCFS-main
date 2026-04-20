import os
import sys
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
from dataloaders.fundus_dataloader import TorchDataset, ToTensor, FundusDataset

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str,  default='xxxx', help='model_name')
parser.add_argument('--max_epoch', type=int,  default=300, help='maximum epoch number to train')
parser.add_argument('--client_num', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--batch_size', type=int, default=5, help='batch_size per gpu')
parser.add_argument('--clip_value', type=float,  default=100, help='maximum epoch number to train')
parser.add_argument('--meta_step_size', type=float,  default=1e-3, help='maximum epoch number to train')
parser.add_argument('--base_lr', type=float,  default=0.001, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='1', help='GPU to use')
parser.add_argument('--display_freq', type=int, default=5, help='batch_size per gpu')
# parser 区新增
parser.add_argument('--data_root', type=str,
                    default='/home/users/chenchen/projects/FedDG-ELCFS-main/dataset',
                    help='dataset root path (contains client0..client3)')

parser.add_argument('--unseen_site', type=int, default=0, help='batch_size per gpu')
args = parser.parse_args()

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

slice_num = np.array([101, 159, 400, 400])
volume_size = [384, 384, 3]
unseen_site_idx = args.unseen_site
source_site_idx = [0, 1, 2, 3]
source_site_idx.remove(unseen_site_idx)
client_weight = slice_num[source_site_idx] / np.sum(slice_num[source_site_idx])
client_weight = np.round(client_weight, decimals=2)
client_weight[-1] = 1 - np.sum(client_weight[:2])
client_weight = np.insert(client_weight, unseen_site_idx, 0)
print(client_weight)
num_classes = 3

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

def update_global_model(net_clients, client_weight):
    """
    修复#12: FedAvg 聚合, 动态支持任意数量 client, 不再硬编码4个.
    """
    # 获取所有 client 的参数列表
    all_params = [list(net.parameters()) for net in net_clients]
    num_params = len(all_params[0])

    for p_idx in range(num_params):
        new_para = torch.zeros_like(all_params[0][p_idx].data)
        for c_idx in range(len(net_clients)):
            new_para.add_(all_params[c_idx][p_idx].data, alpha=float(client_weight[c_idx]))
        for c_idx in range(len(net_clients)):
            all_params[c_idx][p_idx].data.copy_(new_para)

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

def test(site_index, test_net):

    test_data_list = client_data_list[site_index]

    dice_array = []
    for fid, filename in enumerate(test_data_list):
        data = np.load(filename)
        image = np.expand_dims(data[..., :3].transpose(2, 0, 1), axis=0)
        mask = np.expand_dims(data[..., 3:].transpose(2, 0, 1), axis=0)
        image = torch.from_numpy(image).float().cuda()

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
    dice_array = np.array(dice_array)
    # print (dice_array.shape)
    dice_avg = np.mean(dice_array, axis=0).tolist()
    # print (dice_avg)
    # haus_avg = np.mean(haus_array, axis=0).tolist()[0]
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
    ignore=shutil.ignore_patterns('.git', '__pycache__', 'output')
)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    # define dataset, model, optimizer for each client
    # 修复#10: unseen client 只创建 model (用于聚合接收), 不创建训练 dataloader 和 optimizer
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    dataloader_clients = {}
    net_clients = []
    optimizer_clients = {}
    for client_idx in range(client_num):
        net = Unet2D()
        net = net.cuda()
        net_clients.append(net)

        if client_idx == unseen_site_idx:
            # unseen client 只需要模型用于聚合, 不需要 dataloader 和 optimizer
            continue

        freq_site_idx = source_site_idx.copy()
        freq_site_idx.remove(client_idx)
        dataset = FundusDataset(client_idx=client_idx, freq_site_idx=freq_site_idx,
                                split='train', data_root=args.data_root, transform=None)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn)
        optimizer = torch.optim.Adam(net.parameters(), lr=args.base_lr, betas=(0.9, 0.999))
        dataloader_clients[client_idx] = dataloader
        optimizer_clients[client_idx] = optimizer

    for name, param in  net_clients[0].named_parameters():
        print (name)

    temperature = 0.05
    cont_loss_func = losses.NTXentLoss(temperature)

    # start federated learning
    writer = SummaryWriter(snapshot_path+'/log')
    # best trackers
    best_od = -1.0
    best_od_epoch = -1

    best_oc = -1.0
    best_oc_epoch = -1

    best_avg = -1.0
    best_avg_epoch = -1

    lr_ = base_lr
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        for client_idx in source_site_idx:
            dataloader_current = dataloader_clients[client_idx]
            net_current = net_clients[client_idx]
            net_current.train()
            optimizer_current = optimizer_clients[client_idx]
            time1 = time.time()
            iter_num = 0

            for i_batch, sampled_batch in enumerate(dataloader_current):
                time2 = time.time()

                # obtain training data
                volume_batch, label_batch, disc_contour, disc_bg, cup_contour, cup_bg = sampled_batch['image'], sampled_batch['label'], \
                sampled_batch['disc_contour'], sampled_batch['disc_bg'], sampled_batch['cup_contour'], sampled_batch['cup_bg']

                # 论文: dataloader 为每个外部 client 生成 1 个变换图像
                # volume_batch shape: [B, 3 + (K-1)*3, H, W]
                # 第 0~2 通道: raw image, 之后每 3 通道一个变换图像
                volume_batch_raw_np = volume_batch[:, :3, ...]
                volume_batch_raw = volume_batch_raw_np.cuda()
                label_batch = label_batch.cuda()
                disc_contour, disc_bg, cup_contour, cup_bg = disc_contour.cuda(), disc_bg.cuda(), cup_contour.cuda(), cup_bg.cuda()

                # 动态提取所有 K-1 个变换图像
                n_transformed = (volume_batch.shape[1] - 3) // 3  # K-1
                volume_batch_trs_list = []
                volume_batch_trs_np_list = []
                for t_idx in range(n_transformed):
                    start_ch = 3 + t_idx * 3
                    trs_np = volume_batch[:, start_ch:start_ch+3, ...]
                    volume_batch_trs_np_list.append(trs_np)
                    volume_batch_trs_list.append(trs_np.cuda())

                # ========== Inner Loop (meta-train on raw data, 论文 Eq.4) ==========
                logits_inner, pred_inner, embedding_inner = net_current(volume_batch_raw)
                loss_inner = dice_loss(pred_inner, label_batch)
                grads = torch.autograd.grad(loss_inner, net_current.parameters(), retain_graph=True)

                fast_weights = OrderedDict((name, param - torch.mul(meta_step_size, torch.clamp(grad, 0-clip_value, clip_value))) for
                                                  ((name, param), grad) in
                                                  zip(net_current.named_parameters(), grads))

                # ========== Outer Loop (meta-test on ALL K-1 transformed images, 论文 Eq.8) ==========
                embedding_outer_list = []
                loss_outer_dice = 0.0
                for t_idx in range(n_transformed):
                    logits_out, pred_out, emb_out = net_current(volume_batch_trs_list[t_idx], fast_weights)
                    loss_outer_dice += dice_loss(pred_out, label_batch)
                    embedding_outer_list.append(emb_out)
                loss_outer_dice = loss_outer_dice / max(n_transformed, 1)

                # ========== Boundary-oriented Contrastive Loss (论文 Eq.6-7) ==========
                # 提取 inner 的 boundary/background features
                inner_disc_ct_em, inner_disc_bg_em, inner_cup_ct_em, inner_cup_bg_em = \
                    extract_contour_embedding([disc_contour, disc_bg, cup_contour, cup_bg], embedding_inner)

                # 收集所有 K 组特征: 1(inner) + K-1(outer)
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

                # 拼接为 2K 个 region-level features
                disc_ct_em = torch.cat(all_disc_ct, 0)
                disc_bg_em = torch.cat(all_disc_bg, 0)
                cup_ct_em = torch.cat(all_cup_ct, 0)
                cup_bg_em = torch.cat(all_cup_bg, 0)
                disc_em = torch.cat((disc_ct_em, disc_bg_em), 0)
                cup_em = torch.cat((cup_ct_em, cup_bg_em), 0)
                # label: boundary=1, background=0
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

                iter_num = iter_num + 1
                if iter_num % display_freq == 0:
                    writer.add_scalar('lr', lr_, iter_num)
                    writer.add_scalar('loss/inner', loss_inner, iter_num)
                    writer.add_scalar('loss/outer', loss_outer, iter_num)
                    writer.add_scalar('loss/total', total_loss, iter_num)
                    logging.info('Epoch: [%d] client [%d] iteration [%d / %d] : inner loss : %f outer dice loss : %f outer cont loss : %f outer loss : %f total loss : %f' % \
                        (epoch_num, client_idx, iter_num, len(dataloader_current), loss_inner.item(), loss_outer_dice.item(), cont_loss.item(), loss_outer.item(), total_loss.item()))

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


        ## model aggregation
        update_global_model(net_clients, client_weight)

        ## evaluation
        with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:
            dice_list = []
            haus_list = []
            print("epoch {} testing , site {}".format(epoch_num, unseen_site_idx), file=f)
            dice, dice_array, haus, haus_array = test(unseen_site_idx, net_clients[unseen_site_idx])
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

            print(("   OD dice is: {}, std is {}".format(dice[0], np.std(dice_array[:, 0]))), file=f)
            print(("   OC dice is: {}, std is {}".format(dice[1], np.std(dice_array[:, 1]))), file=f)
            
        ## save model
        save_mode_path = os.path.join(snapshot_path + 'model', 'epoch_' + str(epoch_num) + '.pth')
        torch.save(net_clients[0].state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))

    logging.info("==== Best Results Over All Epochs ====")
    logging.info("Best OD dice: %.6f at epoch %d", best_od, best_od_epoch)
    logging.info("Best OC dice: %.6f at epoch %d", best_oc, best_oc_epoch)
    logging.info("Best AVG dice: %.6f at epoch %d", best_avg, best_avg_epoch)

    with open(os.path.join(snapshot_path, 'evaluation_result.txt'), 'a') as f:
        print("\n==== Best Results Over All Epochs ====", file=f)
        print("Best OD dice: {:.6f} at epoch {}".format(best_od, best_od_epoch), file=f)
        print("Best OC dice: {:.6f} at epoch {}".format(best_oc, best_oc_epoch), file=f)
        print("Best AVG dice: {:.6f} at epoch {}".format(best_avg, best_avg_epoch), file=f)


    writer.close()

