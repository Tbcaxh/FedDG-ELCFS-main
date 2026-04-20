import os
import argparse
from glob import glob

import numpy as np
import torch

from networks.unet2d import Unet2D
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis


parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='xxxx', help='experiment name used in training output dir')
parser.add_argument('--batch_size', type=int, default=4, help='batch size (kept for compatibility)')
parser.add_argument('--client_num', type=int, default=4, help='number of clients')
parser.add_argument('--gpu', type=str, default='2', help='GPU id(s), e.g. "0" or "0,1"')
parser.add_argument('--unseen_site', type=int, default=0, help='test site index')
parser.add_argument('--model_idx', type=int, default=99, help='checkpoint epoch index')
parser.add_argument(
    '--data_root',
    type=str,
    default='/home/users/chenchen/projects/FedDG-ELCFS-main/dataset',
    help='dataset root path (contains client0..client3/data_npy)'
)
parser.add_argument('--model_dir', type=str, default='./model',
                    help='directory containing epoch_xx.pth')
parser.add_argument('--result_dir', type=str, default='./prediction',
                    help='directory to save predictions')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

# Fixed test result root path
test_output_root = '/home/users/chenchen/projects/FedDG-ELCFS-main/output'
model_dir = args.model_dir
result_dir = args.result_dir
os.makedirs(result_dir, exist_ok=True)
os.makedirs(test_output_root, exist_ok=True)


# Keep naming consistent with your modified train_ELCFS.py
client_names = [f"client{i}" for i in range(args.client_num)]
client_data_list = []
for client_idx in range(args.client_num):
    files = glob(os.path.join(args.data_root, client_names[client_idx], 'data_npy', '*.npy'))
    files.sort()
    client_data_list.append(files)
    print(f"client{client_idx}: {len(files)} files")


def _save_image(img, gth, pred, out_folder, out_name):
    np.save(os.path.join(out_folder, out_name + '_img.npy'), img)
    np.save(os.path.join(out_folder, out_name + '_pred.npy'), pred)
    np.save(os.path.join(out_folder, out_name + '_gth.npy'), gth)


def test(site_index, test_net_idx):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_net = Unet2D().to(device)
    ckpt_path = os.path.join(model_dir, f'epoch_{test_net_idx}.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    state_dict = torch.load(ckpt_path, map_location=device)
    test_net.load_state_dict(state_dict)
    test_net.eval()

    test_data = client_data_list[site_index]
    if len(test_data) == 0:
        raise RuntimeError(f'No test files found for site index {site_index}: {client_names[site_index]}')

    dice_array = []
    haus_array = []

    with torch.no_grad():
        for filename in test_data:
            data = np.load(filename)  # H, W, 5 (RGB + disc + cup)
            image = data[..., :3]
            mask = data[..., 3:]  # H, W, 2
            mask = np.expand_dims(mask.transpose(2, 0, 1), axis=0)  # 1,2,H,W

            image_test = np.expand_dims(image.transpose(2, 0, 1), axis=0)  # 1,3,H,W
            image_test = torch.from_numpy(image_test).float().to(device)

            _, pred, _ = test_net(image_test)
            pred_y = pred.detach().cpu().numpy()
            pred_y[pred_y > 0.75] = 1
            pred_y[pred_y <= 0.75] = 0

            pred_y_0 = pred_y[:, 0:1, ...]
            pred_y_1 = pred_y[:, 1:2, ...]
            processed_pred_y_0 = _connectivity_region_analysis(pred_y_0)
            processed_pred_y_1 = _connectivity_region_analysis(pred_y_1)
            processed_pred_y = np.concatenate([processed_pred_y_0, processed_pred_y_1], axis=1)

            dice_subject = _eval_dice(mask, processed_pred_y)
            haus_subject = _eval_haus(processed_pred_y, mask)  # fixed arg order
            dice_array.append(dice_subject)
            haus_array.append(haus_subject)

            base = os.path.splitext(os.path.basename(filename))[0]
            out_name = f"{site_index}_{base}"
            _save_image(image.transpose(2, 0, 1), mask[0], pred_y[0], result_dir, out_name)

    dice_array = np.array(dice_array)
    haus_array = np.array(haus_array)

    dice_avg = np.mean(dice_array, axis=0).tolist()
    haus_avg = np.mean(haus_array, axis=0).tolist()
    return dice_avg, dice_array, haus_avg, haus_array


if __name__ == '__main__':
    test_net_idx = args.model_idx
    unseen_site_idx = args.unseen_site

    test_result_path = os.path.join(
        test_output_root,
        f'testing_result_unseensite_{unseen_site_idx}.txt'
    )

    with open(test_result_path, 'a') as f:
        print(f"epoch {test_net_idx} testing")
        print(f"epoch {test_net_idx} testing", file=f)
        dice, dice_array, haus, haus_array = test(unseen_site_idx, test_net_idx)

        print(
            f"   OD dice is: {dice[0]}, std is {np.std(dice_array[:, 0])}, array is {dice_array[:, 0]}",
            file=f
        )
        print(f"      {dice_array[:, 0]}", file=f)
        print(
            f"   OC dice is: {dice[1]}, std is {np.std(dice_array[:, 1])}, array is {dice_array[:, 1]}",
            file=f
        )
        print(f"      {dice_array[:, 1]}", file=f)

        # Optional: also write hausdorff summary
        print(
            f"   OD haus is: {haus[0]}, std is {np.std(haus_array[:, 0])}, array is {haus_array[:, 0]}",
            file=f
        )
        print(
            f"   OC haus is: {haus[1]}, std is {np.std(haus_array[:, 1])}, array is {haus_array[:, 1]}",
            file=f
        )

        print((dice[0] + dice[1]) / 2.0)
