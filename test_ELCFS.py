import os
import argparse
import inspect
from glob import glob

import numpy as np
import torch

from networks.unet2d import Unet2D
from utils.util import _eval_dice, _eval_haus, _connectivity_region_analysis
from utils.memory_routing import (
    DualMemoryRouter,
    clone_adapter_state,
    embedding_to_content_query,
    load_adapter_state,
    low_freq_image_vector_torch,
    weighted_adapter_state,
)


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
parser.add_argument('--use_adapter', type=int, default=1,
                    help='fallback adapter setting for old checkpoints')
parser.add_argument('--adapter_bottleneck', type=int, default=8,
                    help='fallback adapter bottleneck for old checkpoints')
parser.add_argument('--adapter_scale', type=float, default=1.0,
                    help='fallback adapter scale for old checkpoints')
parser.add_argument(
    '--adapter_scale_sweep',
    type=str,
    default='',
    help='comma-separated inference-only adapter scales, e.g. 0,0.1,0.3,0.5,0.7,1.0'
)
parser.add_argument('--adapter_type', type=str, default='residual',
                    choices=['residual', 'spatial_gate'],
                    help='fallback adapter type for old checkpoints')
parser.add_argument('--memory_routing', type=int, default=1,
                    help='use checkpoint dual-memory routing when available')
parser.add_argument('--memory_low_freq_ratio', type=float, default=0.01,
                    help='low-frequency window ratio used by spectral query')
parser.add_argument('--save_predictions', type=int, default=1,
                    help='save per-image prediction arrays; use 0 for a metric-only scale sweep')
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


def _load_checkpoint(test_net_idx, device):
    ckpt_path = os.path.join(model_dir, f'epoch_{test_net_idx}.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        payload = checkpoint
        state_dict = checkpoint['state_dict']
        model_config = checkpoint.get('model_config', {})
    else:
        payload = {}
        state_dict = checkpoint
        model_config = {}

    model_config.setdefault('use_adapter', bool(args.use_adapter))
    model_config.setdefault('adapter_bottleneck', int(args.adapter_bottleneck))
    model_config.setdefault('adapter_scale', float(args.adapter_scale))
    model_config.setdefault('adapter_type', args.adapter_type)
    model_config.setdefault('gate_bias', float(getattr(args, 'gate_bias', -3.0)))

    # Checkpoints and model files may come from slightly different revisions.
    # Pass only constructor options supported by the currently imported Unet2D.
    supported_config = set(inspect.signature(Unet2D.__init__).parameters)
    supported_config.discard('self')
    ignored_config = sorted(set(model_config) - supported_config)
    if ignored_config:
        print('Ignoring unsupported model_config keys: {}'.format(', '.join(ignored_config)))
    model_config = {
        key: value for key, value in model_config.items()
        if key in supported_config
    }

    test_net = Unet2D(**model_config).to(device)
    test_net.load_state_dict(state_dict, strict=False)
    test_net.eval()
    return test_net, payload


def test(site_index, test_net_idx, adapter_scale_override=None, prediction_dir=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_net, payload = _load_checkpoint(test_net_idx, device)
    checkpoint_scale = None
    if getattr(test_net, 'adapter', None) is not None:
        checkpoint_scale = float(test_net.adapter.scale)
        if adapter_scale_override is not None:
            # Inference-only intervention: keep the checkpoint weights fixed and
            # change only the final residual multiplier.
            test_net.adapter.scale = float(adapter_scale_override)
    elif adapter_scale_override is not None:
        raise ValueError('adapter_scale_sweep requires a checkpoint with an adapter')

    router = DualMemoryRouter.from_checkpoint(payload, device='cpu') if args.memory_routing else None
    adapter_states = payload.get('client_adapter_states', {}) if isinstance(payload, dict) else {}
    neutral_adapter_state = clone_adapter_state(test_net)

    # Use ResNet-18 for content query (same encoder used to build content memory)
    content_encoder = None
    if router is not None:
        from treefeddg.features import ResNet18DomainEncoder
        content_encoder = ResNet18DomainEncoder(weights="imagenet").to(device)
        content_encoder.eval()

    test_data = client_data_list[site_index]
    if len(test_data) == 0:
        raise RuntimeError(f'No test files found for site index {site_index}: {client_names[site_index]}')

    dice_array = []
    haus_array = []
    residual_ratios = []

    adapter_hook = None
    if getattr(test_net, 'adapter', None) is not None:
        def _record_residual_ratio(module, inputs, output):
            adapter_input = inputs[0].detach()
            applied_residual = output.detach() - adapter_input
            input_norm = torch.linalg.vector_norm(adapter_input.flatten(1), dim=1)
            residual_norm = torch.linalg.vector_norm(applied_residual.flatten(1), dim=1)
            ratio = residual_norm / input_norm.clamp_min(1e-12)
            residual_ratios.extend(ratio.cpu().tolist())

        adapter_hook = test_net.adapter.register_forward_hook(_record_residual_ratio)

    with torch.no_grad():
        for filename in test_data:
            data = np.load(filename)  # H, W, 5 (RGB + disc + cup)
            image = data[..., :3]
            mask = data[..., 3:]  # H, W, 2
            mask = np.expand_dims(mask.transpose(2, 0, 1), axis=0)  # 1,2,H,W

            image_test = np.expand_dims(image.transpose(2, 0, 1), axis=0)  # 1,3,H,W
            image_test = torch.from_numpy(image_test).float().to(device)

            if neutral_adapter_state is not None:
                load_adapter_state(test_net, neutral_adapter_state, device=device)

            if router is not None and adapter_states:
                style_query = low_freq_image_vector_torch(image_test, args.memory_low_freq_ratio)
                content_query = content_encoder(image_test) if content_encoder is not None else None
                route_weights = router.route(style_query, content_query)
                routed_adapter = weighted_adapter_state(adapter_states, route_weights, device=device)
                if routed_adapter is not None:
                    load_adapter_state(test_net, routed_adapter, device=device)

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

            if args.save_predictions:
                output_dir = prediction_dir or result_dir
                os.makedirs(output_dir, exist_ok=True)
                base = os.path.splitext(os.path.basename(filename))[0]
                out_name = f"{site_index}_{base}"
                _save_image(image.transpose(2, 0, 1), mask[0], pred_y[0], output_dir, out_name)

    if adapter_hook is not None:
        adapter_hook.remove()
    load_adapter_state(test_net, neutral_adapter_state, device=device)

    dice_array = np.array(dice_array)
    haus_array = np.array(haus_array)

    dice_avg = np.mean(dice_array, axis=0).tolist()
    haus_avg = np.mean(haus_array, axis=0).tolist()
    residual_ratio = float(np.mean(residual_ratios)) if residual_ratios else None
    return dice_avg, dice_array, haus_avg, haus_array, checkpoint_scale, residual_ratio


def _parse_adapter_scale_sweep(value):
    if not value.strip():
        return [None]

    scales = []
    for item in value.split(','):
        item = item.strip()
        if not item:
            continue
        scale = float(item)
        if scale < 0.0:
            raise ValueError('adapter inference scales must be non-negative')
        scales.append(scale)
    if not scales:
        raise ValueError('adapter_scale_sweep did not contain any valid scales')
    return scales


if __name__ == '__main__':
    test_net_idx = args.model_idx
    unseen_site_idx = args.unseen_site

    test_result_path = os.path.join(
        test_output_root,
        f'testing_result_unseensite_{unseen_site_idx}.txt'
    )

    scales = _parse_adapter_scale_sweep(args.adapter_scale_sweep)
    with open(test_result_path, 'a') as f:
        print(f"epoch {test_net_idx} testing")
        print(
            f"\nmodel_dir={model_dir} epoch={test_net_idx} inference-scale sweep",
            file=f,
        )

        for scale in scales:
            scale_label = 'checkpoint' if scale is None else f'{scale:g}'
            prediction_dir = result_dir
            if len(scales) > 1 and args.save_predictions:
                prediction_dir = os.path.join(result_dir, f'scale_{scale_label}')

            dice, dice_array, haus, haus_array, checkpoint_scale, residual_ratio = test(
                unseen_site_idx,
                test_net_idx,
                adapter_scale_override=scale,
                prediction_dir=prediction_dir,
            )
            applied_scale = checkpoint_scale if scale is None else scale
            avg_dice = (dice[0] + dice[1]) / 2.0
            scale_text = 'none' if applied_scale is None else f'{applied_scale:g}'
            checkpoint_scale_text = (
                'none' if checkpoint_scale is None else f'{checkpoint_scale:g}'
            )
            residual_ratio_text = (
                'none' if residual_ratio is None else f'{residual_ratio:.6f}'
            )
            summary = (
                f"checkpoint_scale={checkpoint_scale_text} eval_scale={scale_text} "
                f"OD={dice[0]:.6f} "
                f"OC={dice[1]:.6f} AVG={avg_dice:.6f} "
                f"residual_ratio={residual_ratio_text}"
            )
            print(summary)
            print(summary, file=f)
            print(
                f"   OD std={np.std(dice_array[:, 0]):.6f}, array={dice_array[:, 0]}",
                file=f
            )
            print(
                f"   OC std={np.std(dice_array[:, 1]):.6f}, array={dice_array[:, 1]}",
                file=f
            )
            print(
                f"   OD haus={haus[0]}, std={np.std(haus_array[:, 0])}",
                file=f
            )
            print(
                f"   OC haus={haus[1]}, std={np.std(haus_array[:, 1])}",
                file=f
            )
