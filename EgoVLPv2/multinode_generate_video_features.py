import os
import argparse
import tqdm
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import data_loader.data_loader as module_data
import model.metric as module_metric
from model.model import FrozenInTime
from parse_config import ConfigParser

def setup_distributed():
    """ Initialize Distributed Process Group """
    dist.init_process_group(
        backend="nccl",  # Use NCCL for GPU communication
        init_method="env://",  # Read settings from environment variables
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)  # Set correct GPU for each process
    return local_rank

def run():
    local_rank = setup_distributed()  # Initialize distributed environment

    # Setup DataLoader with Distributed Sampler
    config._config['data_loader']['type'] = 'TextVideoDataLoader'
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['shuffle'] = False  # Disable shuffle for DistributedSampler

    data_loader = config.initialize('data_loader', module_data)
    sampler = torch.utils.data.distributed.DistributedSampler(data_loader.dataset)
    data_loader = torch.utils.data.DataLoader(data_loader.dataset, batch_size=args.batch_size, sampler=sampler)

    # Initialize Model
    model = config.initialize('arch', FrozenInTime)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)  # Use DDP

    model.eval()
    print(f"Rank {dist.get_rank()} - DataLoader Length: {len(data_loader)}")

    if not os.path.exists(args.save_dir) and dist.get_rank() == 0:
        os.mkdir(args.save_dir)

    num_frame = config.config['data_loader']['args']['video_params']['num_frames']
    dim = config.config['arch']['args']['projection_dim']

    with torch.no_grad():
        for i, data in enumerate(tqdm.tqdm(data_loader)):
            if os.path.exists(os.path.join(args.save_dir, data['meta']['clip_uid'][0]+'.pt')):
                print(f"{data['meta']['clip_uid']} already exists.")
                continue

            data['video'] = data['video'].to(local_rank)
            outs = torch.zeros(data['video'].shape[0], dim).to(local_rank)

            b_s = 64  # Adjust based on memory limits

            times = (data['video'].shape[0] + b_s - 1) // b_s

            for j in range(times):
                start, end = j * b_s, min((j+1) * b_s, data['video'].shape[0])
                data_batch = {'video': data['video'][start:end]}
                video_embeds = model(data=data_batch, n_embeds=None, v_embeds=None, allgather=None, n_gpu=None, args=None, config=None, loss_egonce=None, gpu=local_rank, task_names='Feature_Extraction')
                outs[start:end] = video_embeds

            if data['meta']['dataset'][0] == "LEMMA_video_NG":
                feature_rel_path = os.path.join(data['meta']['video_uid'][0], data['meta']['view_name'][0]) 
            else:
                feature_rel_path = data['meta']['video_uid'][0]

            if dist.get_rank() == 0:
                os.makedirs(os.path.join(args.save_dir, feature_rel_path), exist_ok=True)
                torch.save(outs.cpu(), os.path.join(args.save_dir, feature_rel_path, data['meta']['clip_uid'][0]+'.pt'))

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Distributed Training')

    args.add_argument('-r', '--resume', help='path to latest checkpoint')
    args.add_argument('-c', '--config', default=None, type=str, help='config file path')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int, help='Sliding window stride.')
    args.add_argument('--split', default='test', choices=['train', 'val', 'test'], help='Dataset split')
    args.add_argument('--batch_size', default=1, type=int, help='Batch size')
    args.add_argument('--save_dir', help='Path to save extracted features')
    args.add_argument('--cuda_base', help="CUDA device base (e.g., cuda:0)")

    config = ConfigParser(args, test=True, eval_mode='epic_vfg')
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride

    run()
