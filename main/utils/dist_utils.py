import os
import torch.distributed as dist


def setup_distributed(backend='nccl', port='12355'):
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    os.environ['MASTER_ADDR'] = 'localhost'  # Adjust as necessary
    os.environ['MASTER_PORT'] = port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    
def cleanup():
    dist.destroy_process_group()
