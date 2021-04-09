import torch.distributed as dist
import torch.multiprocessing as mp
import torch


def f(rank, world_size):
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    distributed_backend = 'nccl'
    dist.init_process_group(backend=distributed_backend,
                            init_method='tcp://127.0.0.1:23456', world_size=world_size, rank=rank)
    t = torch.rand(1).cuda()
    print(f'{rank}  {t}')
    # gather_t = [torch.ones_like(t)] * dist.get_world_size()  # NG
    gather_t = [torch.ones_like(t) for _ in range(world_size)]
    dist.all_gather(gather_t, t)
    if rank == 0:
        print(gather_t)


if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(f, nprocs=world_size, args=(world_size, ))
