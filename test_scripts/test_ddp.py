import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def init_process(rank, world_size):
    distributed_backend = 'mpi'
    if torch.cuda.is_available():
        device = rank % torch.cuda.device_count()
        torch.cuda.set_device(device)
        distributed_backend = 'nccl'

    master_ip = os.getenv('MASTER_ADDR', 'localhost')
    master_port = os.getenv('MASTER_PORT', '6000')
    init_method = 'tcp://' + master_ip + ':' + master_port

    dist.init_process_group(backend=distributed_backend,
                            rank=rank,
                            world_size=world_size,
                            init_method=init_method)


def main_per_process(rank, world_size):
    init_process(rank, world_size)

    # create input and label data
    inputs = torch.randn(1, 2).to(rank)
    labels = torch.randn(1, 1).to(rank)

    # create local model
    model = nn.Linear(2, 1)
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    print(f'rank:{rank}/{world_size} bias:{ddp_model.module.bias.data.cpu().numpy()} '
          f'weight:{ddp_model.module.weight.data.cpu().numpy()} inputs:{inputs.data.cpu().numpy()}')
    dist.barrier()

    # define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    max_epoch = 2
    for epoch in range(1, max_epoch + 1):
        if rank == 0:
            print(f'Epoch[{epoch}/{max_epoch}]')
        dist.barrier()

        # forward and backward
        outputs = ddp_model(inputs)  # (1x2) x (2x1) = (1x1)
        # calc_outputs_1 = inputs.data[0][0] * ddp_model.module.weight.data[0][0]
        # calc_outputs_2 = inputs.data[0][1] * ddp_model.module.weight.data[0][1]
        # calc_outputs = calc_outputs_1 + calc_outputs_2 + ddp_model.module.bias.data
        print(f'rank{rank}_a bias:{ddp_model.module.bias.data.cpu().numpy()} '
              f'weight:{ddp_model.module.weight.data.cpu().numpy()} outputs:{outputs.data.cpu().numpy()} '
              # f'caloutputs:{calc_outputs.data.cpu().numpy()}'
              )
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'rank{rank}_b bias:{ddp_model.module.bias.data.cpu().numpy()} '
              f'weight:{ddp_model.module.weight.data.cpu().numpy()} loss:{loss}')

        # get loss from all gpu
        loss_list = [torch.ones_like(loss) for _ in range(world_size)]
        dist.all_gather(loss_list, loss)
        if rank == 0:
            print(loss_list)

        dist.barrier()


def main():
    world_size = torch.cuda.device_count()
    # world_size = 1
    mp.spawn(main_per_process, args=(world_size,),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
