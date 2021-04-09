import os
from datetime import datetime
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import wandb
from common.arg_parser import *
from common.data_loader import *
from common.model_loader import make_model
from common.trainer import *
from common.utils import *


def init_process(rank, world_size):
    distributed_backend = 'gloo'
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


def main_per_process(rank, world_size, args):
    init_process(rank, world_size)
    start_epoch = 1
    if args.wandb and rank == 0:
        run_name = get_run_name(args)
        wandb.init(project='myproject', entity='myaccount')
        wandb.run.name = run_name
        wandb.config.update(args)
    if rank == 0:
        output_cuda_info()

    # load dataset
    train_val_split = 0.2
    batch_size_per_proc = int(args.batch_size/world_size)
    train_set, val_set, test_set = load_cifar10(train_val_split, args.pretrained)

    # create sampler for ddp
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False)

    # create data loader for ddp
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size_per_proc, num_workers=args.num_workers,
        pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size_per_proc, num_workers=args.num_workers,
        pin_memory=True, sampler=val_sampler)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    # create ddp model
    model = make_model(args.model, 10, pretrained=args.pretrained, fix_param=args.fixparam)
    model = model.to(rank)
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    if rank == 0:
        output_summary(ddp_model, train_loader)

    # settings for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # synchronize
    dist.barrier()

    # start training
    print(f'[{datetime.now()}]#{rank}: start training')
    for epoch in range(start_epoch, start_epoch + args.epoch):
        if rank == 0:
            print(f'Epoch[{epoch}/{args.epoch}]')
        train_sampler.set_epoch(epoch)
        val_sampler.set_epoch(epoch)
        dist.barrier()  # synchronize

        # train and validate
        train_loss, train_acc = train_epoch(ddp_model, train_loader, optimizer, criterion, rank)
        val_loss, val_acc = validate_epoch(ddp_model, val_loader, criterion, rank)
        dist.barrier()  # synchronize

        # sharing loss and accuracy among all gpus(processes)
        train_loss_list = [0.] * world_size
        train_acc_list = [0.] * world_size
        val_loss_list = [0.] * world_size
        val_acc_list = [0.] * world_size
        dist.all_gather_object(train_loss_list, train_loss)
        dist.all_gather_object(train_acc_list, train_acc)
        dist.all_gather_object(val_loss_list, val_loss)
        dist.all_gather_object(val_acc_list, val_acc)

        # save data to wandb
        if args.wandb and rank == 0:
            avg_train_loss = sum(train_loss_list) / world_size
            avg_train_acc = sum(train_acc_list) / world_size
            avg_val_loss = sum(val_loss_list) / world_size
            avg_val_acc = sum(val_acc_list) / world_size
            wandb.log({'acc': avg_train_acc, 'loss': avg_train_loss,
                       'val_acc': avg_val_acc, 'val_loss': avg_val_loss,
                       'lr': scheduler.get_last_lr()[0]})
        scheduler.step()
    print(f'[{datetime.now()}]#{rank}: finished training')

    if rank == 0:
        print('# final test')
        test_loss, test_acc, class_acc = final_test(model, test_loader, criterion, rank)
        for key, value in class_acc.items():
            print(f'{key} : {value: .3f}')

        # save data to wandb
        if args.wandb:
            wandb.log({'test_acc': test_acc, 'test_loss': test_loss})
            wandb.finish()
        print('# all finished')


def main():
    # torch.cuda.is_available = lambda: False
    args = get_args()
    cpu_count = os.cpu_count()
    gpu_count = torch.cuda.device_count()
    print(f'cpu_count:{cpu_count} gpu_count:{gpu_count}')

    world_size = gpu_count
    args.num_workers = world_size * 2
    if args.num_workers > cpu_count:
        args.num_workers = cpu_count

    # create and run multiple processes(not threads)
    mp.spawn(main_per_process, args=(world_size, args),
             nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
