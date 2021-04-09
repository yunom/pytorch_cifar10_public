import torch.nn as nn
import torch.optim as optim
import wandb
from common.data_loader import load_data, get_classes
from common.trainer import *
from common.arg_parser import *
from common.utils import *
from common.model_loader import make_model
from common import define


def main():
    best_acc = 0
    start_epoch = 1

    # load arguments
    args = get_args()
    run_name = get_run_name(args)

    # init wandb
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))
    if args.wandb and rank == 0:
        wandb.init(project='myproject', entity='myaccount')
        wandb.run.name = run_name
        wandb.config.update(args)
    output_cuda_info()

    # load data
    train_val_split_ratio = 0.2
    dataloaders = load_data(
        'cifar10', args.batch_size, args.num_workers, train_val_split_ratio, args.pretrained)
    train_loader = dataloaders['train']
    val_loader = dataloaders['validate']
    test_loader = dataloaders['test']
    classes = get_classes(train_loader)

    # create model
    model = make_model(args.model, len(classes), pretrained=args.pretrained, fix_param=args.fixparam)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
    output_summary(model, train_loader)

    # loading model data from saved data
    model_file_path = os.path.join(define.TRAINED_MODEL_DIR, f'{run_name}.pth')
    if args.resume:
        best_acc, start_epoch = load_model(model, model_file_path)

    # settings for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # start training
    print('# start training')
    for epoch in range(start_epoch, start_epoch + args.epoch):
        print(f'Epoch[{epoch}/{args.epoch}]')
        # train and validation
        train_loss, train_acc, val_loss, val_acc = \
            train_val_epoch(model, train_loader, val_loader, optimizer, criterion, device)

        # calculate accuracy of each class
        class_acc = test_each_class(model, val_loader, device)
        for key, value in class_acc.items():
            print(f'{key} : {value: .3f}')

        # save data to wandb
        if args.wandb and rank == 0:
            wandb.log(class_acc, commit=False)
            wandb.log({'acc': train_acc, 'loss': train_loss,
                       'val_acc': val_acc, 'val_loss': val_loss, 'lr': scheduler.get_last_lr()[0]})

        # save model
        if val_acc > best_acc:
            save_model(model, epoch, val_acc, model_file_path)
            best_acc = val_acc
        scheduler.step()
    print('# finished training')

    print('# final test')
    test_loss, test_acc, class_acc = final_test(model, test_loader, criterion, device)
    for key, value in class_acc.items():
        print(f'{key} : {value: .3f}')
    # save data to wandb
    if args.wandb and rank == 0:
        wandb.log({'test_acc': test_acc, 'test_loss': test_loss})
    print('# all finished')


if __name__ == '__main__':
    main()
