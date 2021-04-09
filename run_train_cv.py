from sklearn.model_selection import StratifiedKFold
import torch.nn as nn
import torch.optim as optim
import wandb
from common.data_loader import *
from common.trainer import *
from common.arg_parser import *
from common.utils import *
from common.model_loader import make_model
from common import define


def main():
    args = get_args()
    rank = int(os.getenv('OMPI_COMM_WORLD_RANK', '0'))

    train_set, val_set, test_set = load_cifar10(args.pretrained)
    cv = 5
    sk_fold = StratifiedKFold(n_splits=cv)

    for fold_idx, (train_indices, val_indices) in enumerate(sk_fold.split(train_set.data, train_set.targets)):
        print(f'### StratifiedKFold[{fold_idx + 1}/{cv}]')
        best_acc = 0
        run_name = f'{get_run_name(args)}_cv{fold_idx}'
        model_file_path = os.path.join(define.TRAINED_MODEL_DIR, f'{run_name}.pth')
        if args.wandb and rank == 0:
            run = wandb.init(project='myproject', entity='myaccount')
            wandb.run.name = run_name
            wandb.config.update(args)
        output_cuda_info()

        # create dataloader
        train_subset = torch.utils.data.Subset(train_set, train_indices)
        val_subset = torch.utils.data.Subset(val_set, val_indices)
        dataloaders = dict()
        dataloaders['train'] = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        dataloaders['validate'] = torch.utils.data.DataLoader(
            val_subset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        dataloaders['test'] = torch.utils.data.DataLoader(
            test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        output_dataset_info(dataloaders, show_stats=True)
        classes = get_classes(dataloaders['train'])

        # create model
        model = make_model(args.model, len(classes), pretrained=args.pretrained, fix_param=args.fixparam)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
        output_summary(model, dataloaders['train'])

        # settings for training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

        # start training
        print('# start training')
        for epoch in range(1, args.epoch + 1):
            print(f'Epoch[{epoch}/{args.epoch}]')
            # train and validation
            train_loss, train_acc, val_loss, val_acc = \
                train_val_epoch(model, dataloaders['train'], dataloaders['validate'], optimizer, criterion, device)

            # calculate accuracy of each class
            class_acc = test_each_class(model, dataloaders['validate'], device)
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
        print('# finish training')

        print('# test')
        test_loss, test_acc, class_acc = final_test(model, dataloaders['test'], criterion, device)
        print(f'test_loss:{test_loss}, test_acc:{test_acc: .3f}')
        for key, value in class_acc.items():
            print(f'{key} : {value: .3f}')
        print(f'cv_loss:{test_loss}, cv_acc:{test_acc: .3f}')
        if args.wandb and rank == 0:
            run.finish()
    print('# all finish')


if __name__ == '__main__':
    main()
