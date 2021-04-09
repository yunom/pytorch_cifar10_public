import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='test_model')
    parser.add_argument('-p', '--pretrained', action='store_true')  # if true, use pretrained model
    # if true, fix param of middle layers(True:Transfer Learning, False:Fine-Tuning)
    parser.add_argument('-f', '--fixparam', action='store_true')
    parser.add_argument('-b', '--batch_size', default=128, type=int)
    parser.add_argument('-s', '--split', default=0, type=float)  # train-val split ratio (0=no split)
    parser.add_argument('-e', '--epoch', default=10, type=int)
    parser.add_argument('-w', '--wandb', action='store_true')  # if true, save to wandb
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--resume', action='store_true')  # if true, resume from saved model data
    return parser.parse_args()


def get_run_name(args):
    return f'{args.model}-b{args.batch_size}-lr{args.lr}-m{args.momentum}-ep{args.epoch}'
