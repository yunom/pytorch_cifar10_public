import os
import torch
from tqdm import tqdm
from common import define
from common.data_loader import get_classes


def _main_epoch(model, dataloader, optimizer, criterion, device, is_train=True, epoch=None, max_epoch=None):
    sum_total = 0
    sum_correct = 0
    sum_loss = 0

    pbar = tqdm(dataloader, unit='batch', bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
    desc = ''
    if epoch is not None and max_epoch is not None:
        desc = f'Epoch[{epoch}/{max_epoch}]_'
    if is_train:
        model.train()
        pbar.set_description(f'{desc}dev{device}_train')
    else:
        model.eval()
        pbar.set_description(f'{desc}dev{device}___val')

    for (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        batch_size = labels.size(0)
        sum_total += batch_size

        # calculate loss and backward
        if is_train:
            optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        if is_train:
            loss.backward()
            optimizer.step()

        # calculate current loss
        sum_loss += loss.item() * batch_size
        current_loss = sum_loss / sum_total

        # calculate current accuracy
        predicted = outputs.max(1)[1]
        sum_correct += predicted.eq(labels).sum().item()
        current_acc = sum_correct / sum_total

        # update progress bar
        pbar.set_postfix({'loss': current_loss, 'acc': f'{current_acc:.3f}({sum_correct}/{sum_total})'})
    return current_loss, current_acc


def train_epoch(model, train_loader, optimizer, criterion, device, epoch=None, max_epoch=None):
    loss, acc = _main_epoch(model, train_loader, optimizer, criterion, device, True, epoch, max_epoch)
    return loss, acc


def validate_epoch(model, val_loader, criterion, device, epoch=None, max_epoch=None):
    with torch.no_grad():
        loss, acc = _main_epoch(model, val_loader, None, criterion, device, False, epoch, max_epoch)
    return loss, acc


def train_val_epoch(model, train_loader, val_loader, optimizer, criterion, device, epoch=None, max_epoch=None):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch, max_epoch)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device, epoch, max_epoch)
    return train_loss, train_acc, val_loss, val_acc


def test_each_class(model, test_loader, device):
    classes = get_classes(test_loader)
    num_classes = len(classes)
    class_acc = {}
    class_correct = [0] * num_classes  # [0 for i in range(num_classes)]などより高速
    class_total = [0] * num_classes
    with torch.no_grad():
        for (inputs, labels) in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            c = predicted.eq(labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    for i in range(num_classes):
        class_acc[f'Accuracy of {classes[i]: >10s}'] = class_correct[i] / class_total[i]
    return class_acc


def final_test(model, test_loader, criterion, device):
    with torch.no_grad():
        loss, acc = _main_epoch(model, test_loader, None, criterion, device, False)
    class_acc = test_each_class(model, test_loader, device)
    return loss, acc, class_acc


def load_model(model, model_file_path):
    print('# load saved model')
    checkpoint = torch.load(model_file_path)
    model.load_state_dict(checkpoint['model'])
    acc = checkpoint['acc']
    epoch = checkpoint['epoch']
    return acc, epoch


def save_model(model, epoch, val_acc, model_file_path):
    print('# save model')
    state = {
        'model': model.state_dict(),
        'acc': val_acc,
        'epoch': epoch,
    }
    if not os.path.isdir(define.TRAINED_MODEL_DIR):
        os.mkdir(define.TRAINED_MODEL_DIR)
    torch.save(state, model_file_path)
