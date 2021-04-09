import torch.nn as nn
from torchvision.models import *
from models.unet import UNet
from models.sample_net import TestModel


def make_model(model_name, num_classes, pretrained=False, fix_param=False):
    if model_name == 'res18':
        model = resnet18(pretrained=pretrained)
        set_param_no_grad(model, fix_param)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'res34':
        model = resnet34(pretrained=pretrained)
        set_param_no_grad(model, fix_param)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'res50':
        model = resnet50(pretrained=pretrained)
        set_param_no_grad(model, fix_param)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'res101':
        model = resnet101(pretrained=pretrained)
        set_param_no_grad(model, fix_param)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'res152':
        model = resnet152(pretrained=pretrained)
        set_param_no_grad(model, fix_param)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'vgg11':
        model = vgg11(pretrained=pretrained)
        set_param_no_grad(model, fix_param)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg13':
        model = vgg13(pretrained=pretrained)
        set_param_no_grad(model, fix_param)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg16':
        model = vgg16(pretrained=pretrained)
        set_param_no_grad(model, fix_param)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'vgg19':
        model = vgg19(pretrained=pretrained)
        set_param_no_grad(model, fix_param)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == 'unet':
        model = UNet()
    else:
        model = TestModel()
    return model


def set_param_no_grad(model, fix_param):
    if fix_param:
        for param in model.parameters():
            param.requires_grad = False
