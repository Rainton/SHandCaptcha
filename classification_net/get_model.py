import classification_net.model_store as model_store
from torchvision import models
from classification_net.mobile_net_v2 import mobilenetV2
from classification_net.small_model import Small_Net
import torch
import collections
use_model_dict = {"RESNET18": model_store.resnet18,
                  "RESNET34": model_store.resnet34,
                  "RESNET50": model_store.resnet50,
                  "RESNET101": model_store.resnet101,
                  "RESNET152": model_store.resnet152,
                  "MOBILENETV2": mobilenetV2,
                  "SMALL_NET": Small_Net}

model_weight_path_dict = {"RESNET18": "pre_train_model/resnet18.pth",
                          "RESNET34": "pre_train_model/resnet34.pth",
                          "RESNET50": "pre_train_model/resnet50.pth",
                          "RESNET101": "pre_train_model/resnet101.pth",
                          "RESNET152": "pre_train_model/resnet152.pth",
                          "MOBILENETV2": "pre_train_model/mobilenetv2.pth",
                          "SMALL_NET": ""}

ignore_weights_dict = {"RESNET18": ['fc.weight', 'fc.bias'],
                       "RESNET34": ['fc.weight', 'fc.bias'],
                       "RESNET50": ['fc.weight', 'fc.bias'],
                       "RESNET101": ['fc.weight', 'fc.bias'],
                       "RESNET152": ['fc.weight', 'fc.bias'],
                       "MOBILENETV2": ['classifier.1.weight', 'classifier.1.bias'],
                       "SMALL_NET": [''],
                       }


def get_model(use_model, class_num, pretrained=True):
    net = use_model_dict[use_model](pretrained=False, num_classes=class_num)
    model_weight_path = model_weight_path_dict[use_model]
    ignore_weights = ignore_weights_dict[use_model]
    version_convert = True
    if pretrained:
        init_weights = net.state_dict()
        model_weights = torch.load(model_weight_path)
        new_weights = collections.OrderedDict()
        # print(init_weights.keys())
        # print(model_weights.keys())
        for key in init_weights.keys():
            if key in ignore_weights:
                new_weights[key] = init_weights[key]
            else:
                if version_convert:
                    new_weights[key] = model_weights["module."+key]
                else:
                    new_weights[key] = model_weights[key]
        net.load_state_dict(new_weights)
    return net
