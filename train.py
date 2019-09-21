from __future__ import print_function

from lib.ssds_train import train_model
from lib.utils.config_parse import cfg_from_file


def train():
    cfg_from_file('./experiments/cfgs_new/yolo_v3_small_inceptionv4_v4_8.15.yml')
    train_model()


if __name__ == '__main__':
    train()
