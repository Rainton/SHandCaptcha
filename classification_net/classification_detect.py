import classification_net.VGG_model as VGG_model
import classification_net.get_model as get_model
import torch

import cv2
import numpy as np
import os
from torch.autograd import Variable
import torch.nn.functional as F
use_model = "SMALL_NET"
# id_to_label = {0: "baoquan", 1: "chashouli", 2: "pray",
#                3: "fivea", 4: "fiveb", 5: "diss",
#                6: "finger0", 7: "palmdown", 8: "palmup",
#                9: "qheart", 10: "yeah", 11: "wuxiao",
#                12: "good", 13: "gun", 14: "raise",
#                15: "stop", 16: "circle"}
# id_to_label = {0: "good", 1: "pinky", 2: "bixin1",
#                3: "ok", 4: "666", 5: "antenna",
#                6: "rock", 7: "quantou", 8: "finger1",
#                9: "finger2", 10: "finger3", 11: "finger4",
#                12: "finger5"}
# id_to_label = {0: "good", 1: "bixin1",
#                2: "ok", 3: "666", 4: "rock",
#                5: "quantou", 6: "finger1", 7: "finger2",
#                8: "finger3", 9: "finger4", 10: "finger5",
#                11: "face", 12: "arm", 13: "hand"}
# id_to_label = {0: "good", 1: "bixin1", 2: "ok",
#                3: "666", 4: "rock", 5: "antenna",
#                6: "quantou", 7: "finger1", 8: "pinky",
#                9: "diss", 10: "finger2", 11: "finger3",
#                12: "finger4", 13: "finger5a", 14: "finger5b"}
id_to_label = {0: "good", 1: "bixin1", 2: "ok",
               3: "666", 4: "rock", 5: "rock",
               6: "quantou", 7: "finger1", 8: "finger1",
               9: "finger1", 10: "finger2", 11: "finger3",
               12: "finger4", 13: "finger5", 14: "finger5"}
model = get_model.get_model(use_model, 15, False)


def init_detection(model_path):
    print("loading from {}".format(model_path))
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    # model.eval()
    print("Done!")


def process_input(image):
    # print("0", image.shape)
    img_h, img_w, c = image.shape
    if img_h != 64 or img_w != 64:
        image = cv2.resize(image, (64, 64))
    image = image.transpose([2, 0, 1])
    # print("1", image.shape)
    return image


def detect(images):
    is_single_image = False
    if len(images.shape) == 3:
        images = [images]
    if len(images) == 1:
        is_single_image = True
    input_images = []
    for img in images:
        input_images.append(process_input(img))
    input_images = np.asarray(input_images)
    input_images = Variable(torch.from_numpy(input_images).float().cuda())
    input_images.cuda()
    pred = model.forward(input_images)
    pred = F.softmax(pred)
    pred = pred.data.cpu().numpy()
    if is_single_image:
        pred_label = np.argmax(pred)
        pred_label = np.asarray([pred_label])
        pred = np.asarray([pred])
    else:
        pred_label = np.argmax(pred, 1)
    pred_score = [pred[i][pred_label[i]] for i in range(len(pred_label))]
    pred_label = [id_to_label[pred_label[i]] for i in range(len(pred_label))]

    # print(pred_label, pred_score)
    return pred_label, pred_score


def get_label(path):
    res = path.split("/")[1].split("_")[1]
    return res


if __name__ == '__main__':
    root_path = "/home/fitz/Data/classification_data"
    eval_txt = open("/home/fitz/Data/classification_data/label_txt/eval_list.txt")
    test_lines = eval_txt.readlines()
    init_detection("model/RESNET18_e36_acc0.8700000047683716.pkl")
    # "/home/fitz/Data/classification_data/images/0_good"
    for img_name in test_lines:
        tmp_name = img_name.split("/")[-1].split(".")[0]
        img_path = os.path.join(root_path, img_name[:-1])
        tmp_img = cv2.imread(img_path)
        tmp_pred = detect(tmp_img)[0][0]
        tmp_label = get_label(img_name)
        if tmp_pred != tmp_label:
            print("pred:{} label:{}".format(tmp_pred, tmp_label))
            cv2.imwrite("error_imgs/"+tmp_name+"{}_{}.jpg".format(tmp_pred, tmp_label), tmp_img)
