from utils.Metrics import DiceMetric, get_classwise_dice, get_soft_label, reshape_prediction_and_ground_truth
from core.networks import DeepLabv3_Plus
from torch.utils.data import DataLoader
from utils.LoadData_with_bg import Wsss_dataset, Wsss_test_dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import argparse
import torch
from PIL import Image
import numpy as np
import sys
import os
import time
import logging
from utils.utils import monte_augmentation
sys.path.append(os.getcwd())


class KDLoss(nn.Module):
    """
    Distilling the Knowledge in a Neural Network
    https://arxiv.org/pdf/1503.02531.pdf
    """

    def __init__(self, T):
        super(KDLoss, self).__init__()
        self.T = T

    def forward(self, out_s, out_t):
        loss = (
            F.kl_div(F.log_softmax(out_s / self.T, dim=1),
                     F.softmax(out_t / self.T, dim=1), reduction="batchmean")
            * self.T
            * self.T
        )
        return loss

def get_arguments():
    parser = argparse.ArgumentParser(description="Wsss pytorch implementation")
    parser.add_argument("--dataset_root", type=str,
                        default="", help="training images")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Train batch size")
    # optimizer SGD, lr = 5e-4 or 1e-3, wd = 5e-4
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--wd", type=float, default=5e-4)

    parser.add_argument("--gpu", nargs="+", type=int)
    parser.add_argument("--max_epoch", default=100, type=int)
    # default=50
    parser.add_argument("--save_folder", default="checkpoints2/train_seg")
    parser.add_argument("--checkpoint1", type=str, default="")
    parser.add_argument("--checkpoint2", type=str, default="")
    parser.add_argument("--checkpoint3", type=str, default="")

    # (320,256,256),(256,224,224)
    parser.add_argument("--image_resize", default=256, type=int)
    parser.add_argument("--image_crop", default=224, type=int)
    parser.add_argument("--test_image_size", default=224, type=int)
    parser.add_argument("--alpha", type=float,
                        default=0.2, help="Weight factor")
    parser.add_argument("--T", type=float, default=30,
                        help="Temperature for KD")
    return parser.parse_args()

def get_model(args):
    model = DeepLabv3_Plus("resnet50", use_group_norm=True)
    model = torch.nn.DataParallel(model, device_ids=args.gpu)
    param_groups = model.module.get_parameter_groups(None)
    optimizer = optim.SGD(
        [
            {"params": param_groups[0], "lr": args.lr},
            {"params": param_groups[1], "lr": 2 * args.lr},
            {"params": param_groups[2], "lr": 10 * args.lr},
            {"params": param_groups[3], "lr": 20 * args.lr},
        ],
        momentum=0.9,
        weight_decay=args.wd,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80])
    return model, optimizer, scheduler

def selection(outputs_main, outputs_aux1, outputs_aux2, mask):
    n = outputs_main.shape[0]
    loss_main = F.cross_entropy(
        outputs_main, mask, reduction='none').view(n, -1)
    hard_aux1 = torch.argmax(outputs_aux1, dim=1).view(n, -1)
    hard_aux2 = torch.argmax(outputs_aux2, dim=1).view(n, -1)
    loss_select = 0
    for i in range(n):
        aux1_sample = hard_aux1[i]
        aux2_sample = hard_aux2[i]
        loss_sample = loss_main[i]
        agree_aux = (aux1_sample == aux2_sample)
        disagree_aux = (aux1_sample != aux2_sample)
        loss_select += 2*torch.sum(loss_sample[agree_aux]) + \
            0.5*torch.sum(loss_sample[disagree_aux])
    return loss_select / (n*loss_main.shape[1])


def weight_loss(loss):
    n = loss.shape[0]
    loss = loss.view(n, -1)
    loss_weight = F.softmax(loss.clone().detach(), dim=1) / torch.mean(
        F.softmax(loss.clone().detach(), dim=1), dim=1, keepdim=True
    )
    loss = torch.sum(loss * loss_weight) / (n * loss.shape[1])
    return loss


def joint_optimization(outputs_main, outputs_aux1, outputs_aux2, mask, kd_weight, kd_T):
    kd_loss = KDLoss(T=kd_T)
    avg_aux = (outputs_aux1 + outputs_aux2) / 2

    L_kd = kd_loss(outputs_main.permute(0, 2, 3, 1).reshape(-1, 4),
                   avg_aux.permute(0, 2, 3, 1).reshape(-1, 4))
    L_ce = selection(outputs_main, outputs_aux1, outputs_aux2, mask)
    L = L_ce + kd_weight * L_kd
    return L


def train_tri(model1, model2, model3, optimizer1, optimizer2, optimizer3, train_dataloader, args, epoch):
    model1.train()
    model2.train()
    model3.train()
    train_loss_epoch = 0
    
    for img, mask, soft_mask, img_names in train_dataloader:
        n, c, h, w = img.size()
        img, mask = img.cuda(), mask.type(torch.LongTensor).cuda()
        soft_mask = soft_mask.to(torch.double).cuda()

        outputs1 = model1(img)
        outputs2 = model2(img)
        outputs3 = model3(img)

        loss1 = joint_optimization(outputs1, outputs2.detach(
        ), outputs3.detach(), mask, args.alpha, args.T)
        loss2 = joint_optimization(outputs2, outputs3.detach(
        ), outputs1.detach(), mask, args.alpha, args.T)
        loss3 = joint_optimization(outputs3, outputs1.detach(
        ), outputs2.detach(), mask, args.alpha, args.T)

        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()
        optimizer3.zero_grad()
        loss3.backward()
        optimizer3.step()

        train_loss_epoch += (loss1 + loss2 + loss3) / 3
    print(f"tri-kd, train_loss:{train_loss_epoch.item()/len(train_dataloader)}")

def validate(model1, model2, model3, valid_dataloader, verbose=False, Monte=False, save=False):
    Dice_Metric = DiceMetric(4)
    model1.eval()
    model2.eval()
    model3.eval()
    my_background_root = "/mnt/data1/dataset/WSSS4LUAD/2.validation/my_bg_mask_patch_256/"
    with torch.no_grad():
        for img1, mask, img_label, img_names in valid_dataloader:
            img1 = img1.cuda()
            ori_size = mask.shape[1:]
            H, W = ori_size[0], ori_size[1]

            # bg
            my_bg_mask = Image.open(my_background_root + img_names[0])
            my_bg_mask = np.array(my_bg_mask, np.uint8)

            img_path = valid_dataset.img_path + "/" + img_names[0]

            # monte carlo test
            if not Monte:
                pred1 = (model1(img1) + model2(img1) + model3(img1)) / 3
            else:
                pred1 = (
                    monte_augmentation(20, model1, img_path,
                                       H, W, args.test_image_size)
                    + monte_augmentation(20, model2, img_path,
                                         H, W, args.test_image_size)
                    + monte_augmentation(20, model3, img_path,
                                         H, W, args.test_image_size)
                ) / 3
            pred1 = F.interpolate(pred1, size=ori_size,
                                  mode="bilinear", align_corners=True)
            pred = pred1

            Dice_Metric.add_batch(pred, mask, my_bg_mask)
    if not verbose:
        return Dice_Metric.compute_dice(False, save)
    else:
        return Dice_Metric.compute_dice(True, save)

def validate_trainset(model1, model2, model3, valid_dataloader, verbose=False, Monte=False, save=False):
    Dice_Metric = DiceMetric(4)
    model1.eval()
    model2.eval()
    model3.eval()
    my_background_root = "/mnt/data1/dataset/WSSS4LUAD/1.training/gamma_crf_train/"
    with torch.no_grad():
        for img1, mask, soft_mask, img_names in valid_dataloader:
            img1 = img1.cuda()
            ori_size = mask.shape[1:]
            H, W = ori_size[0], ori_size[1]
            # bg
            my_bg_mask = Image.open(my_background_root + img_names[0]).resize([224,224], Image.Resampling.NEAREST)
            my_bg_mask = np.array(my_bg_mask, np.uint8)

            pred1 = (model1(img1) + model2(img1) + model3(img1)) / 3
            pred1 = F.interpolate(pred1, size=ori_size, mode="bilinear", align_corners=True)
            pred = pred1

            Dice_Metric.add_batch(pred, mask, my_bg_mask)
    if not verbose:
        return Dice_Metric.compute_dice(False, save)
    else:
        return Dice_Metric.compute_dice(True, save)

if __name__ == "__main__":
    start_time = time.time()
    args = get_arguments()
    logging.basicConfig(level=logging.INFO, filename=f'log/train_tri_kd.txt')
    # training and validation dataset
    train_dataset = Wsss_dataset(args)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

    train_dataloader2 = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=6)

    valid_dataset = Wsss_test_dataset(args, valid=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    test_dataset = Wsss_test_dataset(args, valid=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # network and optimizer
    model1, optimizer1, scheduler1 = get_model(args)
    model2, optimizer2, scheduler2 = get_model(args)
    model3, optimizer3, scheduler3 = get_model(args)
    # use gpu
    torch.cuda.set_device(args.gpu[0])
    model1.cuda()
    model2.cuda()
    model3.cuda()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    dice_max = 0
    for i in range(args.max_epoch):
        print(f"\nepoch:{i+1} \n-------------------")
        t0 = time.time()
        train_tri(model1, model2, model3, optimizer1, optimizer2,
                  optimizer3, train_dataloader, args, i)
        t1 = time.time()
        train_dice = validate_trainset(model1, model2, model3, train_dataloader2, verbose=True)
        valid_dice = validate(model1, model2, model3, valid_dataloader, verbose=True)
        t2 = time.time()
        print("training/validation time: {0:.2f}s/{1:.2f}s".format(t1 - t0, t2 - t1))
        logging.info('train dice {0:.4f} valid dice {1:.4f}'.format(train_dice, valid_dice))

        scheduler1.step()
        scheduler2.step()
        scheduler3.step()
        if valid_dice > dice_max:
            dice_max = valid_dice
            "save"
            cpkt1 = model1.module.state_dict()
            cpkt2 = model2.module.state_dict()
            cpkt3 = model3.module.state_dict()
            print("current best model dice=", dice_max)
    torch.save(cpkt1, os.path.join(
        args.save_folder, "tri-kd-pre1.pth"))
    torch.save(cpkt2, os.path.join(
        args.save_folder, "tri-kd-pre2.pth"))
    torch.save(cpkt3, os.path.join(
        args.save_folder, "tri-kd-pre3.pth"))
    # test
    print("test")
    model1 = DeepLabv3_Plus("resnet50", use_group_norm=True, mode="fix")
    model1.cuda()
    model1.eval()
    ckpt = torch.load(args.checkpoint1, map_location="cpu")
    model1.load_state_dict(ckpt, strict=True)

    model2 = DeepLabv3_Plus("resnet50", use_group_norm=True, mode="fix")
    model2.cuda()
    model2.eval()
    ckpt = torch.load(args.checkpoint2, map_location="cpu")
    model2.load_state_dict(ckpt, strict=True)

    model3 = DeepLabv3_Plus("resnet50", use_group_norm=True, mode="fix")
    model3.cuda()
    model3.eval()
    ckpt = torch.load(args.checkpoint3, map_location="cpu")
    model3.load_state_dict(ckpt, strict=True)

    validate(model1, model2, model3, test_dataloader, verbose=True)
    end_time = time.time()
    print("running time / mins:", (end_time - start_time) / 60)
