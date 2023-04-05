from torch.utils.data import DataLoader
from utils.LoadData import Wsss_test_dataset
from utils.LoadData import Wsss_dataset
import torchvision.transforms.functional as transF
from torchvision import transforms
from PIL import Image
from utils.Metrics import DiceMetric
from networks.ham_net import ham_net
import torch.nn.functional as F
from utils.utils import bg2mask, monte_augmentation
import torch.nn as nn
import torch.optim as optim
import argparse
import torch
import numpy as np
import sys
import os
import time
import logging
sys.path.append(os.getcwd())


def seg_to_color(seg):
    H, W = seg.shape[1], seg.shape[2]
    # whit, green, blue, yellow
    classes = ["background", "Tumor", "Stroma", "Normal"]
    color_map = [[255, 255, 255], [0, 64, 128], [64, 128, 0], [243, 152, 0]]
    img = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            img[i, j, :] = color_map[seg[0, i, j]]
    return img

def get_arguments():
    parser = argparse.ArgumentParser(description="Wsss pytorch implementation")
    parser.add_argument("--dataset_root", type=str,
                        default="", help="training images")
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Train batch size")
    parser.add_argument("--num_classes", type=int,
                        default=3, help="Train class num")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--gpu", nargs="+", type=int)
    parser.add_argument("--train_epochs", default=100, type=int)
    parser.add_argument("--save_folder", default="checkpoints2")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)

    return parser.parse_args()

def get_model(args, pre_trained=False):
    model = ham_net()
    if pre_trained:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)

    model = torch.nn.DataParallel(model, device_ids=args.gpu)
    param_groups = model.module.get_parameter_groups()
    optimizer = optim.SGD(
        [
            {"params": param_groups[0], "lr": args.lr},
            {"params": param_groups[1], "lr": 2 * args.lr},
            {"params": param_groups[2], "lr": 10 * args.lr},
            {"params": param_groups[3], "lr": 20 * args.lr},
        ],
        momentum=0.9,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 80])
    return model, optimizer, scheduler


def train(model, optimizer, train_dataloader):
    model.train()
    loss_epoch = 0
    for img, label, _ in train_dataloader:

        img, label = img.cuda(), label.cuda()
        logit = model(img)

        # loss
        loss = F.multilabel_soft_margin_loss(logit, label)
        loss_epoch += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"train loss:{loss_epoch.item()/len(train_dataloader)}")

def train_deep(model, optimizer, train_dataloader):
    model.train()
    loss_epoch = 0
    cls_acc = 0
    count = 0

    for img, label, _ in train_dataloader:

        img, label = img.cuda(), label.cuda()
        logit_b6, logit_b5, logit_b4 = model(img)

        # compute classification loss
        loss1 = F.multilabel_soft_margin_loss(logit_b6, label)
        loss2 = F.multilabel_soft_margin_loss(logit_b5, label)
        loss3 = F.multilabel_soft_margin_loss(logit_b4, label)
        
        loss = (loss1+loss2+loss3)/3

        loss_epoch += loss
        # compute cls
        label_cpu = label.cpu().detach().numpy()
        logit_cpu = logit_b6.cpu().detach().numpy()
        logit_cpu = logit_cpu > 0
        correct_num = np.sum(label_cpu == logit_cpu, axis=0)
        cls_acc += correct_num
        count += label_cpu.shape[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"train loss:{loss_epoch.item()/len(train_dataloader)}")
    return sum(cls_acc) / count / 3

def compute_dice(model, valid_dataloader, verbose=False, save=False):
    model.eval()
    # Dice_metric
    Dice_Metric = DiceMetric(4)
    # cls acc
    cls_acc = 0
    count = 0

    # my background
    my_background_root = "/mnt/data1/dataset/WSSS4LUAD/2.validation/my_bg_mask_patch_256/"
    """for every image, compute the dice"""
    with torch.no_grad():
        for img, label, bg_mask, gt, raw_img, img_name in valid_dataloader:
            # H, W
            H, W = gt.shape[1], gt.shape[2]

            img, label = img.cuda(), label.cuda()
            # logit, cam = model(img, True, (H, W))
            # multi-scale
            img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            scale = 256
            cam_a = torch.zeros((1, 3, H, W)).cuda()
            logit = torch.zeros((1, 3)).cuda()
            img_trans = transforms.Compose([transforms.Resize(
                (scale, scale)), transforms.ToTensor(), transforms.Normalize(img_mean, img_std)])
            img_path = valid_dataset.img_path + "/" + img_name[0]
            _img = Image.open(img_path).convert("RGB")
            _img = img_trans(_img)
            _img = torch.unsqueeze(_img, 0)
            _img = _img.cuda()

            logit1, _, _, _, _, _ = model(_img, True, (H, W))
            logit = logit1

            cam_a = monte_augmentation(20, model, img_path, H, W)
            cam = cam_a.clone()

            # compute cls
            label_cpu = label.cpu().detach().numpy()
            logit_cpu = logit.cpu().detach().numpy()
            logit_cpu = logit_cpu > 0
            correct_num = np.sum(label_cpu == logit_cpu, axis=0)
            cls_acc += correct_num
            count += label_cpu.shape[0]

            cam = cam.detach() * label[:, :, None, None]
            my_bg_mask = Image.open(my_background_root + img_name[0])
            my_bg_mask = np.array(my_bg_mask, np.uint8)
            # compute dice
            Dice_Metric.add_batch(cam, gt, my_bg_mask, label_cpu)
    print(f"cls_acc:{sum(cls_acc)/count/3}", cls_acc, count, cls_acc / count)
    return Dice_Metric.compute_dice(verbose=verbose, save=save), sum(cls_acc)/count/3


def save_pic(model, dataloader):
    model.eval()
    # my background
    my_background_root = "/mnt/data1/dataset/WSSS4LUAD/2.validation/my_bg_mask_patch_256/"
    # my_background_root = "/mnt/data1/lanf_zhong/label_supervision_segmentation/WSSS_background_extract/bg_masks_hht/"
    """for every image, compute the dice"""
    with torch.no_grad():
        for img, label, bg_mask, gt, raw_img, img_name in dataloader:
            # H, W
            H, W = gt.shape[1], gt.shape[2]
            img, label = img.cuda(), label.cuda()

            img_path = test_dataset.img_path + "/" + img_name[0]
            # logit, cam = model(img, True, (H, W))
            img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            s = [256]
            cam_a = torch.zeros((1, 3, H, W)).cuda()
            logit = torch.zeros((1, 3)).cuda()
            for scale in s:
                img_trans = transforms.Compose([transforms.Resize(
                    (scale, scale)), transforms.ToTensor(), transforms.Normalize(img_mean, img_std)])

                _img = Image.open(img_path).convert("RGB")
                _img = img_trans(_img)
                _img = torch.unsqueeze(_img, 0)
                _img = _img.cuda()

                # baseline
                # logit, cam_b6 = model(_img, True, (H, W))
                # cam_a += cam_b6

                # deep3
                logit, _, _, cam_b6, cam_b5, cam_b4 = model(
                    _img, True, (H, W))
                cam_a += (cam_b4+cam_b5+cam_b6)/3

            cam_a = monte_augmentation(20, model, img_path, H, W)
            # cam_a = cam_a/len(s)
            logit = logit/len(s)

            cam = cam_a.clone()

            # cam = cam.detach() * label[:, :, None, None]
            cam = cam.detach() * logit[:, :, None, None] + 1e-7
            my_bg_mask = Image.open(my_background_root + img_name[0])
            my_bg_mask = np.array(my_bg_mask, np.uint8)

            # save pic
            cam_with_bg = np.concatenate(
                (np.expand_dims(my_bg_mask, 0), cam[0].cpu().numpy()), axis=0)
            segmentation = cam_with_bg.argmax(0)
            segmentation = np.reshape(segmentation, (1, H, W))

            color_img = seg_to_color(segmentation)
            color_img = color_img.astype(np.uint8)
            color_img = Image.fromarray(color_img)
            color_img.save(f"pseudo_masks/stage1/" + img_name[0])


if __name__ == "__main__":
    args = get_arguments()
    logging.basicConfig(level=logging.INFO, filename=f'log/train_cls_deep.txt')

    time_start = time.time()
    # training and validation dataset
    train_dataset = Wsss_dataset(args, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6)

    valid_dataset = Wsss_test_dataset(args, test=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1)

    test_dataset = Wsss_test_dataset(args, test=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1)

    # network and optimizer
    model, optimizer, scheduler = get_model(args, pre_trained=False)

    # # gpu setting
    torch.cuda.set_device(args.gpu[0])
    model.cuda()

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    cls_acc_max = 0

    for i in range(args.train_epochs):
        print(f"\nepoch:{i+1} \n-------------------")

        t0 = time.time()
        train_cls_acc = train_deep(model, optimizer, train_dataloader)
        t1 = time.time()
        valid_dice, valid_cls_acc = compute_dice(model, valid_dataloader, verbose=True)
        t2 = time.time()
        print("training/validation time: {0:.2f}s/{1:.2f}s".format(t1-t0, t2-t1))
        logging.info('train cls_acc {0:.4f}, valid cls_acc {1:.4f}'.format(train_cls_acc, valid_cls_acc))

        scheduler.step()
        if valid_dice > cls_acc_max:
            cls_acc_max = valid_dice
            "save"
            ckpt = model.module.state_dict()
            print("current best model")
            torch.save(ckpt, args.checkpoint)

    model_test = ham_net()
    model_test.cuda()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_test.load_state_dict(ckpt['model'], strict=True)
    # print('\nvalidation with multi test')
    compute_dice(model_test, test_dataloader, verbose=True, save='log/DRS.csv')

    time_end = time.time()
    print(f'done, time:{(time_end-time_start)/60}')
