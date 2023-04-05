import argparse
import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import re
from torchvision.transforms import functional as F
import random

class Normalization(object):
    def __call__(self, image):
        image_cpu = image.cpu().detach().numpy()
        mean_image = [0, 0, 0]
        std_image = [0, 0, 0]
        r, g, b = (
            image_cpu[0, :, :].copy(),
            image_cpu[1, :, :].copy(),
            image_cpu[2, :, :].copy(),
        )
        mean_image[0], mean_image[1], mean_image[2] = (
            np.mean(r, axis=(0, 1)),
            np.mean(g, axis=(0, 1)),
            np.mean(b, axis=(0, 1)),
        )
        std_image[0], std_image[1], std_image[2] = (
            np.std(r, axis=(0, 1)),
            np.std(g, axis=(0, 1)),
            np.std(b, axis=(0, 1)),
        )
        image = F.normalize(image, mean=mean_image, std=std_image)
        return image


class Wsss_dataset(Dataset):
    def __init__(self, args, train=True):
        super().__init__()
        self.dataset_root = args.dataset_root + "/" + "1.training/"
        self.my_bg_path = "/mnt/data1/dataset/WSSS4LUAD/1.training/gamma_crf_train/"
        self.input_size = args.input_size
        self.crop_size = args.crop_size
        img_paths = os.listdir(self.dataset_root)
        self.img_paths = self._filter(img_paths)
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # shuffle
        np.random.seed(0)
        np.random.shuffle(self.img_paths)

        self.img_paths = self.get_new_set(self.img_paths)
        # self.img_paths = self.get_single_cls(self.img_paths)
        print(f'train images:{len(self.img_paths)}')
        if train:
            self.img_transform = transforms.Compose(
                [
                    transforms.Resize((self.input_size, self.input_size)),
                    # transforms.Resize((random.randint(args.min_image_size, args.max_image_size),)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation((0, 180)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                    # [0.6270, 0.5013, 0.7519], [0.1627, 0.1682, 0.0977] another dataset.
                    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] ImageNet
                    # [0.737, 0.505, 0.678], [0.176, 0.210, 0.146] WSSS train dataset
                ]
            )
        else:
            self.img_paths = self.img_paths[:]
            print(f'generate pseudo masks:{len(self.img_paths)}')
            self.img_transform = transforms.Compose(
                [
                    transforms.Resize((self.input_size, self.input_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean, self.std),
                ]
            )

    def _filter(self, img_paths):
        img_list_new = []
        for img_name in img_paths:
            if img_name[-4:] == ".png":
                img_list_new.append(img_name)
            else:
                pass
        return img_list_new

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img_path = self.dataset_root + "/" + img_name
        label = self.get_label(img_name)
        img = Image.open(img_path).convert("RGB")

        # my_bg_mask, and covert the background to 0
        # my_bg_mask = np.array(Image.open(self.my_bg_path+img_name),dtype=np.uint8)
        # img = np.array(img,dtype=np.uint8)
        # img[my_bg_mask==255]=0
        # img = Image.fromarray(img)

        raw_img = np.array(img, dtype=np.uint8)
        img = self.img_transform(img)

        return img, label, img_name

    def get_label(self, img_name):
        res = re.findall(r"\[(.*?)\]", img_name)
        label = torch.tensor(list(eval(res[0])))
        return label

    def get_single_cls(self, img_names):
        new_names = []
        for img_name in img_names:
            label = self.get_label(img_name)
            if sum(label) == 1:
                new_names.append(img_name)
        return new_names

    def get_new_set(self, img_names):
        new_names = []
        for img_name in img_names:
            label = self.get_label(img_name)
            if sum(label) != 1:
                new_names.append(img_name)
        single_names = self.get_single_cls(img_names)
        portion = 0.25
        single_names = single_names[:int(portion*len(single_names))]
        new_names = new_names + single_names
        return new_names

    def __len__(self):
        return len(self.img_paths)


class Wsss_test_dataset(Dataset):
    def __init__(self, args, test=False):
        super().__init__()
        self.dataset_root = args.dataset_root + "/" + "2.validation"
        self.input_size = args.input_size
        self.crop_size = args.crop_size
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # "_patch_224_320"
        self.mask_path = self.dataset_root + "/" + "mask_patch_256"
        self.img_path = self.dataset_root + "/" + "img_patch_256"
        self.my_bg_path = self.dataset_root + "/" + "my_bg_mask_patch_256"
        self.img_names = os.listdir(self.img_path)
        # self.img_names = self.img_names[:31]
        np.random.seed(42)
        np.random.shuffle(self.img_names)
        "colormap"
        self.classes = ["background", "Tumor", "Stroma", "Normal"]
        self.color_map = [[255, 255, 255], [0, 64, 128], [64, 128, 0], [243, 152, 0]]
        if test==False:
            self.img_names = self.img_names[:300]
            # self.img_names = self.filter_single_class_image()
            print(f'valid images:{len(self.img_names)}')
        else:
            self.img_names = self.img_names[300:]
            print(f'test images:{len(self.img_names)}')
        "transform."
        self.img_trans = transforms.Compose(
            [
                transforms.Resize((self.crop_size, self.crop_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def __getitem__(self, index):

        img_name = self.img_names[index]

        img_path = self.img_path + "/" + img_name
        mask_path = self.mask_path + "/" + img_name
        my_bg_path = self.my_bg_path + "/" + img_name

        img = Image.open(img_path).convert("RGB")
        raw_img = np.array(img, dtype=np.uint8)
        mask = Image.open(mask_path).convert("RGB")

        my_bg_mask = Image.open(my_bg_path)
        my_bg_mask = np.array(my_bg_mask,dtype=np.uint8)

        # convert the background to 0
        # img = np.array(img,dtype=np.uint8)
        # img[my_bg_mask==255]=0
        # img = Image.fromarray(img)

        mask = np.array(mask).astype(np.uint8)
        mask = self.image2label(mask)
        label = self.get_label_from_img(mask)
        img = self.img_trans(img)
        # my_bg_map,
        return img, label, my_bg_mask, mask, raw_img, img_name

    def image2label(self, im):
        color2int = np.zeros(256 ** 3) 
        for idx, color in enumerate(self.color_map):
            color2int[(color[0] * 256 + color[1]) * 256 + color[2]] = idx 
            data = np.array(im, dtype=np.int32)
            idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(color2int[idx], dtype=np.int32)   

    def get_label_from_img(self, img_label):
        temp = np.unique(img_label)
        cls_label = np.zeros(
            3,
        )
        if 1 in temp:
            cls_label[0] = 1
        if 2 in temp:
            cls_label[1] = 1
        if 3 in temp:
            cls_label[2] = 1
        return torch.tensor(cls_label, dtype=torch.int32)

    def filter_single_class_image(self):
        single_class_images = []
        for img_name in self.img_names:
            mask_path = self.mask_path + "/" + img_name
            mask = Image.open(mask_path).convert("RGB")
            mask = np.array(mask).astype(np.uint8)
            mask = self.image2label(mask)
            label = self.get_label_from_img(mask)
            if sum(label) != 1:
                if label[2] == 1:
                    single_class_images.append(img_name)
        return single_class_images

    def filter_multi_class_image(self):
        multi_class_images = []
        for img_name in self.img_names:
            mask_path = self.mask_path + "/" + img_name
            mask = Image.open(mask_path).convert("RGB")
            mask = np.array(mask).astype(np.uint8)
            mask = self.image2label(mask)
            label = self.get_label_from_img(mask)
            if sum(label) != 1:
                multi_class_images.append(img_name)
        return multi_class_images

    def filter(self, img_paths):
        img_list_new = []
        for img_name in img_paths:
            if "_" in img_name:
                img_list_new.append(img_name)
            else:
                pass
        return img_list_new

    def __len__(self):
        return len(self.img_names)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Wsss pytorch implementation")
    parser.add_argument("--dataset_root", type=str, default="/mnt/data1/dataset/WSSS4LUAD/", help="training images")
    parser.add_argument("--batch_size", type=int, default=64, help="Train batch size")
    parser.add_argument("--num_classes", type=int, default=3, help="Train class num")
    parser.add_argument("--delta", type=float, default=0, help="set 0 for the...")
    # 0.01 for deep3 -->81.6; 0.005 for deep2; 0.005 for deep4; 0.001 for baseline
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--gpu", nargs="+", type=int)
    parser.add_argument("--train_epochs", default=30, type=int)
    parser.add_argument("--save_folder", default="checkpoints")
    parser.add_argument("--checkpoint", type=str, default="")

    # (280, 256), (256, 224), (324, 256),
    parser.add_argument("--input_size", type=int, default=256)
    parser.add_argument("--crop_size", type=int, default=224)
    parser.add_argument("--test_size", type=int, default=384)

    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--min_image_size', default=256, type=int)
    parser.add_argument('--max_image_size', default=512, type=int)
    
    args = parser.parse_args()

    train_set = Wsss_dataset(args)
    for item in train_set:
        break