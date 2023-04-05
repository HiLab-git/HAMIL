import os
import numpy as np
import torch
import torch.nn.functional as Func
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import re
import torchvision
import math
import numpy as np
import random
from PIL import Image
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size,
                              interpolation=F.InterpolationMode.NEAREST)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size):
        self.target_size = random.randint(min_size, max_size)
        self.size = (self.target_size, self.target_size)

    def __call__(self, image, target=None):
        image = F.resize(image, self.size)
        if target is not None:
            target = F.resize(target, self.size,
                              interpolation=F.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target=None):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            if target is not None:
                target = F.vflip(target)
        return image, target


class RandomRotation(object):
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, target=None):
        angle = random.randint(self.degrees[0], self.degrees[1])
        image = F.rotate(image, angle)
        if target is not None:
            target = F.rotate(target, angle)
        return image, target


class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        crop_params = T.RandomCrop.get_params(image, self.size)
        image = F.crop(image, *crop_params)
        if target is not None:
            target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        if target is not None:
            target = F.center_crop(target, self.size)
        return image, target


class NormalizeWithMeanStd(object):
    """Nomralize the image (shape [C, D, H, W] or [C, H, W]) with mean and std for given channels"""

    def __init__(self, mean=None, std=None):
        """
        :param chanels: (None or tuple/list) the indices of channels to be noramlized.
        :param mean: (None or tuple/list): The mean values along each channel.
        :param  std : (None or tuple/list): The std values along each channel.
            When mean and std are not provided, calculate them from the entire image
            region or the non-positive region.
        :param ignore_non_positive: (bool) Only used when mean and std are not given.
            Use positive region to calculate mean and std, and set non-positive region to random.
        :param inverse: (bool) Whether inverse transform is needed or not.
        """
        super(NormalizeWithMeanStd, self).__init__()
        self.mean = mean
        self.std = std
        self.ingore_np = False

    def __call__(self, image, target=None):
        image = image
        chns = range(image.shape[0])

        if self.mean is None:
            self.mean = [None] * len(chns)
            self.std = [None] * len(chns)

        for i in range(len(chns)):
            chn = chns[i]
            chn_mean, chn_std = self.mean[i], self.std[i]
            if chn_mean is None:
                if self.ingore_np:
                    pixels = image[chn][image[chn] > 0]
                    chn_mean, chn_std = pixels.mean(), pixels.std()
                else:
                    chn_mean, chn_std = image[chn].mean(), image[chn].std()

            chn_norm = (image[chn] - chn_mean) / chn_std

            if self.ingore_np:
                chn_random = np.random.normal(0, 1, size=chn_norm.shape)
                chn_norm[image[chn] <= 0] = chn_random[image[chn] <= 0]
            image[chn] = chn_norm
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class Pad(object):
    def __init__(self, padding_n, padding_fill_value=0, padding_fill_target_value=0):
        self.padding_n = padding_n
        self.padding_fill_value = padding_fill_value
        self.padding_fill_target_value = padding_fill_target_value

    def __call__(self, image, target):
        image = F.pad(image, self.padding_n, self.padding_fill_value)
        if target is not None:
            target = F.pad(target, self.padding_n,
                           self.padding_fill_target_value)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if target is not None:
            target = torch.as_tensor(np.array(target), dtype=torch.uint8)
        return image, target


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask=None):
        for t in self.transforms:
            image, mask = t(image, mask)
        return {"image": image, "mask": mask}


class Wsss_dataset(Dataset):
    def __init__(self, args):
        super().__init__()
        # pseudo_masks
        self.dataset_root = args.dataset_root + "/" + \
            "1.training/pseudo_masks_deep3_res32/"
        img_paths = os.listdir(self.dataset_root)
        self.img_paths = self._filter(img_paths)
        self.dataset_root = args.dataset_root + "/" + "1.training"

        self.color_map = [[255, 255, 255], [
            0, 64, 128], [64, 128, 0], [243, 152, 0]]
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # self.mean, self.std = [0.737, 0.505, 0.678], [0.176, 0.210, 0.146]
        self.image_size = args.image_crop
        np.random.seed(0)
        np.random.shuffle(self.img_paths)

        self.img_paths = self.img_paths[:]
        self.count_two()
        # self.img_paths = self.get_not_single(self.img_paths)
        print(f"train images:{len(self.img_paths)}")
        self.img_transform = Compose(
            [
                Resize((args.image_resize, args.image_resize)),
                RandomCrop((args.image_crop, args.image_crop)),
                RandomHorizontalFlip(0.5),
                ToTensor(),
                # NormalizeWithMeanStd(),
                Normalize(self.mean, self.std),
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
        seg_path = self.dataset_root + "/pseudo_masks_deep3_res32/" + img_name
        # seg_path = self.dataset_root + "/pseudo_masks_baseline/" + img_name
        # seg_path = self.dataset_root + "/pseudo_masks_update/" + img_name
        # seg_path = self.dataset_root + "/pseudo_masks_histoseg/" + img_name
        # soft label; .npy
        soft_seg_path = (
            "/mnt/data1/lanf_zhong/label_supervision_segmentation/WSSS_vgg/pseudo_masks/pseudo_masks_deep_soft/"
            + img_name[:-4]
            + ".npy"
        )
        seg = Image.open(seg_path)
        seg = np.array(seg, dtype=np.uint8)
        seg = self.image2label(seg)
        seg = Image.fromarray(seg.astype(np.uint8))
        img = Image.open(img_path).convert("RGB")
        soft_seg = np.load(soft_seg_path, allow_pickle=True)
        soft_seg = torch.from_numpy(soft_seg)
        soft_seg = Func.interpolate(torch.unsqueeze(
            soft_seg, 0), (self.image_size, self.image_size))
        soft_seg = torch.squeeze(soft_seg)

        output_dic1 = self.img_transform(img, seg)

        image1 = output_dic1["image"]
        mask1 = output_dic1["mask"]
        label = self.get_label(img_name)
        # test dataset.
        # sample_img, sample_mask = sample["image"], sample['mask']
        # sample_img = F.to_pil_image(sample_img).save(img_name)
        # sample_mask = F.to_pil_image(sample_mask).save('mask'+img_name)

        return image1, mask1, label, img_name

    def get_label(self, img_name):
        res = re.findall(r"\[(.*?)\]", img_name)
        label = torch.tensor(list(eval(res[0])))
        label = torch.cat((torch.tensor([1, ]), label))
        return label

    def erase_bg(self, img, saliency_map):
        r, g, b = img[0, :, :], img[1, :, :], img[2, :, :]
        r[saliency_map == 1] = 0
        g[saliency_map == 1] = 0
        b[saliency_map == 1] = 0
        H, W = img.shape[1], img.shape[2]
        erase_img = torch.cat(
            (r.reshape(1, H, W), g.reshape(1, H, W), b.reshape(1, H, W)), axis=0)
        return erase_img

    def image2label(self, im):
        color2int = np.zeros(256 ** 3)
        for idx, color in enumerate(self.color_map):
            color2int[(color[0] * 256 + color[1]) * 256 + color[2]] = idx
            data = np.array(im, dtype=np.int32)
            idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(color2int[idx], dtype=np.int32).astype(np.uint8)

    def get_single_cls(self, img_names):
        new_names = []
        for img_name in img_names:
            label = self.get_label(img_name)
            if sum(label) == 1:
                new_names.append(img_name)
        return new_names

    def get_not_single(self, img_names):
        new_names = []
        for img_name in img_names:
            label = self.get_label(img_name)
            if sum(label) != 1:
                new_names.append(img_name)
        return new_names

    def count_two(self):
        all_num = len(self.img_paths)
        TCGA_num = 0
        for img_name in self.img_paths:
            if 'TCGA' in img_name:
                TCGA_num += 1
        print(f'GDPH:{all_num-TCGA_num}, TCGA:{TCGA_num}')

    def __len__(self):
        return len(self.img_paths)


class Wsss_test_dataset(Dataset):
    def __init__(self, args, valid=True):
        super().__init__()
        self.dataset_root = args.dataset_root + "/" + "2.validation"
        self.test_size = args.test_image_size
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        # self.mean, self.std = [0.737, 0.505, 0.678], [0.176, 0.210, 0.146]
        # "_patch_224_320"
        self.mask_path = self.dataset_root + "/" + "mask_patch_256"
        self.img_path = self.dataset_root + "/" + "img_patch_256"
        self.bg_path = self.dataset_root + "/" + "bg_mask_patch_256"
        self.img_names = os.listdir(self.img_path)

        np.random.seed(42)
        np.random.shuffle(self.img_names)
        if valid == True:
            self.img_names = self.img_names[:300]
            print(f"valid images:{len(self.img_names)}")
        else:
            self.img_names = self.img_names[300:]
            print(f"test images:{len(self.img_names)}")
        "transform"
        # self.trans = Compose([Resize((256,256)),ToTensor(),NormalizeWithMeanStd()])
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.test_size, self.test_size)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        # self.img_trans1 = transforms.Compose([transforms.Resize((256, 256)),transforms.ToTensor()])
        # self.img_trans2 = transforms.Compose([transforms.Resize((320, 320)),transforms.ToTensor(),transforms.Normalize(self.mean, self.std)])
        # self.img_trans3 = transforms.Compose([transforms.Resize((384, 384)),transforms.ToTensor(),transforms.Normalize(self.mean, self.std)])
        # self.img_trans4 = transforms.Compose([transforms.Resize((448, 448)),transforms.ToTensor(),transforms.Normalize(self.mean, self.std)])
        # self.img_trans5 = transforms.Compose([transforms.Resize((512, 512)),transforms.ToTensor(),transforms.Normalize(self.mean, self.std)])
        # self.img_trans6 = transforms.Compose([transforms.Resize((192, 192)),transforms.ToTensor(),transforms.Normalize(self.mean, self.std)])

        "colormap"
        self.classes = ["background", "Tumor", "Stroma", "Normal"]
        self.color_map = [[255, 255, 255], [
            0, 64, 128], [64, 128, 0], [243, 152, 0]]

    def __getitem__(self, index):

        img_name = self.img_names[index]

        img_path = self.img_path + "/" + img_name
        mask_path = self.mask_path + "/" + img_name

        seg = Image.open(mask_path).convert("RGB")
        seg = np.array(seg, dtype=np.uint8)
        mask = self.image2label(seg)
        # seg = Image.fromarray(seg)
        img_label = self.get_label_from_img(seg)

        img = Image.open(img_path).convert("RGB")
        image = self.img_transform(img)

        # output_dic1 = self.trans(img, seg)
        # image = output_dic1['image']
        # mask = output_dic1['mask']
        # test dataset.
        # sample_img, sample_mask = sample["image"], sample['mask']
        # sample_img = F.to_pil_image(sample_img).save(img_name)
        # sample_mask = seg_to_color(np.expand_dims(np.array(sample_mask), 0))
        # sample_mask = F.to_pil_image(sample_mask.astype(np.uint8)).save('mask'+img_name)
        return image, mask, img_label, img_name

    def image2label(self, im):
        color2int = np.zeros(256 ** 3)
        for idx, color in enumerate(self.color_map):
            color2int[(color[0] * 256 + color[1]) * 256 + color[2]] = idx
            data = np.array(im, dtype=np.int32)
            idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(color2int[idx], dtype=np.int32).astype(np.uint8)

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


class Wsss_trainset_update(Dataset):
    def __init__(self, args):
        super().__init__()
        # pseudo_masks
        self.dataset_root = args.dataset_root + "/" + "1.training/pseudo_masks_deep/"
        img_paths = os.listdir(self.dataset_root)
        self.img_paths = self._filter(img_paths)
        self.dataset_root = args.dataset_root + "/" + "1.training"

        self.color_map = [[255, 255, 255], [
            0, 64, 128], [64, 128, 0], [243, 152, 0]]
        self.mean, self.std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        np.random.seed(0)
        np.random.shuffle(self.img_paths)

        self.img_paths = self.img_paths[:]
        self.single_img_paths = self.get_single_cls(self.img_paths)
        self.img_paths = self.get_not_single(self.img_paths)
        print(f"update multi cls images:{len(self.img_paths)}")

        self.img_trans1 = transforms.Compose(
            [transforms.Resize((256, 256)), transforms.ToTensor(
            ), transforms.Normalize(self.mean, self.std)]
        )
        self.img_trans2 = transforms.Compose(
            [transforms.Resize((320, 320)), transforms.ToTensor(
            ), transforms.Normalize(self.mean, self.std)]
        )
        self.img_trans3 = transforms.Compose(
            [transforms.Resize((384, 384)), transforms.ToTensor(
            ), transforms.Normalize(self.mean, self.std)]
        )
        self.img_trans4 = transforms.Compose(
            [transforms.Resize((448, 448)), transforms.ToTensor(
            ), transforms.Normalize(self.mean, self.std)]
        )
        self.img_trans5 = transforms.Compose(
            [transforms.Resize((512, 512)), transforms.ToTensor(
            ), transforms.Normalize(self.mean, self.std)]
        )
        self.img_trans6 = transforms.Compose(
            [transforms.Resize((192, 192)), transforms.ToTensor(
            ), transforms.Normalize(self.mean, self.std)]
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

        img = Image.open(img_path).convert("RGB")
        img_label = self.get_label(img_name)

        img1 = self.img_trans1(img)
        img2 = self.img_trans2(img)
        img3 = self.img_trans3(img)
        img4 = self.img_trans4(img)
        img5 = self.img_trans5(img)
        img6 = self.img_trans6(img)

        return img1, img2, img3, img4, img5, img6, img_label, img_name

    def get_label(self, img_name):
        res = re.findall(r"\[(.*?)\]", img_name)
        label = torch.tensor(list(eval(res[0])))
        return label

    def image2label(self, im):
        color2int = np.zeros(256 ** 3)
        for idx, color in enumerate(self.color_map):
            color2int[(color[0] * 256 + color[1]) * 256 + color[2]] = idx
            data = np.array(im, dtype=np.int32)
            idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        return np.array(color2int[idx], dtype=np.int32).astype(np.uint8)

    def get_single_cls(self, img_names):
        new_names = []
        for img_name in img_names:
            label = self.get_label(img_name)
            if sum(label) == 1:
                new_names.append(img_name)
        return new_names

    def get_not_single(self, img_names):
        new_names = []
        for img_name in img_names:
            label = self.get_label(img_name)
            if sum(label) != 1:
                new_names.append(img_name)
        return new_names

    def __len__(self):
        return len(self.img_paths)
