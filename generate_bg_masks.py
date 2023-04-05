import argparse
import os
from PIL import Image
import numpy as np
import scipy
from utils.CRF import MyCRF
import re
import cv2
from skimage import morphology


def get_arguments():
    parser = argparse.ArgumentParser(description="HAMIL pytorch implementation")
    parser.add_argument("--train_set_root", type=str, default="", help="training images root")
    parser.add_argument("--gamma_path", type=str, default="", help="gamma transform root")
    parser.add_argument("--gamma_crf_path", type=str, default="", help="background root")
    return parser.parse_args()

def gamma_transform(img_path):
    image = Image.open(img_path).convert("L")
    image = np.array(image, dtype=np.float32)
    image /= 255
    gamma = 2.4
    out = np.power(image, gamma)
    out *= 255
    out = out.astype(np.uint8)

    return out

def seg_to_color(seg):
    H, W = seg.shape[0], seg.shape[1]
    # white, green, blue, yellow
    classes = ["background", "Tumor", "Stroma", "Normal"]
    color_map = [[255, 255, 255], [0, 64, 128], [64, 128, 0], [243, 152, 0]]
    img = np.zeros((H, W, 3))
    for i in range(H):
        for j in range(W):
            img[i, j, :] = color_map[seg[i, j]]
    return img

def open_with_crf(img_path, open_img_path):
    img = Image.open(img_path).convert("RGB")
    img_array = np.array(img, dtype=np.float32)
    open_img = Image.open(open_img_path)
    open_img = np.array(open_img, dtype=np.float32)
    H, W = img_array.shape[0], img_array.shape[1]
    mycrf = MyCRF()
    p = 1.0
    background = open_img.copy()
    background /= np.max(background)
    foreground = (1 - background) * p
    probability_map = np.concatenate(
        (foreground.reshape((1, H, W)), background.reshape((1, H, W))), axis=0
    )
    out = mycrf.inference(img_array, probability_map)
    out = out.argmax(0)
    return (out * 255).astype(np.uint8)

def get_label(img_name):
    res = re.findall(r"\[(.*?)\]", img_name)
    label = np.array(list(eval(res[0])), dtype=np.uint8)
    return label


if __name__ == "__main__":
    args = get_arguments()
    dataset_root = args.train_set_root
    gamma_dir = args.gamma_path
    gamma_crf_dir = args.gamma_crf_path
    img_paths = os.listdir(dataset_root)
    # clean
    img_paths = filter(img_paths)
    print(f"all {len(img_paths)} images")

    for img_name in img_paths:
        """save gamma crf background"""
        img_path = dataset_root + img_name
        img_gamma = gamma_transform(img_path)
        img_gamma = Image.fromarray(img_gamma)
        gamma_path = gamma_dir + img_name
        img_gamma.save(gamma_path)
        open_crf = open_with_crf(img_path, gamma_path)
        img_open_crf = Image.fromarray(open_crf)
        img_open_crf.save(gamma_crf_dir + img_name)
        out = Image.open(img_path)
        out = np.array(out).astype(np.uint8)
        if len(np.unique(out)) == 1:
            os.remove(img_path)
        out_remove = np.array(out, dtype=bool)
        morphology.remove_small_holes(out_remove, 32, 1, True)
        out_remove = Image.fromarray(out_remove)
        out_remove.save("gamma_crf_train/" + img_name)

    print("done!")

