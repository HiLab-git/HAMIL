import sys
import os
sys.path.append(os.getcwd())
from PIL import Image
import torch
import numpy as np
import torch.nn.functional as F
import torch
from torchvision import transforms
import torchvision.transforms.functional as transF

def bg2mask(bg_path, cls_index, H, W):
    """
    input: single image background, [H,W], [0,255]
    output: mask, [H,W]
    """
    # 0 for foreground, 255 for background
    bg = np.array(Image.open(bg_path),dtype=np.uint8)
    bg[bg==0] = cls_index
    bg[bg==255] = 0
    bg = torch.tensor(bg,dtype=torch.uint8).unsqueeze(0).unsqueeze(0)
    bg = F.interpolate(bg,(H,W),mode='nearest')
    return bg.squeeze(0).squeeze(0)


def monte_augmentation(n, model, img_path, H, W):
    """
    n: the number
    """
    cam = torch.zeros((1, 3, H, W)).cuda()
    img_mean, img_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    for i in range(n):
        scale_factor = 1+np.random.choice([0,0.25,0.5])
        H_scale, W_scale = int(scale_factor*256), int(scale_factor*256)
        p_vflip = np.random.rand()
        p_hflip = np.random.rand()
        vflip_flag = p_vflip>0.5
        hflip_flag = p_hflip>0.5
        rotation_degree = int(np.random.choice([0,45,90,135,180,225,270]))

        img_trans = transforms.Compose([transforms.Resize((H_scale, W_scale)),transforms.ToTensor(),transforms.Normalize(img_mean, img_std)])
        _img = Image.open(img_path).convert("RGB")
        _img = img_trans(_img)
        _img = torch.unsqueeze(_img, 0)
        _img = _img.cuda()

        if vflip_flag:
            _img = transF.vflip(_img)
        
        if hflip_flag:
            _img = transF.hflip(_img)
        
        if rotation_degree:
            # print(rotation_degree)
            _img = transF.rotate(_img,rotation_degree)

        _,_,_,cam_b6,cam_b5,cam_b4 = model(_img,True,(H,W))

        if rotation_degree:
            cur_degree = 360-rotation_degree
            cam_b6,cam_b5,cam_b4 = transF.rotate(cam_b6,cur_degree),transF.rotate(cam_b5,cur_degree),transF.rotate(cam_b4,cur_degree)

        if hflip_flag:
            cam_b6,cam_b5,cam_b4 = transF.hflip(cam_b6),transF.hflip(cam_b5),transF.hflip(cam_b4)
        
        if vflip_flag:
            cam_b6,cam_b5,cam_b4 = transF.vflip(cam_b6),transF.vflip(cam_b5),transF.vflip(cam_b4)

        cam += (cam_b6+cam_b5+cam_b4)/3

    return cam/n

if __name__ == "__main__":
    monte_augmentation(15,True,True)