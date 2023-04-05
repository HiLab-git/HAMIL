## HAMIL: High-resolution Activation Maps and Interleaved Learning for Weakly Supervised Segmentation of Histopathological Images

This repository provides the code for "HAMIL: High-resolution Activation Maps and Interleaved Learning for Weakly Supervised Segmentation of Histopathological Images"
under TMI minor revision.

## Usage
1. To obtain the background masks. (If the background is not white regions, skip this step)
```train_set_root```: training set root, ```gamma_path```: path to save gamma transform for training set, ```gamma_crf_path```: path to save extracted backgrounds for training set, 

```
generate_bg_masks.py --train_set_root train_set_root --gamma_path gamma_path --gamma_crf_path gamma_crf_path
```

2. Train classification network
```
train_cls.py
```

3. Train segmentation network
```
train_seg.py
```

## Acknowledgement
The code of DeepLabv3+ is borrowed from [PuzzleCAM](https://github.com/shjo-april/PuzzleCAM)
