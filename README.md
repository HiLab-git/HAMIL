## HAMIL: High-resolution Activation Maps and Interleaved Learning for Weakly Supervised Segmentation of Histopathological Images

This repository provides the code for "HAMIL: High-resolution Activation Maps and Interleaved Learning for Weakly Supervised Segmentation of Histopathological Images"
under TMI minor revision.

## Usage
1. To obtain the background masks. (If the background is not white regions, skip this step)

```
utils.generate_bg_masks.py
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
