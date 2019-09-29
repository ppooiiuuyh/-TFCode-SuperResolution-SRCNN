# [Super Resolution] SRCNN - tensorflow implementation
tensorflow implementation of SRCNN

## Prerequisites
 * python 3.x
 * Tensorflow > 1.x
 * Pillow
 * OpenCV
 * argparse

## Properties (what's different from reference code)
 * This code requires Tensorflow. This code was fully implemented based on Python 3 differently from the original.
 * This code supports both RGB and YCBCR channel space.
 * This code use Adam optimizer instead of GradienDecentent Optimizer differently from the original (it can be easily modified).
 * This code supports tensorboard summarization
 * This code supports model saving and restoration
 * We use convolution with "same pading" for simplicity
 * this code uses PILLOW library to resize image. Note that the performance of Bicubic function in PILLOW is lower than that of Matlab library. 

## Usage
```
usage: python3 trainer.py --gpu 0 

[options]
""" system """
parser.add_argument("--exp_type", type=int, default=0, help='experiment type')
parser.add_argument("--gpu", type=str, default=1)  # -1 for CPU
parser.add_argument("--model_tag", type=str, default="default", help='Exp name to save logs/checkpoints.')
parser.add_argument("--checkpoint_dir", type=str, default='../__outputs/checkpoints/', help='Dir for checkpoints.')
parser.add_argument("--summary_dir", type=str, default='../__outputs/summaries/', help='Dir for tensorboard logs.')
parser.add_argument("--restore_model_file", type=str, default=None, help='file for restoration')
#parser.add_argument("--restore_model_file", type=str, default='../__outputs/checkpoints/SRCNN_SRCNN_model_default_09_29_15_10_00/model.ckpt-170', help='file for resotration')

""" model """
parser.add_argument("--batch_size", type=int, default=64, help='Minibatch size(global)')
parser.add_argument("--patch_size", type=int, default=33, help='Minipatch size(global)')
#parser.add_argument("--patch_stride", type=int, default=13, help='patch stride')
parser.add_argument("--operating_channel", type=str, default="RGB", help="operating channel [RGB, YCBCR")  # -1 for CPU
parser.add_argument("--num_channels", type=int, default=3, help='the number of channels')
parser.add_argument("--scale", type=int, default=3, help='scaling factor')
parser.add_argument("--data_root_train", type=str, default="/projects/datasets/restoration/SR_training_datasets/T91", help='Data root dir')
parser.add_argument("--data_root_test", type=str, default="/projects/datasets/restoration/SR_testing_datasets/Set5", help='Data root dir')

""" training """
parser.add_argument("--learning_rate", type=float, default=0.0001, help="lr")
parser.add_argument("--learning_rate_sub", type=float, default=0.00001, help="lr2")
config = parser.parse_args()
```

 * For running tensorboard, `tensorboard --logdir=../__outputs/summaries` then access localhost:6006 with your browser

## Result
<p align="center">
<img src="https://github.com/ppooiiuuyh/assets/blob/master/srcnn_result1.png?raw=true" width="600">
</p>

<p align="center">
<img src="https://github.com/ppooiiuuyh/assets/blob/master/srcnn_result2.png?raw=true" width="600">
</p>




## References
* [tegg89/SRCNN-Tensorflow](https://github.com/tegg89/SRCNN-Tensorflow) : reference source code
* [SRCNN](https://arxiv.org/abs/1501.00092) : reference paper
* [taki0112/Tensorflow-cookbook](https://github.com/taki0112/Tensorflow-Cookbook) : useful tensorflow cook reference

## Author
Dohyun Kim

