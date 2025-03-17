# UPerNet with DAMamba backbone on ADE20K

## Model Zoo

**ADE20K semantic segmentation results using the UPerNet method:**

| Backbone  |                               Pretrained Model                               | Crop Size |Lr Schd| mIoU | mIoU (ms) | #Params |                                                Download                                                |
|:---------:|:----------------------------------------------------------------------------:|:---:|:---:|:----:|:---------:|:-------:|:------------------------------------------------------------------------------------------------------:|
| DAMamba-T | [ImageNet-1K](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-T.pth) |512x512|160K| 50.3 |   51.2    |   55M   | [model](https://huggingface.co/ltzovo/DAMamba/blob/main/upernet_DAMamba_tiny_512x512_160k_ade20k.pth)  |
| DAMamba-S | [ImageNet-1K](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-S.pth) |512x512|160K| 51.2 |   52.0    |   75M   | [model](https://huggingface.co/ltzovo/DAMamba/blob/main/upernet_DAMamba_small_512x512_160k_ade20k.pth) |
| DAMamba-B | [ImageNet-1K](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-B.pth) |512x512|160K| 51.9 |   52.3    |  117M   | [model](https://huggingface.co/ltzovo/DAMamba/blob/main/upernet_DAMamba_base_512x512_160k_ade20k.pth)  |
* In the context of multi-scale evaluation, DAMamba reports test results under two distinct scenarios: **interpolation** and **extrapolation** of relative position bias. 

## Requirements

    pip install -r requirements.txt



## Evaluation

***Single-scale Evaluation:***

To run single-scale evaluation of UPerNet models with DAMamba backbone on ADE20K, you can use the following command:

    bash dist_test.sh <config-file-ending-with-"ss"> <checkpoint-path> <gpu-num> --eval mIoU

For example, to evaluate the DAMamba-T on a single GPU:
    
    bash dist_test.sh ./configs/upernet_DAMamba_tiny_512x512_160k_ade20k_ss.py /path/to/checkpoint_file 1 --eval mIoU
    
For example, to evaluate the DAMamba-T on 8 GPUs:
    
    bash dist_test.sh ./configs/upernet_DAMamba_tiny_512x512_160k_ade20k_ss.py /path/to/checkpoint_file 8 --eval mIoU

***Multi-scale Evaluation:***

To evaluate the pre-trained models with multi-scale inputs and flip augmentations on ADE20K under`interpolation of relative position bias` strategy, you can use the following command:
    
    bash dist_test.sh <config-file-ending-with-"ms"> <checkpoint-path> <gpu-num> --eval mIoU  --aug-test



## Training
In order to train UPerNet models with DAMamba backbone on the ADE20K dataset, first, you need to fill in the path of your downloaded pretrained checkpoint in `./configs/<config-file-ending-with-"ss">`. Specifically, change it to:
    
    pretrained=<path-to-checkpoint>, 

After setting up, to train DAMamba on ADE20K dataset, you can use the following command:
    
    bash dist_train.sh <config-file-ending-with-"ss"> <gpu-num> 

For example, to train the DAMamba-T on 8 GPUs, with a total batch-size of 16:

    bash dist_train.sh ./configs/upernet_DAMamba_tiny_512x512_160k_ade20k_ss.py 8

***Notice:** Our DAMamba models are trained with single-scale images, if you want to reproduce accurately, please use `<config-file-ending-with-"ss">`.*

## Acknowledgement

The released script for Object Detection with DAMamba is built based on the [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [timm](https://github.com/huggingface/pytorch-image-models) library.

## License

This project is released under the Apache 2.0 license. Please see the [LICENSE](/LICENSE) file for more information.


## Citation

If you find our work helpful, please consider citing the following bibtex. We would greatly appreciate a star for this
project.

    @article{li2025damamba,
    title={DAMamba: Vision State Space Model with Dynamic Adaptive Scan},
    author={Li, Tanzhe and Li, Caoshuo and Lyu, Jiayi and Pei, Hongjuan and Zhang, Baochang and Jin, Taisong and Ji, Rongrong},
    journal={arXiv preprint arXiv:2502.12627},
    year={2025}
    }