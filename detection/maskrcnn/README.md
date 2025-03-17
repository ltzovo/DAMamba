# Mask R-CNN with DAMamba backbone on COCO

## Model Zoo

**COCO object detection and instance segmentation results using the Mask R-CNN 1x method:**

| Backbone  |                               Pretrained Model                               | Lr Schd| box mAP | mask mAP | #Params |                                             Download                                             |
|:---------:|:----------------------------------------------------------------------------:|:---:|:-------:|:--------:|:-------:|:------------------------------------------------------------------------------------------------:|
| DAMamba-T | [ImageNet-1K](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-T.pth) |1x|  48.5   |   43.4   |   45M   | [model](https://huggingface.co/ltzovo/DAMamba/blob/main/mask_rcnn_DAMamba_tiny_fpn_1x_coco.pth)  |
| DAMamba-S | [ImageNet-1K](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-S.pth) |1x|  49.8   |   44.5   |   65M   | [model](https://huggingface.co/ltzovo/DAMamba/blob/main/mask_rcnn_DAMamba_small_fpn_1x_coco.pth) |
| DAMamba-B | [ImageNet-1K](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-B.pth) |1x|  50.6   |   44.9   |  105M   | [model](https://huggingface.co/ltzovo/DAMamba/blob/main/mask_rcnn_DAMamba_base_fpn_1x_coco.pth)  |

**COCO object detection and instance segmentation results using the Mask R-CNN 3x method:**

| Backbone  |                               Pretrained Model                               | Lr Schd | box mAP | mask mAP | #Params |                                          Download                                           |
|:---------:|:----------------------------------------------------------------------------:|:-------:|:-------:|:--------:|:-------:|:-------------------------------------------------------------------------------------------:|
| DAMamba-T | [ImageNet-1K](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-T.pth) |   3x    |  50.4   |   44.8   |   45M   | [model](https://huggingface.co/ltzovo/DAMamba/blob/main/mask_rcnn_DAMamba_tiny_fpn_3x.pth)  |
| DAMamba-S | [ImageNet-1K](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-S.pth) |   3x    |  51.2   |   45.1   |   65M   | [model](https://huggingface.co/ltzovo/DAMamba/blob/main/mask_rcnn_DAMamba_small_fpn_3x.pth) |
| DAMamba-B | [ImageNet-1K](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-B.pth) |   3x    |  51.4   |   45.3   |  105M   | [model](https://huggingface.co/ltzovo/DAMamba/blob/main/mask_rcnn_DAMamba_base_fpn_3x.pth)  |


## Requirements

    pip install -r requirements.txt



## Evaluation
To evaluate Mask R-CNN models with DAMamba backbone on COCO val, you can use the following command:

    bash dist_test.sh <config-file> <checkpoint-path> <gpu-num> --eval bbox segm

For example, to evaluate the DAMamba-T on a single GPU:
    
    bash dist_test.sh ./configs/mask_rcnn_DAMamba_tiny_fpn_1x_coco.py /path/to/checkpoint_file 1 --eval bbox segm
    
For example, to evaluate the DAMamba-T on 8 GPUs:
    
    bash dist_test.sh ./configs/mask_rcnn_DAMamba_tiny_fpn_1x_coco.py /path/to/checkpoint_file 8 --eval bbox segm


## Training
In order to train Mask R-CNN models with DAMamba backbone on the COCO dataset, first, you need to fill in the path of your downloaded pretrained checkpoint in `./configs/<config-file>`. Specifically, change it to:
    
    pretrained=<path-to-checkpoint>, 

After setting up, to train DAMamba on COCO dataset, you can use the following command:
    
    bash dist_train.sh <config-file> <gpu-num> 

For example, to train the DAMamba-T on 8 GPUs, with a total batch-size of 16:

    bash dist_train.sh ./configs/mask_rcnn_DAMamba_tiny_fpn_1x_coco.py 8

## Acknowledgement

The released script for Object Detection with DAMamba is built based on the [MMDetection](https://github.com/open-mmlab/mmdetection) and [timm](https://github.com/huggingface/pytorch-image-models) library.

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