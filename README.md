


## 🛠️ Getting Started

3. Install dependent packages
   ```
   pip install --upgrade pip
   pip install -r requirements.txt
   cd models/selective_scan && pip install .
   cd models/ops_dcnv3
   sh ./make.sh
   ```

## ✨ Pre-trained Models


<summary> ImageNet-1k Image Classification </summary>
<br>

<div>

|      name       |   pretrain   | resolution | acc@1 | #param | FLOPs |                               download                                |
|:---------------:| :----------: | :--------: |:-----:|:------:|:-----:|:---------------------------------------------------------------------:|
|    DAMamba-T    | ImageNet-1K  |  224x224   | 83.8  |  26M   | 4.8G  | [ckpt](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-T.pth) |        |
| DAMamba-S | ImageNet-1K  |  224x224   | 84.8  |  45M   | 10.3G | [ckpt](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-S.pth) |
| DAMamba-B | ImageNet-1K  |  224x224   | 85.2  |  86M   | 16.3G | [ckpt](https://huggingface.co/ltzovo/DAMamba/blob/main/DAMamba-B.pth) |

</div>




## 📚 Data Preparation

* ImageNet is an image database organized according to the WordNet hierarchy. Download and extract ImageNet train and val images from http://image-net.org/. Organize the data into the following directory structure:
  
  ```
  imagenet/
  ├── train/
  │   ├── n01440764/  (Example synset ID)
  │   │   ├── image1.JPEG
  │   │   ├── image2.JPEG
  │   │   └── ...
  │   ├── n01443537/  (Another synset ID)
  │   │   └── ...
  │   └── ...
  └── val/
      ├── n01440764/  (Example synset ID)
      │   ├── image1.JPEG
      │   └── ...
      └── ...
  ```
* COCO is a large-scale object detection, segmentation, and captioning dataset. Please visit http://cocodataset.org/ for more information, including for the data, paper, and tutorials. [COCO API](https://github.com/cocodataset/cocoapi) also provides a concise and efficient way to process the data.
* ADE20K is composed of more than 27K images from the SUN and Places databases. Please visit https://ade20k.csail.mit.edu/ for more information and see the [GitHub Repository](https://github.com/CSAILVision/ADE20K) for an overview of how to access and explore ADE20K.

## 🚀 Quick Start

* **Image Classification**
  
  To train DAMamba models for classification on ImageNet, use the following commands for different configurations:
  
  ```bash
  cd classification 
  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=8 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp
  ```
  
  To evaluate the performance with pre-trained weights:
  
  ```bash
  cd classification 
  python -m torch.distributed.launch --nnodes=1 --node_rank=0 --nproc_per_node=1 --master_addr="127.0.0.1" --master_port=29501 main.py --cfg </path/to/config> --batch-size 128 --data-path </path/of/dataset> --output /tmp --pretrained </path/of/checkpoint>
  ```
  To test the throughput of model:
  
  ```bash
  cd classification/models 
  python3 benchmark.py --batch-size 128 --model DAMamba_T
  ```

## 🖊️ Citation

```BibTeX
@article{li2025damamba,
  title={DAMamba: Vision State Space Model with Dynamic Adaptive Scan},
  author={Li, Tanzhe and Li, Caoshuo and Lyu, Jiayi and Pei, Hongjuan and Zhang, Baochang and Jin, Taisong and Ji, Rongrong},
  journal={arXiv preprint arXiv:2502.12627},
  year={2025}
}
```

## 💌 Acknowledgments

This project is largely based on [Mamba](https://github.com/state-spaces/mamba), [VMamba](https://github.com/MzeroMiko/VMamba), [Swin-Transformer](https://github.com/microsoft/Swin-Transformer), [InternImage](https://github.com/OpenGVLab/InternImage) and [OpenMMLab](https://github.com/open-mmlab). We are truly grateful for their excellent work.

## 🎫 License
This project is released under the [Apache 2.0 license](LICENSE).

