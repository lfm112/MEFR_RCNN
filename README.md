# Faster R-CNN



## 环境配置：
* Python3.6/3.7/3.8
* Pytorch1.7.1(注意：必须是1.6.0或以上，因为使用官方提供的混合精度训练1.6.0后才支持)
* pycocotools(Linux:`pip install pycocotools`; Windows:`pip install pycocotools-windows`(不需要额外安装vs))
* Ubuntu或Centos(不建议Windows)
* 最好使用GPU训练
* 详细环境配置见`requirements.txt`

## 数据集结构：
```
VOCdevkit/
└── VOC2012/
    ├── Annotations/
    │   └── *.xml              # 每张图像对应的标注文件（XML格式），文件名与图像名一致
    ├── ImageSets/
    │   └── Main/
    │       ├── train.txt      # 训练集图像名列表（不含扩展名，每行一个）
    │       └── val.txt   # 训练+验证集图像名列表（可选）
    └── JPEGImages/
        └── *.jpg              # 所有原始图像文件，格式通常为 JPG
```


 
 
## 数据集，本例程使用的是PASCAL VOC2012数据集
* Pascal VOC2012 train/val数据集下载地址：http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
* 如果不了解数据集或者想使用自己的数据集进行训练，请参考我的bilibili：https://b23.tv/F1kSCK
* 使用ResNet50+FPN以及迁移学习在VOC2012数据集上得到的权重: 链接:https://pan.baidu.com/s/1ifilndFRtAV5RDZINSHj5w 提取码:dsz8

## 训练方法
* 确保提前准备好数据集
* 直接使用train.py训练脚本


## 注意事项
* 验证与可视化代码正在整理，稍后上传
