# Pytorch Version, here we implement VGG, Resnet, Alexnet, etc

Please go to: https://github.com/ccny-ros-pkg/concreteIn_inpection_VGGF/ for more details

## System Requirements
Tested on Ubuntu 16.04, Pytorch 0.4.0, CUDA 8.0 0r 9.0
>- require python3.5

>- pytorch 0.4.0 or above

## Concrete Inpection Dataset and Baseline @ [CCNY Robotics Lab](https://ccny-ros-pkg.github.io/)

Authors: [Liang Yang](https://ericlyang.github.io/),  [Bing LI](https://robotlee2002.github.io/), [Wei LI](http://ccvcl.org/~wei/), Zhaoming LIU, Guoyong YANG, [Jizhong XIAO](http://www-ee.ccny.cuny.edu/www/web/jxiao/jxiao.html)


CCNY Concrete Structure Spalling and Crack database (CSSC) that aims to assist the civil inspection of performing in an automatical approach. In the first generate of our work, we mainly focusing on dataset creation and prove the concepts of innovativity. We provide the first complete the detailed dataset for concrete spalling and crack defects witht the help from Civil Engineers, where we also show our sincere thanks to the under-graduate student at Hostos Community College for their effort on data labeling. For our experiments, we deliever an UAV to perform field data-collection and inspection, and also perform a 3D semantic metric resconstructiont.


### If you find this could be helpful for your project, please cite the following related papers:


@INPROCEEDINGS{8575365,
author={L. {Yang} and B. {Li} and W. {Li} and B. {Jiang} and J. {Xiao}},
booktitle={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
title={Semantic Metric 3D Reconstruction for Concrete Inspection},
year={2018},
volume={},
number={},
pages={1624-16248},
ISSN={2160-7516},
month={June},}

@inproceedings{yangi2018wall,
  title={Wall-climbing robot for visual and GPR inspection},
  author={Yang{\'\i}, Liang and Yang, Guoyong and Liu, Zhaoming and Chang, Yong and Jiang, Biao and Awad, Youssef and Xiao, Jizhong},
  booktitle={2018 13th IEEE Conference on Industrial Electronics and Applications (ICIEA)},
  pages={1004--1009},
  year={2018},
  organization={IEEE}
}


[IROS 2017] Liang YANG, Bing LI, Wei LI, Zhaoming LIU, Guoyong YANG,Jizhong XIAO (2017). Deep Concrete Inspection Using Unmanned Aerial Vehicle Towards CSSC Database. 2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), One Page Abstract, [PDF](https://ericlyang.github.io/img/IROS2017/IROS2017.pdf).


[ROBIO 2017] Liang YANG, Bing LI, Wei LI, Zhaoming LIU, Guoyong YANG,Jizhong XIAO (2017). A Robotic System Towards Concrete Structure Spalling And Crack Database. 2017 IEEE Int. Conf. on Robotics and Biomimetics (ROBIO 2017), [Project](https://ericlyang.github.io/project/deepinspection/).


### The under going project

If you are interested in this project, please check the [project link](https://ericlyang.github.io/project/deepinspection/) and our current 3D semantic pixel-level reconstruction [project](https://ericlyang.github.io/project/robot-inspection-net/). Also, you can shoot [Liang Yang](https://ericlyang.github.io/) an email any time for other authors.

# How to implement
## training
1) First download the data from: https://github.com/ccny-ros-pkg/concreteIn_inpection_VGGF/

2) Put the data in folder SPallData

3) run script:
> - python3 main.py

## test
1) Change the model directory of the model

2) run script:
> - python3 demo_test.py


## How to change model

In file model.py, you can change any model you prefer:

>- #from models import resnet as resnet
>-from models import resnet as resnet
>-#from models import vgg as resnet
>-#from models.alexnet import *
>-def generate_model(opt):
>-    model = resnet.resnet34(pretrained = True, num_classes=opt.n_classes)

## define epoch, batch size, output folder

Please check file opts.py


# Example result:
![](https://github.com/ccny-ros-pkg/pytorch_Concrete_Inspection/blob/master/image_and_results/output/175.png)
![](https://github.com/ccny-ros-pkg/pytorch_Concrete_Inspection/blob/master/image_and_results/output/329.png)
![](https://github.com/ccny-ros-pkg/pytorch_Concrete_Inspection/blob/master/image_and_results/output/596.png)
