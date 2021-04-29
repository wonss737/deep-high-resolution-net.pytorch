# HRNET with Skeletal heatmap

## Introduction
This is an pytorch implementation HRNet with skeletal heatmap. I am interested in semantic information of feature maps with each resolution in HRNet. So, I make ground truth heatmap of joints(default), skeletal, upper/lower and full body. Then, train the model while each predicted feature maps are trained with aforementioned GT heatmaps. The algorithm used to create heatmap referred to [Human Pose Estimation Using Skeletal Heatmaps
](https://ieeexplore.ieee.org/document/9306241). Implementation is only for COCO and HRNet (No ResNet & No MPII).</br>

![Illustrating the architecture of the proposed HRNet](figures/hrnet_with_skeletal_heatmap.jpg.png)
## Main Results

### Results on COCO val2017 with detector having human AP of 56.4 on COCO val2017 dataset
| Arch               | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |    AR | AR .5 | AR .75 | AR (M) | AR (L) |
|--------------------|------------|---------|--------|-------|-------|--------|--------|--------|-------|-------|--------|--------|--------|
| **pose_hrnet_w48** |    384x288 | 65.4M   |   33.3 | pose_hrnet | 0.710 | 0.915 | 0.793 | 0.683 | 0.757 | 0.741 | 0.923 | 0.811 | 0.708 | 0.792 |

### Note:
- Flip test is used.
- Person detector has person AP of 56.4 on COCO val2017 dataset.
- pose_resnet_[50,101,152] is our previous work of [*Simple Baselines for Human Pose Estimation and Tracking*](http://openaccess.thecvf.com/content_ECCV_2018/html/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.html).
- GFLOPs is for convolution and linear layers only.

### Visualization

#### Visualizing ground truth on COCO training
Skeletal Heatmap</br>
<img src="figures/example/2_skeletal.jpg" height="215"></br>
Upper/Lower body Heatmap</br>
<img src="figures/example/3_upper&lower.jpg" height="215"></br>
Full body Heatmap</br>
<img src="figures/example/4_full.jpg" height="209">
