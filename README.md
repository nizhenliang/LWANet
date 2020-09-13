# Attention-Guided Lightweight Network for Real-Time Segmentation of Robotic Surgical Instruments (ICRA2020)
LWANet (Accepted by ICRA2020)

Zhen-Liang Ni, Gui-Bin Bian, Zeng-Guang Hou, Xiao-Hu Zhou, Xiao-Liang Xie, Zhen Li

Paper address: https://arxiv.org/abs/1910.11109

![Image](https://github.com/nizhenliang/LWANet/blob/master/img/lwdnet.png)

   LWANet can segment surgical instruments in real-time while takes little computational costs. Based on 960×544 inputs, its inference speed can reach 39 fps with only 3.39 GFLOPs. Also, it has a small model size and the number of parameters is only 2.06 M. The proposed network is evaluated on two datasets. It achieves state-of-the-art performance 94.10% mean IOU on Cata7 and obtains a new record on EndoVis 2017 with a 4.10% increase on mean IOU.

## Results
Cata7 
![Image](https://github.com/nizhenliang/LWANet/blob/master/img/table1.png)

![Image](https://github.com/nizhenliang/LWANet/blob/master/img/table2.png)

EndoVis 2017
![Image](https://github.com/nizhenliang/LWANet/blob/master/img/table3.png)


## Citation
If you find LWANet useful in your research, please consider citing:

```
@article{ni2019attention,
  title={Attention-guided lightweight network for real-time segmentation of robotic surgical instruments},
  author={Ni, Zhen-Liang and Bian, Gui-Bin and Hou, Zeng-Guang and Zhou, Xiao-Hu and Xie, Xiao-Liang and Li, Zhen},
  journal={arXiv preprint arXiv:1910.11109},
  year={2019}
}
```
