<p align="center">
<h1 align="center">UDiFF: Generating Conditional Unsigned Distance Fields with Optimal Wavelet Diffusion(CVPR 2024)</h1>
<p align="center">
    <a href="https://junshengzhou.github.io/"><strong>Junsheng Zhou*</strong></a>
    ·
    <a href="https://weiqi-zhang.github.io/UDiFF/"><strong>Weiqi Zhang*</strong></a>
    ·
    <a href="https://mabaorui.github.io/"><strong>Baorui Ma</strong></a>
    ·
    <a href="https://dblp.org/pid/261/1098.html"><strong>Kanle Shi</strong></a>
    ·
    <a href="https://yushen-liu.github.io/"><strong>Yu-Shen Liu</strong></a>
    ·
    <a href="https://h312h.github.io/"><strong>Zhizhong Han</strong></a>
</p>
<p align="center"><strong>(* Equal Contribution)</strong></p>
<h3 align="center"><a href="">Paper</a> | <a href="https://weiqi-zhang.github.io/UDiFF/">Project Page</a></h3>
<div align="center"></div>
</p>
<p align="center">
    <img src="figs/mainfig.png" width="780" />
</p>
We will release the code of the paper <a href="">UDiFF: Generating Conditional Unsigned Distance Fields with Optimal Wavelet Diffusion</a> in this repository.

## Abstract

<p>
            Diffusion models have shown remarkable results for image generation, editing and inpainting. Recent works explore diffusion models for 3D shape generation with neural implicit functions, i.e., signed distance function and occupancy function. However, they are limited to shapes with closed surfaces, which prevents them from generating diverse 3D real-world contents containing open surfaces. 
            In this work, we present UDiFF, a 3D diffusion model for unsigned distance fields (UDFs) which is capable to generate textured 3D shapes with open surfaces from text conditions or unconditionally. Our key idea is to generate UDFs in spatial-frequency domain with an optimal wavelet transformation, which produces a compact representation space for UDF generation. Specifically, instead of selecting an appropriate wavelet transformation which requires expensive manual efforts and still leads to large information loss, we propose a data-driven approach to learn the optimal wavelet transformation for UDFs. We evaluate UDiFF to show our advantages by numerical and visual comparisons with the latest methods on widely used benchmarks.  
          </p>

## Method

<p align="center">
  <img src="figs/overview.png" width="780" />
</p>

<p style="margin-top: 30px">
            <b>Overview of UDiFF.</b> <b>(a)</b> We propose a data-driven approach to attain the optimal wavelet transformation for UDF generation. We optimize wavelet filter parameters through the decomposition and inversion by minimizing errors in UDF self-reconstruction. <b>(b)</b> We fix the learned decomposition wavelet parameters and leverage it to prepare the data as a compact representation of UDFs including pairs of coarse and fine coefficient volumes. <b>(c)</b> is the architecture of the generator in diffusion models, where text conditions are introduced with cross-attentions. <b>(d)</b> The diffusion process of UDiFF. We train the generator to produce coarse coefficient volumes from random noises guided by input texts and train the fine predictor to predict fine coefficient volumes from the coarse ones. Follow the <b><span style="color: green;">green</span></b> arrows for inference, we start from a random noise and an input text to leverage the trained generator to produce a coarse coefficient volume. The trained fine predictor then predicts the fine coefficient volume. Together with the coarse one, we recover the UDFs with the fixed pre-optimized inversion wavelet filter parameters. Finally, we extract surfaces from UDFs and further texture them with the guiding text.
          </p>

## Visualization Results

### Category Conditional Generations

<img src="./figs/Video_UDiFF.gif" class="center">

Outfit Designs with UDiFF Garment Generations

<img src="./figs/Video_UDiFF2.gif" class="center">


## Generation Results

### DeepFashion3D

​        <img src="./figs/cate1.png" class="center">
​        <img src="./figs/cate2.png" class="center">

<center>Category conditional generations.</center>

### ShapeNet dataset       <img src="./figs/un_condition_shapenet.png" class="center">

<center>Unconditional generations.
</center>

## Comparison Results

<img src="./figs/deepfashion2.png" class="center">

<img src="./figs/comp2.png" class="center">


## Citation
If you find our code or paper useful, please consider citing

    @inproceedings{udiff,
        title={UDiFF: Generating Conditional Unsigned Distance Fields with Optimal Wavelet Diffusion},
        author={Zhou, Junsheng and Zhang, Weiqi and Ma, Baorui and Shi, Kanle and Liu, Yu-Shen and Han, Zhizhong},
        booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
        year={2024}
    }
