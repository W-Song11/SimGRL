---
layout: project_page
permalink: /

title: <p style="font-size:40px">A Simple Framework for Generalization in Visual RL under Dynamic Scene Perturbations</p> <p style="font-size:38px">NeurIPS 2024</p>
authors:
    Wonil Song<sup>1</sup>, Hyesong Choi<sup>2</sup>, Kwanghoon Sohn<sup>1</sup>, Dongbo Min<sup>2</sup>
affiliations:
    <p style="font-size:18px"><sup>1</sup>Yonsei University, Seoul, Korea, <sup>2</sup>Ewha Womans University, Seoul, Korea</p>
paper: https://openreview.net/pdf?id=0AumdfLzpK
video: #https://www.youtube.com/results?search_query=turing+machine
code: https://github.com/W-Song11/SimGRL-Code
data: #https://huggingface.co/docs/datasets
---

<!-- Using HTML to center the abstract -->
<div class="columns is-centered has-text-centered">
    <div class="column is-four-fifths">
        <h2>Abstract</h2>
        <div class="content has-text-justified">
In the rapidly evolving domain of vision-based deep reinforcement learning (RL),
a pivotal challenge is to achieve generalization capability to dynamic environmental
changes reflected in visual observations. Our work delves into the intricacies
of this problem, identifying two key issues that appear in previous approaches
for visual RL generalization: (i) imbalanced saliency and (ii) observational
overfitting. Imbalanced saliency is a phenomenon where an RL agent
disproportionately identifies salient features across consecutive frames in a frame
stack. Observational overfitting occurs when the agent focuses on certain background
regions rather than task-relevant objects. To address these challenges, we
present a simple yet effective framework for generalization in visual RL (SimGRL)
under dynamic scene perturbations. First, to mitigate the imbalanced saliency
problem, we introduce an architectural modification to the image encoder to
stack frames at the feature level rather than the image level. Simultaneously,
to alleviate the observational overfitting problem, we propose a novel technique
called shifted random overlay augmentation, which is specifically designed
to learn robust representations capable of effectively handling dynamic visual
scenes. Extensive experiments demonstrate the superior generalization capability
of SimGRL, achieving state-of-the-art performance in benchmarks including the
DeepMind Control Suite.
        </div>
    </div>
</div>

---

## Motivation
<p align="center">
<img src="./static/image/fig1.PNG" align="center" width="80%">
</p>

## Overview
TBD

<p align="center">
<img src="./static/image/overview.png" align="center" width="80%">
</p>

*Figure 1: Overview.*

## Results
DMControl-GB

<p align="center">
<img src="./static/image/fig_result.PNG" align="center" width="90%">
</p>

## Demonstrations
TBD
<p align="center">
<img src="./static/image/walker_walk_train.gif" align="center" width="12%">
<img src="./static/image/arrow.png" align="center" width="6%">
<img src="./static/image/walker_walk_test1.gif" align="center" width="12%">
<img src="./static/image/walker_walk_test2.gif" align="center" width="12%">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="./static/image/walker_stand_train.gif" align="center" width="12%">
<img src="./static/image/arrow.png" align="center" width="6%">
<img src="./static/image/walker_stand_test1.gif" align="center" width="12%">
<img src="./static/image/walker_stand_test2.gif" align="center" width="12%">
</p>

<p align="center">
<img src="./static/image/cartpole_train.gif" align="center" width="12%">
<img src="./static/image/arrow.png" align="center" width="6%">
<img src="./static/image/cartpole_test1.gif" align="center" width="12%">
<img src="./static/image/cartpole_test2.gif" align="center" width="12%">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="./static/image/cup_catch_train.gif" align="center" width="12%">
<img src="./static/image/arrow.png" align="center" width="6%">
<img src="./static/image/cup_catch_test1.gif" align="center" width="12%">
<img src="./static/image/cup_catch_test2.gif" align="center" width="12%">
</p>

<p align="center">
<img src="./static/image/finger_train.gif" align="center" width="12%">
<img src="./static/image/arrow.png" align="center" width="6%">
<img src="./static/image/finger_test1.gif" align="center" width="12%">
<img src="./static/image/finger_test2.gif" align="center" width="12%">
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="./static/image/cheetah_train.gif" align="center" width="12%">
<img src="./static/image/arrow.png" align="center" width="6%">
<img src="./static/image/cheetah_test1.gif" align="center" width="12%">
<img src="./static/image/cheetah_test2.gif" align="center" width="12%">
</p>

Robotic Manipulation

<p align="center">
<img src="./static/image/reach/1_train.gif" align="center" width="10%">
<img src="./static/image/reach/2_train.gif" align="center" width="10%">
&nbsp;
<img src="./static/image/arrow.png" align="center" width="6%">
&nbsp;
<img src="./static/image/reach/1_test1.gif" align="center" width="10%">
<img src="./static/image/reach/2_test1.gif" align="center" width="10%">
&nbsp;&nbsp;
<img src="./static/image/reach/1_test2.gif" align="center" width="10%">
<img src="./static/image/reach/2_test2.gif" align="center" width="10%">
&nbsp;&nbsp;
<img src="./static/image/reach/1_test3.gif" align="center" width="10%">
<img src="./static/image/reach/2_test3.gif" align="center" width="10%">
</p>

<p align="center">
<img src="./static/image/pegbox/1_train.gif" align="center" width="10%">
<img src="./static/image/pegbox/2_train.gif" align="center" width="10%">
&nbsp;
<img src="./static/image/arrow.png" align="center" width="6%">
&nbsp;
<img src="./static/image/pegbox/1_test1.gif" align="center" width="10%">
<img src="./static/image/pegbox/2_test1.gif" align="center" width="10%">
&nbsp;&nbsp;
<img src="./static/image/pegbox/1_test2.gif" align="center" width="10%">
<img src="./static/image/pegbox/2_test2.gif" align="center" width="10%">
&nbsp;&nbsp;
<img src="./static/image/pegbox/1_test3.gif" align="center" width="10%">
<img src="./static/image/pegbox/2_test3.gif" align="center" width="10%">
</p>


## Citation
```
@inproceedings{
anonymous2024a,
title={A Simple Framework for Generalization in Visual {RL} under Dynamic Scene Perturbations},
author={Anonymous},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=0AumdfLzpK}
}
```
