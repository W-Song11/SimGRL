---
layout: project_page
permalink: /

title: A Simple Framework for Generalization in Visual RL under Dynamic Scene Perturbations
authors:
    Wonil Song<sup>1</sup>, Hyesong Choi<sup>2</sup>, Kwanghoon Sohn<sup>1</sup>, Dongbo Min<sup>2</sup>
affiliations:
    <sup>1</sup>Yonsei University, Seoul, Korea, <sup>2</sup>Ewha Womans University, Seoul, Korea
paper: #https://www.cs.virginia.edu/~robins/Turing_Paper_1936.pdf
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
based on shifted random overlay augmentation, which is specifically designed
to learn robust representations capable of effectively handling dynamic visual
scenes. Extensive experiments demonstrate the superior generalization capability
of SimGRL, achieving state-of-the-art performance in benchmarks including the
DeepMind Control Suite.
        </div>
    </div>
</div>

---

## Motivation
TBD

## Overview
TBD

<p align="center">  <img src="./static/image/overview.png" align="center" width="70%"> <figcaption align="center"></p>

*Figure 1: Overview.*

## Results
TBD

<p align="center">  <img src="./static/image/walker_walk_train.gif" align="center" width="15%">  <img src="./static/image/arrow.png" align="center" width="8%">  <img src="./static/image/walker_walk_test1.gif" align="center" width="15%"> <img src="./static/image/walker_walk_test2.gif" align="center" width="15%"></p>

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
