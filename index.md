---
layout: project_page
permalink: /

title: A Simple Framework for Generalization in Visual RL under Dynamic Scene Perturbations
authors:
    Wonil Song^1, Hyesong Choi^2, Kwanghoon Sohn^1, Dongbo Min^1
affiliations:
    1. Yonsei University, Seoul, Korea
    2. Ewha Womans University, Seoul, Korea
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

## Background
The paper "On Computable Numbers, with an Application to the Entscheidungsproblem" was published by Alan Turing in 1936. In this groundbreaking paper, Turing introduced the concept of a universal computing machine, now known as the Turing machine.

## Objective
Turing's main objective in this paper was to investigate the notion of computability and its relation to the Entscheidungsproblem (the decision problem), which is concerned with determining whether a given mathematical statement is provable or not.


## Key Ideas
1. Turing first presented the concept of a "computable number," which refers to a number that can be computed by an algorithm or a definite step-by-step process.

![Turing Machine](/static/image/Turing_machine.png)

*Figure 1: A representation of a Turing Machine. Source: [Wiki](https://en.wikipedia.org/wiki/Turing_machine).*

## Table: Comparison of Computable and Non-Computable Numbers

| Computable Numbers | Non-Computable Numbers |
|-------------------|-----------------------|
| Rational numbers, e.g., 1/2, 3/4 | Transcendental numbers, e.g., π, e |
| Algebraic numbers, e.g., √2, ∛3 | Non-algebraic numbers, e.g., √2 + √3 |
| Numbers with finite decimal representations | Numbers with infinite, non-repeating decimal representations |

He used the concept of a universal Turing machine to prove that the set of computable functions is recursively enumerable, meaning it can be listed by an algorithm.

## Significance
Turing's paper laid the foundation for the theory of computation and had a profound impact on the development of computer science. The Turing machine became a fundamental concept in theoretical computer science, serving as a theoretical model for studying the limits and capabilities of computation. Turing's work also influenced the development of programming languages, algorithms, and the design of modern computers.

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
> Note: Website template borrowed from Nerfies made by Keunhong Park.
