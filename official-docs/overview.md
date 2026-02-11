# Overview

Source URL: https://www.codabench.org/competitions/12566/

# Important Dates

- 2026.01.15 Submission beings
- 2026.03.01 Submission ends
- 2026.03.03 Contribution from participants (top results should provide a brief description of the proposed solution together with an illustration of the architecture; code also should be shared)
- 2026.03.06 Paper submission deadline for entries from the challenge
- 2026.06.03 PBVS workshop and challenges, results and award ceremony (CVPR 2026)

## Challenge overview

Electro-optical (EO) sensors that capture images in the visible spectrum such as RGB and grayscale images, have been most prevalent in the computer vision research area. However, other sensors such as synthetic aperture radar (SAR) can reproduce images from radar signals that in some cases could complement EO sensors when such sensors fail to capture significant information (i.e. weather condition, no visible light, etc).

An ideal automated target recognition system would be based on multi-sensor information to compensate for the individual shortcomings of either of the sensor-based platforms. However, it is currently unclear if/how using EO and SAR data together can improve the performance of automatic target reconition (ATR) systems. Thus, the motivation for this challenge is to understand if and how data from one modality can improve the learning process for the other modality and vice versa. Ideas from domain adaption, transfer learning or fusion are welcomed to solve this problem.

Jointly with PBVS workshop we have a PBVS challenge on Multi-modal Aerial View Imagery Challenge-C, that is, the task of predicting the class label of an aerial low resolution image based on a set of prior examples of images and their class labels. The challenge uses a new dataset:

- **Translation:** Multi-modal image translation (SAR, EO, IR, and RGB imageru).

 The aim is to obtain a network design / solution capable to produce high quality and high fidelity multi-modal image translations. This problem can be framed as a conditioned image generation. We provide a multi-modal dataset built on spatially aligned SAR-EO, EO-IR, and SAR-IR pairs. We strive to temporally align the data as much as possible. Results will evaluated based on an ensemble of image generation metrics. This includes the L2 Norm, Frechet Inception Distance (FID), and Learned Perceptual Image Patch Similarity (LPIPS). 
 
 We expect every participant to submit a description of their method after the final phase. We will not only score the methods by accuracy but also by novelty and creativity. We reserve the right to review the participants code and replicate their results in order to adhere to honor code guidelines.


The top ranked participants will be awarded and invited to follow the CVPR submission guide for workshops to describe their solution and to submit to the associated PBVS workshop at CVPR 2026.

## Paper Submission

Participants are encouraged to submit a paper discussing their method. Submissions should be submitted in the CVPR format. Please submit your paper through: https://pbvs-workshop.github.io/submission.html.

## Competition

The training data is already made available to the registered participants.

## Provided Resources

 

 



- **Check out our solutions from last year:** The results from last year's competition are discussed in greater detail in our paper.
