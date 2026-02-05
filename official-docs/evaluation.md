# Evaluation

Source URL: https://www.codabench.org/competitions/12566/

## Evaluation

The MAVIC-T challenge employs a rigorous evaluation framework designed to assess image translation submissions across four distinct tasks, aiming to achieve highfidelity translations. The challenge focuses on the following translation tasks: SAR→EO, SAR→RGB, SAR→IR, and RGB→IR.

Submissions are evaluated based on their performance in
each task using a composite score derived from three carefully selected metrics:

- **LPIPS (Learned Perceptual Image Patch Similarity)**: This metric, based on the VGG-16 architecture, measures perceptual similarity using deep feature representations, aligning with human visual perception.

- **FID (Frechet Inception Distance)**: ´ Utilizing a pretrained InceptionV3 network, FID quantifies the dissimilarity between the feature distributions of generated and target images, providing insights into visual fidelity and feature similarity.

- **L1 Norm:** This metric calculates the pixel-wise absolute difference between target and generated images, ensuring structural integrity and content accuracy.

These metrics are chosen to comprehensively evaluate
image translation quality, with LPIPS focusing on perceptual accuracy, FID on distributional similarity, and L1 on
content and structural integrity. This strategy aims to minimize artifacts and ensure generated images exhibit highresolution details and structural coherence, aligning with
the target domain

The evaluation process calculates each metric’s score
across all four tasks, followed by task-specific normalization to scale values between 0 and 1:

- **L1 Norm**: Pixel values are adjusted to fit within the desired range. 
- **LPIPS**: Output weights are scaled for normalization. 
- **FID**: A weighted arctan activation function is used to normalize scores, balancing each metric’s influence.

The final score for each task is the average of these normalized metrics:

 Task score = (2/π *arctan*(FID) + LPIPS +L1) / 3

The overall submission performance is then determined by
averaging the Task Scores across the four translation tasks:

 Overall score = (SAR2EO + SAR2RGB + SAT2IR + RGB2IR) / 4

Finally, a score penalty of 1 is added for each unattempted domain. This is done to encourage generalizability, without being too suppressing any models or techniques that may have exceptional results focusing on a smaller number of domains.
