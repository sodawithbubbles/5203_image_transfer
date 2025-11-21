# Milestone Report: Generative Image Stylization using Stable Diffusion and LoRA

**Author(s):** Soda with Bubbles

## Introduction


This project aims to develop a controllable image stylization framework using state-of-the-art generative AI models. By leveraging **Stable Diffusion** in combination with **LoRA (Low-Rank Adaptation)**, we plan to enhance the flexibility and control over the style transfer process. The goal is to offer a user-friendly interface for transforming real-world photographs into various artistic styles. Unlike traditional style transfer methods, our approach enables **fine-tuning models on diverse styles** efficiently without retraining from scratch.

The final outcome will be evaluated based on the effectiveness of style preservation and the ability to tailor images for different applications such as game asset creation and digital art.

## Problem Statement

We are working on the problem of **image stylization**, where we seek to transform input images into different artistic styles. Specifically, the dataset consists of various images labeled with distinct styles, such as anime, watercolor, oil painting, and more. The task requires us to train a model that can learn these styles and apply them to new images efficiently.

The dataset used for training consists of multiple style categories. Each category contains a set of images along with corresponding prompts. The challenge lies in maintaining high-quality content while applying the desired artistic styles to the input image. Our expected result is to achieve **high-quality stylized images** with accurate style preservation, while allowing users to control the strength of the applied style.

The evaluation will be based on:
- **Qualitative assessment** (visual inspection of generated images),
- **Quantitative metrics** (e.g., style consistency and content preservation), and
- **User studies** (feedback from non-expert users about the usability and results).

## Technical Approach

To address the problem, we follow these steps:

1. **Model Choice:**
   - We use **Stable Diffusion** for generating images, utilizing pre-trained weights to start the process.
   - We will apply **LoRA** for efficient style-specific fine-tuning, enabling us to adapt the model to each specific artistic style.

2. **Data Preprocessing:**
   - The dataset consists of images and corresponding textual prompts for each style.
   - Each style is treated as a separate training task, with different subsets of data (e.g., anime images for the anime style).

3. **Training Approach:**
   - For each style, we fine-tune Stable Diffusion using the LoRA approach, where the model weights are adapted without requiring full model retraining.
   - We apply **gradient accumulation** to optimize training for small batch sizes, ensuring stable updates for each iteration.

4. **Evaluation:**
   - Generated images will be compared to original photographs to assess the **content preservation** and **style consistency**.
   - We also plan to conduct user testing to evaluate how well users can customize images based on different style strengths.

## Intermediate/Preliminary Results

At the time of this milestone, we have successfully:
1. Implemented the **Stable Diffusion + LoRA training pipeline**.
2. Trained models for the **anime** and **3D style** datasets.
3. Obtained initial validation results, which include stylized images that show strong style adherence, but further fine-tuning is needed to refine the outputs.

Results to date suggest that while the model can apply basic styles effectively, further training and the addition of more styles will be needed to optimize quality and consistency.

Future plans include:
- Training additional styles (e.g., oil painting, sketch).
- Testing and evaluating the integration of these models into a **WebUI** for user-friendly access.
- Incorporating **LyCORIS** for better control over style strength and flexibility.

## Next Steps
- Implement a WebUI interface to showcase the trained models.
- Extend training to more diverse styles and enhance fine-tuning techniques.
- Perform extensive testing and user feedback to further improve model output.

