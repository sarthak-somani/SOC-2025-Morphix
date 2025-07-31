# Morphix: Sculpting Faces in Real-Time with Latent Spaces

### WnCC Seasons of Code 2025 - Mid-Term Submission

---

## Project Overview

Morphix is a project that explores the fascinating capabilities of Generative Adversarial Networks (GANs) to manipulate and generate hyper-realistic human faces. By navigating and manipulating the latent spaces of a pre-trained StyleGAN2 model, my project aims to provide an intuitive way to "sculpt" facial features in real-time. This report outlines the foundational concepts learned and my progress made by the mid-term point of SoC 2025.

---

## 1. Understanding Generative Adversarial Networks (GANs)

A Generative Adversarial Network (GAN) is a class of machine learning frameworks where two neural networks operate simultaneously. The two networks are called the Generator and the Discriminator.

-   **The Generator (G):** Its goal is to create data that looks real. It takes a random noise vector (from the latent space) as input and outputs data (in our case, an image of a face).
-   **The Discriminator (D):** Its goal is to distinguish between real data (from the training set) and fake data (from the Generator).

These two networks are trained simultaneously. The Generator gets better at creating convincing fakes, while the Discriminator gets better at spotting them. This adversarial process continues until the Generator produces images that are so realistic they can fool the Discriminator.

---

## 2. Introduction to StyleGANs

StyleGAN is a revolutionary generator architecture introduced by NVIDIA. It builds upon the traditional GAN framework but introduces significant changes to the generator network, allowing for more explicit control over the visual "style" of the generated image at different levels of detail.

Key innovations include:

-   **Mapping Network:** Instead of feeding the initial random noise vector (from the **Z space**) directly to the generator, StyleGAN first maps it to an intermediate latent space, **W**. This space is "disentangled," meaning its dimensions tend to correspond to distinct, high-level attributes of the face (like hair style, age, or glasses) rather than being abstract.
-   **Style-Based Synthesis:** The intermediate latent code from the **W space** is used to control the "style" of the generator's output at each resolution layer. This allows for fine-grained control over features, from coarse aspects like head pose to fine details like skin texture.

For this project, we are using **StyleGAN2-ADA**, an official PyTorch implementation from NVIDIA that is optimised for training with limited data.

---

## 3. Project Implementation and Progress

### 3.1. Framework and Model

-   **Repository:** We have set up our development environment by cloning NVIDIA's official `stylegan2-ada-pytorch` repository.
-   **Pre-trained Model:** To leverage the power of StyleGAN without training a model from scratch (which is computationally very expensive), we are using the pre-trained **FFHQ (Flickr-Faces-HQ)** model. This model was trained by NVIDIA on a high-quality dataset of 70,000 human faces and can generate images at a resolution of 1024x1024.

### 3.2. Exploring Latent Spaces

A core part of this project is understanding and utilising StyleGAN's latent spaces.

-   **Z Space:** This is the initial 512-dimensional input space. It's a standard normal distribution from which we sample random vectors. While it's the source of randomness, its dimensions are highly entangled, making it difficult to perform intuitive edits directly.
-   **W Space:** This is the 512-dimensional intermediate latent space produced by the mapping network. The mapping from Z to W is a learned, non-linear transformation. The **W space** is significantly more disentangled, meaning that manipulating vectors in this space often results in more coherent and understandable changes in the final image (e.g., changing age or expression).
-   **W+ Space:** This is the most expressive latent space. Instead of using a single vector from W to control all layers of the synthesis network, **W+** uses a different W vector for each of the 18 layers. This gives maximum control, allowing for very specific, localised edits, as each layer controls features at a different scale.

### 3.3. Generating a Base Image

Our initial work has focused on the fundamental process of generating an image from a random latent vector. The steps are as follows:

1.  **Sample from Z Space:** A random latent vector of shape `(1, 512)` is generated using `np.random.randn()`. This vector serves as the seed for our unique face.
2.  **Map Z to W/W+:** This seed vector is passed through the StyleGAN mapping network to produce a corresponding vector in the more disentangled **W space**. We can then expand this into the **W+ space** to have per-layer control.
3.  **Synthesize the Image:** The **W+** vector is fed into the synthesis network, which progressively generates the final 1024x1024 pixel image of a human face.

This process forms the basis for all future work on this project. By understanding how to generate a face, we can now move on to manipulating the latent vectors to sculpt the face in real-time.
