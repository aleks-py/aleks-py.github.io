---
layout: about
title: about
permalink: /
subtitle:



news: false  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false  # includes social icons at the bottom of the page
---
# **Text to Video Generation (Make-A-Video and Imagen Video) Explained**

![image](assets/img/1.jpg)

In fall of 2022, both Meta and Google released novel text-to-video generation models that output impressive video-format content. These networks build off of recent advancements in text-to-image modeling using stable diffusion (like Dall-E and Imagen). Meta’s Make-A-Video is capable of five second 768x768 clips at variable frame rates while Google’s Imagen Video can produce 1280×768 videos at 24 fps. These generators are capable of creating high-resolution, photorealistic and stylistic content of impossible scenarios. These networks can be powerful tools for artists and creators as well as the basis for predicting future frames of a video.

## History of Text-to-Video
Video generation has progressed rapidly in the past decade. Early video generation models focused on simple, specific domains and next frame prediction with **deterministic autoregressive** methods [CDNA, PredRNN]. Later video prediction models incorporated stochasticity [SV2P]. Another line of work uses generative models, namely **GANs** to synthesize complex scenes without a first frame [VGAN, TGAN]. More recently, text-to-video has been approached with **VQVAEs** to learn latent representations of video frames and then **autoregressive transformers** to generate video samples [NUWA & GODIVA]. This technique allows for open-domain video generation, but frames are still generated one at a time chronologically, resulting in potentially poor text-video alignment. CogVideo adjusts the training procedure to fix alignment (discussed below) and uses pre-trained text-to-image weights [CogVideo]. Make-A-Video and Imagen Video both use **diffusion models**, which we will discuss in the next section.

Make-A-Video and Imagen Video have come out just six months after Open-AI’s DALL-E2. Text to video is a much harder problem than text to image because we don’t have access to as many labeled text-image pairs. Therefore, all the models we highlight take advantage of starting from an existing text-to-image model with pre-trained or frozen weights. Moreover, beyond just generating pixels, the network has to predict how they will all evolve over time to coherently complete any actions in the text prompt.

In this post, we’ll break down the building blocks to make text-to-video generation possible, starting from a brief overview of how text to image generators use stable diffusion, how to make the components 3D to incorporate temporal information for video generation, and how to increase the spatial and temporal resolution.

We’ll focus on how these components make up Make-A-Video and Imagen Video, but also touch on CogVideo (an open-source text to image video generator that uses a VQVAE + autoregressive transformers architecture).

## Text-to-Image Generation
Text to Image uses stable diffusion in latent space and a 2D U-Net architecture for image generation (see link for more details).

### **What is latent space?**
First let’s explain how **auto-encoders** work:

Here an input image is encoded into a lower-dimensional latent space representation and a decoder can reconstruct the image. This network is trained by comparing the input to the reconstructed output.

Sampling within the latent space distribution allows us to generate realistic outputs.

We train our network using **stable diffusion**.

### **How does stable diffusion work in latent space?**
During the forward process, we create a dataset by incrementally adding more noise to our latent variables. In the reverse process, we train a model with a **U-Net architecture** to iteratively denoise these latents. This way, we can efficiently generate new images by starting with random noise and end up with a latent that can be decoded into a real image (while conditioning layers on the input text embedding).
