---
layout: about
title: Building Blocks of Text-to-Video Generation
permalink: /
subtitle:



news: false  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false  # includes social icons at the bottom of the page
---
<!-- # **<span style="font-size:38px;">Building Blocks of Text-to-Video Generation</span>** -->

<!-- ---
layout: about
title: Building Blocks of Text-to-Video Generation
permalink: /
subtitle:



news: false  # includes a list of news items
selected_papers: false # includes a list of papers marked as "selected={true}"
social: false  # includes social icons at the bottom of the page
---
<!-- # **<span style="font-size:38px;">Building Blocks of Text-to-Video Generation</span>** --> -->

### Google's Imagen Video and Meta's Make-a-Video Explained

&nbsp;  
<callout>
Google and Meta have both developed advanced AI networks that can generate new, unseen videos using only simple text prompts. Try clicking through the prompts and compare the results between Google's Imagen Video and Meta's Make-a-Video models:
</callout>
<figure1>
  <iframe height="600px" width="840px" scrolling="No" frameborder="0" hspace="0" vspace="0" src="https://video-gui.onrender.com/"></iframe>
</figure1>

In this post, we dissect and explain the mechanics behind the key building blocks for state-of-the-art Text-to-Video generation. We provide interactive examples of these building blocks and demonstrate the key novelties/differences between two Text-to-Video models: Imagen Video and Make-a-Video. Finally, we summarize by showing how the building blocks fit together into a complete Text-to-Video framework as well as noting the current failure modes and limitations of the models today.
{: style="text-align: justify"}

## **History of Text-to-Video**
Just six months after the release of DALL-E 2, both Meta and Google released novel Text-to-Video generation models that output impressive video-format content. These networks build off of recent advancements in Text-to-Image modeling using stable diffusion (like DALL-E [[1]](https://arxiv.org/pdf/2102.12092) and Imagen [[2]](https://arxiv.org/pdf/2205.11487)). Meta’s Make-A-Video [[3]](https://arxiv.org/pdf/2209.14792) is capable of five second 768x768 clips at variable frame rates while Google’s Imagen Video [[4]](https://arxiv.org/pdf/2210.02303) can produce 1280×768 videos at 24 fps. Rather than training strictly on text-video pair datasets, both Imagen Video and Make-a-Video leverage the massive text-image pair databases to construct video from pretrained Text-to-Image generation models. These Text-to-Video generators are capable of creating high-resolution, photorealistic and stylistic content of impossible scenarios. Networks such as these can be powerful tools for artists and creators as well as the basis for predicting future frames of a video.
{: style="text-align: justify"}

Video generation has progressed rapidly in the past decade. Early video generation models focused on simple, specific domains and next frame prediction with **deterministic autoregressive** methods (CDNA [[5]](https://proceedings.neurips.cc/paper/2016/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf), PredRNN [[6]](https://papers.nips.cc/paper/2017/file/e5f6ad6ce374177eef023bf5d0c018b6-Paper.pdf)). Later video prediction models incorporated stochasticity (SV2P [[7]](https://openreview.net/pdf?id=rk49Mg-CW)). Another line of work uses generative models, namely **GANs**, to synthesize complex scenes without a first frame (VGAN [[8]](https://arxiv.org/pdf/1611.01799.pdf), TGAN [[9]](https://arxiv.org/pdf/1611.06624)). More recently, Text-to-Video has been approached with **VQVAEs** to learn latent representations of video frames and then **autoregressive transformers** to generate video samples (GODIVA [[10]](https://arxiv.org/pdf/2104.14806), NUWA [[11]](https://arxiv.org/pdf/2111.12417)). This technique allows for open-domain video generation, but frames are still generated one at a time chronologically, resulting in potentially poor text-video alignment. CogVideo [[12]](https://arxiv.org/pdf/2205.15868) adjusts the training procedure to fix alignment (discussed below) and uses pre-trained Text-to-Image weights. Make-A-Video and Imagen Video both use **diffusion models** (VDM [[13]](https://openreview.net/pdf?id=2LdBqxc1Yv)), which we will discuss in the next section.
{: style="text-align: justify"}

Make-A-Video and Imagen Video have come out just six months after Open-AI’s DALL-E 2. Text-to-Video is a much harder problem than Text-to-Image because we don’t have access to as many labeled text-video pairs. Therefore, all the models we highlight take advantage of starting from an existing Text-to-Image model with pre-trained or frozen weights. Moreover, beyond just generating pixels, the network has to predict how they will all evolve over time to coherently complete any actions in the text prompt.
{: style="text-align: justify"}

<figure>
  <img src="assets/img/2023-02-01-text2vid/timeline1.png" width="670" />
  <figcaption>Figure 1. Timeline of video generation and prediction techniques. The modern-day Text-to-Video generators each leverage the capabilities of massively pre-trained Text-to-Image networks, such as Imagen and DALL-E 2.</figcaption>
</figure>
&nbsp;  

We’ll break down the building blocks to make Text-to-Video generation possible, starting from a brief overview of how Text-to-Image generators use stable diffusion, how to make the components 3D to incorporate temporal information for video generation, and how to increase the spatial and temporal resolution. We focus on how these components make up Make-A-Video and Imagen Video, but also touch on CogVideo (an open-source Text-to-Image video generator that uses a VQVAE + autoregressive transformers architecture).
{: style="text-align: justify"}

<figure>
  <img src="assets/img/2023-02-01-text2vid/T2V1.png" width="600" />
  <figcaption>Figure 2. A simplified foundational network diagram for Text-to-Video generation that leverages pretrained Text-to-Image encodings. These encodings are trained to be decoded into image "batches", e.g., videos, that get upsampled spatially and temporally to have higher framerates and higher resolution. Scroll to Figure 9 to see the building blocks in more detail and how they fit together.</figcaption>
</figure>
&nbsp;  


## **Text-to-Image Generation**
##### **What is latent space?**
Text-to-Image generation uses stable diffusion in latent space and a 2D U-Net architecture for image generation (see link for more details). First let’s explain how **auto-encoders** work:
{: style="text-align: justify"}

<figure>
  <img src="assets/img/2023-02-01-text2vid/autoencoder1.png" width="500" />
  <figcaption>Figure 3. Demonstration of an autoencoder network. Images get compressed into lower-dimensional embeddings that are later decoded and "reconstructed." Training this autoencoder network learns weights for the decoder, such that when randomly sampled embeddings from Z are passed through the decoder, D, brand new images unseen by the network are generated.</figcaption>
</figure>
&nbsp;

Here an input image is encoded into a lower-dimensional latent space representation and a decoder can reconstruct the image. This network is trained on the [Frey Faces Dataset](https://cs.nyu.edu/~roweis/data.html) by comparing the input to the reconstructed output. Sampling within the latent space distribution allows us to generate realistic outputs.
{: style="text-align: justify"}

&nbsp;  
<callout>

Try it out yourself! We’ve trained an autoencoder with a two-dimensional latent space embedding. The two latents happen to correspond to expression and pose (the two variables changed in the dataset). Slowly Drag your cursor through latent space to change the values of the latent variable, Z, and see how the reconstructed image from the decoder changes (dragging too quickly will buffer image generation):

</callout>
<figure>
  <iframe height="420px" width="720px" scrolling="No" frameborder="0" hspace="0" vspace="0" src="https://vae-gui.onrender.com/"></iframe>
</figure>

For Text-to-Video generation, the encoding-decoding network is trained using **stable diffusion** [[15]](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf).
{: style="text-align: justify"}

##### **How does stable diffusion work in latent space?**
During the forward process, we create a dataset by incrementally adding more noise to our latent variables. In the reverse process, we train a model with a **U-Net architecture** [[16]](https://arxiv.org/pdf/1505.04597) to iteratively denoise these latents. This way, we can efficiently generate new images by starting with random noise and end up with a latent that can be decoded into a real image (while conditioning layers on the input text embedding).
{: style="text-align: justify"}

&nbsp;  
<figure>
  <img src="assets/img/2023-02-01-text2vid/latents1.png" width="770" />
  <figcaption>Figure 4. Forward and reverse processes in stable diffusion. In the forward process, random noise is progressively added to the latents in order to create a training set. In the reverse process, the noisy latents are iteratively passed through a U-Net model that learns to denoise the latents. Thus, we can pass random noise into this model and generate a new relevant video with the model’s attention layers conditioned to the input.</figcaption>
</figure>
&nbsp;  


The U-Net architecture (which we use as a noise detector) is an **autoencoder**. Downsampling and upsampling is done with convolutional layers. However, because the latent space is lower-dimensional, it’s possible to lose information, meaning that spatial recreation can be imprecise during upsampling. To deal with this, U-Net has **skip connections** that transfer information across the network, bypassing the downsampling and compression.
{: style="text-align: justify"}

<figure>
  <img src="assets/img/2023-02-01-text2vid/unet2.png" width="770" />
  <figcaption>Figure 5. U-Net architecture consists of a convolutional encoder and decoder. Skip connections copy and crop information from downsampling. Attention layers at skip connections help by weighting relevant information. Referenced from U-Net [16].</figcaption>
</figure>
&nbsp;

However, the poor feature representation in the initial layers result in redundant information. To deal with this, we can add **attention layers** at the skip connections to suppress activation in irrelevant regions, reducing the number of redundant features brought across by focusing "attention" only on the most important image features. For Text-to-Image generation, these attention networks also have access to the text embeddings to help condition the attention.
{: style="text-align: justify"}

&nbsp;  
<callout>
Try it out yourself! Click on different points in the first image and see how this would change the weights in the attention layer. The attention highlights similar features to the clicked region. For Text-to-Image and Text-to-Video generation, we include text embeddings to condition attention:
</callout>
<figure>
  <iframe height="400px" width="600px" scrolling="No" frameborder="0" hspace="0" vspace="0" src="https://attn-gui.onrender.com/"></iframe>
</figure>

In the next section we discuss how to modify our convolutional and attention layers to move from image *(2D spatial)* representations to video *(3D)* representations, composed of individual frames *(2D spatial)* + time *(1D temporal)*.

## **Text-to-Video Generation**

##### **How do we extend Text-to-Image to Text-to-Video?**

Text-to-Image generation uses U-Net architecture with 2D spatial convolution and attention layers. For video generation, we need to add a third temporal dimension to the two spatial ones. 3D convolution layers are computationally expensive and 3D attention layers are computationally intractable. Therefore, these papers have their own approaches.
{: style="text-align: justify"}

Make-A-Video creates pseudo 3D convolution and attention layers by stacking a 1D temporal layer over a 2D spatial layer. Imagen Video does spatial convolution and attention for each individual frame, then does temporal attention or convolution across all frames.
{: style="text-align: justify"}

<figure>
  <img src="assets/img/2023-02-01-text2vid/attention1.png" width="770" />
  <figcaption>Figure 6. 3D U-Net components  stacking 1D temporal layers over 2D spatial layers. Make-A-Video has individual pseudo 3D convolutional and attention layers. Imagen video first does spatial processing on each individual frame and then has a  temporal attention layer across frames. The spatial layers can all be pre-trained from Text-to-Image models.</figcaption>
</figure>
&nbsp;

Separating the spatial and temporal operations allows for **building off of existing Text-to-Image models.**
- **CogVideo** freezes all the weights of the spatial layers
- **Make-A-Video** uses pretrained weights for the spatial layers but initializes the temporal layer weights to the identity matrix. This way they can continue tuning all weights with new video data
- **Imagen Video** can jointly train their model with video and image data, doing the latter by masking the temporal connections
{: style="text-align: justify"}

##### **Spatial and Temporal Super Resolution**
The base video decoder creates a fixed number of frames (5 frames for CogVideo, 16 frames for Make-A-Video, and 15 frames for Imagen Video) that need to be upsampled temporally and spatially.
{: style="text-align: justify"}

<figure>
  <img src="assets/img/2023-02-01-text2vid/super_resolution1.png" width="670" />
  <figcaption>Figure 7. Text-to-Video architectures of Imagen Video (top) and Make-A-Video (bottom). Light blue boxes represent temporal upsampling steps and darker blue boxes are for spatial upsampling.</figcaption>
</figure>
&nbsp;

Make-A-Video uses **frame rate conditioning**, meaning they have an additional input that determines the fps in the generated video (unlike how Imagen Video has a fixed frame rate in each stage). During training, this is useful as a form of data augmentation due to the limited dataset of videos. CogVideo also highlights the importance of changing the frame rate in order to retime videos such that an entire action can be encompassed in a fixed video length. For example the action “drinking” is composed of the sub-actions “pick up glass,” “drink,” and “place glass” which need to be performed in that order. If training on videos of a fixed length, changing the frame rate can help ensure text-video alignment.
{: style="text-align: justify"}

**Frame interpolation** for Make-A-Video is done in an autoregressive manner. They fine-tune a spatio-temporal decoder by masking certain frames of a training video and learning to predict them. They train with variable frame-skips and fps conditioning to enable different temporal upsampling rates. The framework is also able to interpolate and extrapolate (extend the beginning or end of a video).
{: style="text-align: justify"}

<figure>
  <img src="assets/img/2023-02-01-text2vid/masking2.png" width="500" />
  <figcaption>Figure 8. Frame masking for frame interpolation (middle) and extrapolation (bottom). For interpolation, variable frames are skipped. For extrapolation, frames at the beginning or end of the video can be masked.</figcaption>
</figure>
&nbsp;

Imagen Video’s approach relies on **cascaded video diffusion models**. They generate entire blocks of frames simultaneously for each network to avoid the artifacts that would result from running super-resolution on independent frames. Each of the 6 super-resolution sub-models after the base video diffusion model, shown in *Figure 7 (top)*, focuses on either temporal or spatial upsampling. While the base model (the video decoder at the lowest frame rate/resolution) uses a temporal attention layer to model long-term temporal dependencies, the super-resolution models only use temporal convolution layers for computational efficiency while still maintaining local temporal consistency. Similarly, spatial attention is only used in the base and first two spatial super-resolution models, while the rest only use convolution.
{: style="text-align: justify"}

Make-A-Video’s approach initially interpolates frames and then increases the spatial resolution with two super-resolution layers, shown in *Figure 7 (bottom)*. The first super-resolution layer operates across spatial and temporal dimensions. The second super-resolution layer only operates across the spatial dimension because of memory and space constraints. However, spatial upsampling requires **detail hallucination** which needs to be consistent across frames (hence the use of the temporal dimension in the previous layer). To deal with this, they use the same noise initialization for each frame to encourage consistent detail hallucination across frames.
{: style="text-align: justify"}

<callout>
Here we show a low-resolution video upsampled using bilinear interpolation (left) and the same video upsampled using a super-resolution neural networks applied to each individual frame (middle). We see flickering artifacts because upsampling was performed separately per frame, rather than hallucinating detail across frames. The difference map (right) highlights the differences between the left and middle videos to demonstrate the flickering effect that occurs in a video when upsampled without proper detail hallucination to maintain temporal coherency. The video generation models we discuss actively consider temporal coherency while hallucinating details during upsampling to avoid these artifacts
</callout>
<figure>
  <video autoplay muted loop controls src="assets/img/2023-02-01-text2vid/artifacts_stack1.mp4"
      style="width:500px"
      type="video/mp4">
  </video>
  <figcaption>(left) Video frames upsampled using bilinear interpolation. (middle) Video frames upsampled using a super-resolution neural network. (right) Difference map between each frame of the left and middle videos, showing flickering artifacts.</figcaption>
</figure>
&nbsp;

## **Conclusions**
##### **Putting Together All The Building Blocks**
In this post, we have described the foundational building blocks of Text-to-Video generation of two popular models, Google's Imagen Video and Meta's Make-a-Video. Although these two methods have various differences, they build off of similar theory and similar building blocks. In *Figure 9*, we visually demonstrate how each of the consituent building blocks discussed in this post fit together to construct the larger Text-to-Video model.

<figure>
  <img src="assets/img/2023-02-01-text2vid/T2V_building_blocks3.png" width="700" />
  <figcaption>Figure 9. The building blocks of Text-to-Video generation.</figcaption>
</figure>
&nbsp;  

The foundational building blocks of *Figure 9* and their utility in Text-to-Video generation are summarized:
1. **Latent Decoder:** An autoencoder encoding-decoding network demonstrates how data can be compressed into a low-dimensional latent representation and then by selecting different points in this latent space, new outputs are generated.
2. **Latent Diffusion:** Training an encoding-decoding network with the stable diffusion process allows completely new images to be generated from latents with added noise. In a Text-to-Video model, image "batches" are generated from a single noisy latent put through the trained decoder to create a new, unseen video at low FPS and low resolution.
3. **2D U-Net:** To retain import information during the latent space compression encoding-decoding process, skip connections are added to tether the encoder with the decoder. This is a data-efficient architecture called a U-Net that is used in the encoding-decoding process of Text-to-Image.
4. **3D U-Net:** The 3D U-Net is an extension of the Text-to-Image 2D U-Net for Text-to-Video generation. Since it is computationally expensive to expand 2D convolution and attention directly into 3D, hence, psuedo 3D convolution and attention are constructed by concatenating 2D spatial and 1D temporal layers.
5. **Attention:** Attention layers help determine which spatial and temporal features to pay attention to, according to the input text conditioning. In a Text-to-Video model, attention helps create realistic videos by connecting important features in each frame and across frames with less required information transfer than a standard fully-connected layer.
6. **Super-Resolution:** Upsampling in both the spatial and temporal dimensions increases the video resolution and frame rate, respectively. In Text-to-Video, upsampling is done across image batches simultaneously or with other considerations to ensure temporal consistency.

By combining these six building blocks together, a **complete Text-to-Video generation model** can be constructed. Google and Meta demonstrate technically unique yet methodically similar approaches for expanding 2D Text-to-Image generation into the 3D realm while significantly improving the resolution, framerate, and temporal coherency of videos generated from text-based prompts.


##### **Limitations of Text-to-Video**
As beautiful as many of these videos are . . .
{: style="text-align: justify"}

<figure>
  <video autoplay muted loop controls src="https://imagen.research.google/video/hdvideos/51.mp4" width="600" type="video/mp4">
  </video>
</figure>
&nbsp;  

Not all of them are perfect . . . *(pay close attention to the legs of the elephant walking)*
{: style="text-align: justify"}

<figure>
  <video autoplay muted loop controls src="https://imagen.research.google/video/hdvideos/14.mp4" width="600" type="video/mp4">
  </video>
</figure>

Although Imagen Video and Make-a-Video have made significant progress in temporal coherency to remove flickering effects, complex videos generated where image data is sparse, have poor realism across the temporal dimension. In the elephant walking underwater example, a lack of training data of elephants walking or perhaps training sets with insufficient frame rates results in latent diffusion having to work harder to interpolate the missing frames, resulting in **poor temporal realism**. However, as both datasets and models continue to grow in size, the videos generated by the methods discussed in this post will improve in realism and these failure modes will become less common.

Undoubtedly, these Text-to-Video generation methods can substantially expand the creative toolbox available to artists and creators, however, key issues should be addressed before these networks become publicly available. For example, misuse of the models can result in fake, explicit, hateful, or otherwise generally **harmful content**. To help address this, additional classifiers can be trained to filter text inputs and video outputs. Moreover, the outputs reflect the composition of the training dataset, which include some problematic data, social biases, and stereotypes.
{: style="text-align: justify"}

##### **Related Works**
Several advancements have been achieved with the methods described in this post, however, video generation is not a new concept, nor do the methods described in this post solve all video generation challenges. So, here is a selection of some other interesting video generation variations/applications developed by other researchers:
{: style="text-align: justify"}
* [Phenaki](https://phenaki.video/) is another video generation tool that can generate videos of several minutes in length from story-like text prompts, compared to 5 second videos generated by Imagen Video and Make-a-Video.
* [Lee *et al.*](https://kuai-lab.github.io/eccv2022sound/) and [Narashimhan *et al.*](https://medhini.github.io/audio_video_textures/) generated video synced with audio inputs.
* [Visual Foresight](https://sites.google.com/view/visualforesight?pli=1) predicts how an object will move given an action in pixel space for more practical robotics planning and control applications.

#### **References**
[[1] Ramesh, A. et al. Zero-Shot Text-to-Image Generation, 2021. *arXiv Preprint*.](https://arxiv.org/pdf/2102.12092)
{: style="font-size: smaller"}

[[2] Saharia, C. et al. Photorealistic Text-to-Image Diffusion Models with Deep Language Understanding, 2022. *arXiv Preprint*.](https://arxiv.org/pdf/2205.11487)
{: style="font-size: smaller"}

[[3] Singer, U. et al. Make-a-Video: Text-to-Video Generation Without Text-Video Data, 2022. *arXiv Preprint*.](https://arxiv.org/pdf/2209.14792)
{: style="font-size: smaller"}

[[4] Ho, J. et al. Imagen Video: High Definition Video Generation with Diffusion Models, 2022. *arXiv Preprint*.](https://arxiv.org/pdf/2210.02303)
{: style="font-size: smaller"}

[[5] Finn, C. et al. Unsupervised Learning for Physical Interaction through Video Prediction, 2016. *30th Conference on Neural Information Processing Systems (NeurIPS)*.](https://proceedings.neurips.cc/paper/2016/file/d9d4f495e875a2e075a1a4a6e1b9770f-Paper.pdf)
{: style="font-size: smaller"}

[[6] Wang, Y. et al. PredRNN: Recurrent Neural Networks for Predictive Learning using Spatiotemporal LSTMs, 2017. *30th Conference on Neural Information Processing Systems (NeurIPS)*.](https://papers.nips.cc/paper/2017/file/e5f6ad6ce374177eef023bf5d0c018b6-Paper.pdf)
{: style="font-size: smaller"}

[[7] Babaeizadeh, M. et al. Stochastic Variational Video Prediction, 2018. *International Conference on Learning Representations (ICLR)*.](https://openreview.net/pdf?id=rk49Mg-CW)
{: style="font-size: smaller"}

[[8] Zhai, S. et al. Generative Adversarial Networks as Variational Training of Energy Based Models, 2017. *arXiv Preprint*.](https://arxiv.org/pdf/1611.01799.pdf)
{: style="font-size: smaller"}

[[9] Saito, M. et al. Temporal Generative Adversarial Nets with Singular Value Clipping, 2016. *arXiv Preprint*.](https://arxiv.org/pdf/1611.06624)
{: style="font-size: smaller"}

[[10] Wu, C. et al. GODIVA: Generating Open-DomaIn Videos from nAtural Descriptions, 2021. *arXiv Preprint*.](https://arxiv.org/pdf/2104.14806)
{: style="font-size: smaller"}

[[11] Wu, C. et al. NÜWA: Visual Synthesis Pre-training for Neural visUal World creAtion, 2021. *arXiv Preprint*.](https://arxiv.org/pdf/2111.12417)
{: style="font-size: smaller"}

[[12] Hong, W. et al. CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers, 2022. *arXiv Preprint*.](https://arxiv.org/pdf/2205.15868)
{: style="font-size: smaller"}

[[13] Kingma, D. P. et al. Variational Diffusion Models, 2021. *35th Conference on Neural Information Processing Systems (NeurIPS)*.](https://openreview.net/pdf?id=2LdBqxc1Yv)
{: style="font-size: smaller"}

[[14] Ding, M. et al. CogView: Mastering Text-to-Image Generation via Transformers, 2021. *35th Conference on Neural Information Processing Systems (NeurIPS)*.](https://proceedings.neurips.cc/paper/2021/file/a4d92e2cd541fca87e4620aba658316d-Paper.pdf)
{: style="font-size: smaller"}

[[15] Rombach, R. et al. High-Resolution Image Synthesis with Latent Diffusion Models, 2022. *IEEE / CVF Computer Vision and Pattern Recognition Conference (CVPR)*.](https://openaccess.thecvf.com/content/CVPR2022/papers/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf)
{: style="font-size: smaller"}

[[16] Ronneberger, O. et al. U-Net: Convolutional Networks for Biomedical Image Segmentation, 2015. *arXiv Preprint*.](https://arxiv.org/pdf/1505.04597)
{: style="font-size: smaller"}
