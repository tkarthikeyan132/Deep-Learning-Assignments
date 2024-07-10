# Deep Learning Assignments

Welcome to the repository containing solutions to deep learning assignments! Each assignment is designed to enhance your understanding of fundamental deep learning concepts and techniques.

## Assignment 1A: ResNet and Normalization Schemes

### 1. Implementing ResNet for Image Classification
- **Task**: Implement a ResNet architecture in PyTorch from scratch.
- **Dataset**: Indian Birds species classification.
- **Details**:
  - Total layers: 6n+2, with n as input.
  - Feature map sizes: 256x256, 128x128, 64x64.
  - Filters: 16, 32, 64.
  - Train for 50 epochs with SGD optimizer, initial learning rate \(10^{-4}\).
  - Report: Accuracy, Micro F1, Macro F1 on Train, Val, and Test splits.

### 2. Impact of Normalization
- **Task**: Implement and compare different normalization schemes within ResNet.
- **Normalization Schemes**:
  - Batch Normalization (BN)
  - Instance Normalization (IN)
  - Batch-Instance Normalization (BIN)
  - Layer Normalization (LN)
  - Group Normalization (GN)
- **Comparison**: Train models with different normalization schemes and compare their performance.

### 3. Visual Explanations with Grad-CAM
- **Task**: Visualize model predictions using Grad-CAM for selected classes.
- **Classes**: Cattle Egret, Coppersmith Barbet, Indian Peacock, Red Wattled Lapwing, Ruddy Shelduck, White Breasted Kingfisher, White Breasted Waterhen.
- **Analysis**: Visualize gradients for correctly and incorrectly classified images and report observations.

## Assignment 1B: Text to Math Program

### 1. Task
- **Goal**: Convert mathematical problems in text to executable programs using seq2seq architectures.
- **Dataset**: Provided with train, test, and dev json files.
- **Evaluation Metrics**:
  - Exact Match
  - Execution Accuracy

### 2. Models
- **Architectures**:
  - Seq2Seq with GloVe embeddings and Bi-LSTM encoder, LSTM decoder.
  - Seq2Seq+Attention with GloVe embeddings and Bi-LSTM encoder, LSTM decoder.
  - Seq2Seq+Attention with pre-trained BERT encoder and LSTM decoder.
  - Seq2Seq+Attention with fine-tuned BERT encoder and LSTM decoder.
- **Experiments**:
  - Report loss curves, Exact Match Accuracy, and Execution Accuracy.
  - Vary teacher forcing probabilities and beam sizes and report the impact on model performance.

### 3. Effect of Teacher Forcing Probability
- **Task**: Train with different teacher forcing probabilities (0.3, 0.6, 0.9) and compare performance.

### 4. Effect of Beam Size
- **Task**: Evaluate the impact of different beam sizes (1, 10, 20) on the best performing model.


## Assignment 2A: Object-Centric Learning With Slot Attention

### 1. Slot Attention
Slot attention is a mechanism that maps from a set of N input feature vectors to a set of K output vectors, referred to as slots. These slots can describe objects or entities in the input. Slot attention uses an iterative attention mechanism to refine these slots.

#### Steps to Implement Slot Attention:
1. **Input Image Encoding**:
    - Given an image, a CNN encoder extracts feature maps from the image.
    - A 2D positional encoding is added to the feature maps.
    - Flatten the enhanced feature maps spatially to get input vectors and pass them through an MLP to get the final inputs.

2. **Slot Initialization**:
    - Initialize slot vectors by sampling from a learnable normal distribution.

3. **Iterative Attention**:
    - Slots attend to inputs iteratively for a set number of times and update slots at the end of each iteration.

4. **Decoding**:
    - Each of the slot vectors is decoded with the Spatial Broadcast Decoder to produce a multi-channeled image.
    - Generate masks by taking softmax for each pixel of the first channel across slots.
    - The final reconstructed image is obtained by combining the masks and the content channels.
    - Compute the image reconstruction loss over a batch.

### 2. Experiments
- **Dataset**: Sub-sampled CLEVRTex dataset.
- **Hyperparameters**: Use 11 slots for this experiment.
- **Tasks**:
    - Report the Adjusted Rand Index (ARI) score between the ground-truth and predicted object segmentation masks on the validation split.
    - Plot the train and validation image reconstruction loss vs epochs.
    - **Compositional Generation**: Create a slot library using the training data, apply K-means clustering, sample new slots to generate images, and report the clean-FID metric using validation images as ground truth.

## Assignment 2B: Slot Learning using Diffusion based Decoder

### 1. Diffusion Models
Diffusion models are generative models that learn the data distribution through a noising process and a learned reverse process to denoise the image. Latent space Diffusion Models (LDM) train the model in the latent space of a pretrained Variational AutoEncoder (VAE).

### 2. Implementation
1. **Slot Encoder**:
    - Same as in Part 2A, encoding images into slots using a CNN-based encoder and slot attention module.

2. **Diffusion Decoder**:
    - Train a conditional diffusion model to generate latent representations given slots.
    - **Forward Process**: Gradually corrupt training latents with Gaussian noise.
    - **Reverse Process**: Denoise the corrupted latents iteratively using a modified UNet architecture.

#### Components:
- **Residual Block**: Used for downsampling and upsampling in UNet, with different operations for each.
- **Transformer Block**: Cross attention with slots added to UNet for conditioning.

#### Decoding (Generation):
- Implement Ancestral Sampling to iteratively denoise from the initial noise to the final latent representation.

### 3. Experiments
- Repeat experiments from Part 2A on the same dataset.
- Report results using slot-input attention maps as proxies for slot masks.
- Utilize the provided VAE checkpoint and inference script.

```python
import VAE from vae_
import torch

vae = VAE()
# Load the checkpoint (replace with your checkpoint path)
ckpt = torch.load('vae_checkpoint.pth')
vae.load_state_dict(ckpt)

# To encode an image
# images: [Batch size x 3 x H x W]
z = vae.encode(images)

# To decode z
# z: [Batch size x 3 x 32 x 32]
rec_images = vae.decode(z)
```

## Note

- All the solutions provided in this repository have been implemented from scratch. Feel free to explore the code.
- This repository is open for reference to everyone except the students currently enrolled in the COL775-Deep Learning course offered at IIT Delhi. If you are currently taking COL775, we encourage you to refrain from accessing this repository to maintain the academic integrity of the course.

## Acknowledgement

A special thanks to [Prof. Parag Singla](https://www.cse.iitd.ac.in/~parags/teaching.html) for offering the machine learning course during the academic year 2023-24 at the Indian Institute of Technology (IIT) Delhi. The content of these assignments is inspired by the course lectures and materials provided during the tenure of this course.