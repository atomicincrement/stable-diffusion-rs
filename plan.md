# Stable diffusion demo using just ndarray

The goal of this demo is to learn how stable diffusion generates images from
text prompts.

We will be just doing the forward inference part of the project, generating an image
from a prompt.

* Fetch a smallish stable diffusion weight set
* Decode the weights to ndarray tensors
* Implement the text understanding (CLIP) phas to generate embeddings
* Latent Diffusion 1 - Forward Process (Training)
* Latent Diffusion 2 - Inference
* Image reconstruction - VAE Decoder
* Demo - Simple text prompt and generated image.
