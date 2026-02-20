# LFW Face Generation and Latent Attribute Editing

Main notebook:
- `lfw-face-generation-latent-editing.ipynb`

## Notebook Flow (Mermaid)

```mermaid
flowchart TD
  A[Load LFW Images + Attributes] --> B[Preprocess: crop, resize 45x45, normalize]
  B --> C[Train/Val Split]
  C --> D[VAE Training]
  D --> E[Reconstruction Diagnostics]
  D --> F[Random Face Generation]
  D --> G[Latent Interpolation]
  D --> H[PCA and t-SNE Latent Analysis]
  D --> I[Smile and Sunglasses Latent Editing]
```

## Dataset Introduction

This project uses the **Labeled Faces in the Wild (LFW)** dataset:
- Dataset link: https://www.kaggle.com/datasets/jessicali9530/lfw-dataset
- Data used in notebook:
- RGB face images
- Face-level attributes (for example: smiling, eyeglasses, sunglasses)

The notebook aligns image files with attribute rows, crops/resizes faces to `45x45`, normalizes pixel values, and prepares train/validation splits.

## What We Are Doing Here

The goal is to learn a latent representation of faces that supports:
- faithful reconstruction,
- realistic sampling/generation,
- controllable semantic edits (smile, sunglasses).

The final training path in this notebook is **VAE-focused**, because it gives a smoother and more structured latent space for generation and editing.

## Autoencoder (AE) Diagram

![Autoencoder](AES.svg)

Autoencoders compress an input image into a latent vector and decode it back to reconstruct the same image.

Why AE matters in this project:
- It defines the core encoder-decoder idea.
- It is the conceptual baseline before moving to probabilistic latent modeling.



## AE vs VAE and Differences

![AE vs VAE Explained](AE_vs_VAE_Explained.svg)


Key differences:
- **AE**:
- deterministic latent vector,
- reconstruction-only objective.
- **VAE**:
- latent distribution (`mu`, `logvar`) + reparameterization,
- reconstruction loss + KL divergence,
- better latent smoothness for interpolation and generation.

## How Each Flow Step Is Implemented

1. `Load + Preprocess`
- Read LFW and attributes, align rows, resize to `45x45`, normalize.

2. `Train/Val Split`
- Create train and validation tensors for stable evaluation.

3. `VAE Training`
- Train encoder/decoder with reconstruction + KL objective.

4. `Reconstruction Diagnostics`
- Compare originals vs reconstructions, inspect error maps, and SSIM.

5. `Random Generation`
- Sample latent vectors and decode to new synthetic faces.

6. `Interpolation`
- Interpolate between two latent points to test latent continuity.

7. `Latent Analysis`
- Use PCA/t-SNE to inspect structure of learned latent embeddings.

8. `Attribute Editing`
- Build smile/sunglasses directions and apply latent vector arithmetic.

## How To Use This For Your Use Case

Use this notebook as a template when you need controllable face generation/editing:

1. Replace dataset paths with your own face dataset + attributes.
2. Keep the same preprocessing contract (`45x45` RGB or adjust model input dims accordingly).
3. Train VAE and monitor both reconstruction and KL components.
4. Define your own semantic directions from attributes (for example: age, beard, glasses).
5. Use latent edits for controlled augmentation or interactive generation workflows.

## Included Files

- `lfw-face-generation-latent-editing.ipynb`
- `AE_vs_VAE_Explained.svg`
- `AES.svg`
- `README.md`
