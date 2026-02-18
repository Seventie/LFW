# LFW Face Generation and Latent Editing

Main notebook:
- `lfw-face-generation-latent-editing.ipynb`

Included assets:
- `Autoencoder.svg`
- `VAE_vs_AE.svg`
- `Dense_AE_Architecture.svg`

Dataset:
- https://www.kaggle.com/datasets/jessicali9530/lfw-dataset

## Workflow

1. Introduce autoencoder and VAE concepts with diagrams.
2. Load and explore LFW data.
3. Build attribute-aligned preprocessing pipeline.
4. Train dense autoencoder (`epochs=50`, `batch_size=64`, MSE loss).
5. Evaluate reconstruction quality (error maps + SSIM where available).
6. Perform latent-space editing (smile/sunglasses) and latent diagnostics.

## Notes

- Notebook is configured for Kaggle-compatible paths.
- SVG files are used directly in markdown sections for visual explanations.
