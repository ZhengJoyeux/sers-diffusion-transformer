# DiT reference

SpectrumDiT1D is a one-dimensional, mask-aware adaptation of the DiT architecture described by William Peebles and Saining Xie, “Scalable Diffusion Models with Transformers”, ICCV 2023.

Official implementation: https://github.com/facebookresearch/DiT
Reference commit: ed81ce2229091fd4ecc9a223645f95cf379d582b
Copyright (c) Meta Platforms, Inc. and affiliates.
Official reference code license: Creative Commons Attribution-NonCommercial 4.0 International, included as DiT_LICENSE.txt.

Changes: direct spectrum residual-space operation; one-dimensional Linear patches; continuous chemical conditioning; valid-point/key masks; dynamic padding and cropping; PyTorch SDPA; gradient checkpointing; existing SERS reconstruction-base interface. Image VAE, ImageNet label embedding, learned variance head and image classifier-free-guidance helper are not transplanted.

Additional implementation reference for gradient checkpointing: https://github.com/chuanyangjin/fast-DiT . No dependency on this repository is required.

D4.28 further adapts the shared DiT blocks to fixed-coordinate floating-point curve images, full-height rectangular Linear patches, two-dimensional pixel diffusion, and differentiable column decoding back to the original SERS residual interfaces. No VAE or natural-image pretrained backbone is introduced.
