# Texture Generation Project

## Overview

This repository contains the Jupyter Notebooks for an image generation project, with aim of on creating virtual doubles for representing movement from the generation of new textures.
Three different generative models were experimented with: **Style GAN** (Generative Adversarial Network) from [this open-source PyTorch implementation of StyleGAN2](https://github.com/lucidrains/stylegan2-pytorch), **CVAE** (Conditional Variational Autoencoder) and **DCGAN** (Deep Convolutional Generative Adversarial Network) from [Tensorflow's tutorials](https://www.tensorflow.org/tutorials?hl=en). 
The goal is to find an approach for generating diverse and realistic textures.

## Project Structure

- **notebooks/**
  - `StyleGAN.ipynb`: Jupyter Notebook for training and generating textures using Style GAN.
  - `DCGAN_+_CVAE.ipynb`: Jupyter Notebook for training and generating textures using Conditional Variational Autoencoder or Deep Convolutional Generative Adversarial Network.
  
- **datasets/**
  - *all_members_256.zip*: 256x256px textures, each encopassing all movement types.
  - *all_members_64.zip*: 64x64px textures, each representing one movement type.
