# Car Brand Logo Classification with CNNs

A deep learning project that classifies car brand logos from images and investigates the vanishing gradient problem through a comparison between a conventional CNN and a ResNet architecture.

## Overview

This project tackles an 8-class image classification problem: automatically recognizing car brand logos (Hyundai, Lexus, Mazda, Mercedes, Opel, Skoda, Toyota, Volkswagen). Beyond building a working classifier, the study is designed as a controlled experiment to answer a theoretical question in deep learning: why does the performance of conventional CNNs degrade as network depth increases, and can residual (skip) connections solve this problem.

The work is split into two stages. The first stage focuses on designing, tuning, and evaluating a custom CNN through systematic hyperparameter experimentation. The second stage compares that CNN against a ResNet-style architecture to directly measure the impact of skip connections on training stability and accuracy.

## Dataset

- Source: Car Brand Logos dataset (Kaggle, curated by user volkandl)
- 2,523 images across 8 balanced classes (~315 images per class)
- Original resolution 256x256, resized to 128x128 for training
- Split into training (70%), validation (15%), and test (15%) using stratified sampling

## Methodology

**Preprocessing pipeline**

Images are resized, normalized to a [0, 1] range, and their labels one-hot encoded. Data augmentation (rotation, shifts, zoom, horizontal flip) is applied to the training set to improve generalization and reduce overfitting.

**Baseline CNN**

A conventional sequential architecture with three convolutional blocks (32, 64, 128 filters), batch normalization, max pooling, and dropout, followed by a dense classification head. Total parameters: 8.71 million.

**Optimized CNN (Part 1)**

The same three-block philosophy, refined through 13 systematic experiments that tuned learning rate, batch size, architecture depth, and regularization strength (L2, dropout). The final configuration reached 92.2% test accuracy, with EarlyStopping and ReduceLROnPlateau used to guarantee stable convergence.

**ResNet Architecture (Part 2)**

A residual network built from an initial block plus seven residual blocks organized into three stages, using skip connections of the form H(x) = F(x) + x. This design achieves better gradient flow to earlier layers and uses only 3.20 million parameters, 63.3% fewer than the baseline CNN.

## Key Findings

- The baseline CNN plateaued at just 13.72% validation accuracy, barely above random chance (12.5% for 8 classes), a clear symptom of the vanishing gradient problem.
- The ResNet model reached 58.84% validation accuracy in the same comparative experiment, a 4.3x improvement while using far fewer parameters.
- In the fully tuned Part 1 experiments, the optimized 3-block CNN with proper regularization and data augmentation achieved 92.2% overall test accuracy, with F1-scores above 0.88 for every class.
- The most frequent misclassifications occurred between visually similar logos (Hyundai and Mercedes), both featuring elliptical outlines.

## Repository Structure
'''
├── src/
│ ├── data_preprocessing.py
│ ├── baseline_cnn.py
│ ├── optimal_cnn.py
│ └── resnet_model.py
├── data/
├── docs/
│ ├── Workshop2_Part1_Report.pdf
│ ├── Workshop2_Part2_Report.pdf
│ └── presentation.pdf
└── README.md
'''

## Tech Stack

Python, TensorFlow/Keras, Scikit-learn, NumPy, Matplotlib, Seaborn, Google Colab (GPU-accelerated training)

## Author

Diego Fiz Garcia
