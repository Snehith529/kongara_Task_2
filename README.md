# SIMCLR Implementation - kongara_Task_2

This repository contains the implementation of Task 2 using the SIMCLR (Contrastive Learning) approach.

## Implementation Logic

### Step 1: Reading CSV File
The initial step involves reading the CSV file named "data.csv". This file likely contains column ACC X data for the task.

### Step 2: Converting ACC X to Spectrograms
In this step, the values from the "ACC X" column of the CSV file are processed to create spectrograms. Spectrograms are a way to represent the frequency content of a signal over time.

### Step 3: Data Augmentation
Data augmentation is applied to the spectrograms to create datset. As a starting point, random blurring is employed to simulate various real-world scenarios.

## To-Do (Future Steps)

### Step 4: Using Pre-trained Network for Encoding
Incorporate a pre-trained network, such VGG-16, ResNet, to perform the encoding of augmented spectrogram data.

### Step 5: Custom Projection-Head
Develop a custom projection head architecture that takes encoded features and projects them into a suitable space for applying contrastive loss.

### Step 6: Contrastive Loss and Network Update
Implement the contrastive loss function, which quantifies the similarity between positive pairs (augmented views of the same data) and maximizes this similarity while minimizing the similarity with negative pairs (views of different data). This step fine-tunes the network for meaningful feature extraction.


