# Continual Commonsense Reasoning with Multi-Task Learning

This project aims to build a robust commonsense reasoning model that can continually learn multiple tasks over time. By leveraging task-specific datasets and novel augmentation techniques, the model is trained to solve a wide range of commonsense QA tasks with improved generalization and transferability.

## 🚀 Project Overview

We develop a continual learning framework for commonsense reasoning that:
- Learns from multiple commonsense QA tasks sequentially.
- Leverages pretrained language models (e.g., RoBERTa) as backbone models.
- Employs various **knowledge distillation** and **sample augmentation** strategies to improve performance on future tasks.
- Supports **bi-encoder** and **SBERT-based** methods to generate additional training samples from incorrect predictions.

## Core Features

- 📚 **Multi-Task Commonsense QA**  
  Trains on diverse commonsense reasoning tasks such as CSQA, SocialIQA, and more.

- 🔁 **Continual Learning**  
  Supports fine-tuning and sequential transfer.

- 🔍 **Sample Selection and Augmentation**  
  Enhances training by modifying or augmenting incorrect samples using semantic similarity models (SBERT, Bi-Encoder).

- 🎯 **Knowledge Distillation**  
  Supports soft-label training using logits from previous models for smoother knowledge transfer.
