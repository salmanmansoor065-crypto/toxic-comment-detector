# Toxic Comment Detection System 🛡️

An AI-powered system to detect toxic comments using Deep Learning with complete MLOps pipeline.

## Problem Statement
Online platforms face challenges with toxic, abusive, and hateful comments. Manual moderation is time-consuming and inconsistent. This system automates toxic comment detection using NLP and Deep Learning.

## Real-World Significance
Social media platforms, forums, and online communities can use this system to automatically moderate content, reduce cyberbullying, and create safer online environments.

## Dataset
- **Source:** Jigsaw Toxic Comment Classification Challenge (Kaggle)
- **Size:** 159,571 comments
- **Labels:** toxic, severe_toxic, obscene, threat, insult, identity_hate

## Models
| Model | Architecture | Accuracy | F1-Score |
|-------|-------------|----------|----------|
| v1 (Baseline) | LSTM + Embedding | 99.39% | - |
| v2 (Upgraded) | Bidirectional LSTM | 99.31% | 73.04% |

## Tech Stack
- **Deep Learning:** TensorFlow/Keras
- **Experiment Tracking:** MLflow
- **API:** FastAPI
- **Containerization:** Docker
- **Orchestration:** Kubernetes (Minikube)
- **CI/CD:** GitHub Actions
- **Version Control:** Git/GitHub

## Project Structure
