# Wine MLOps Pipeline

This project implements an end-to-end MLOps pipeline for Wine cultivar classification using the scikit-learn Wine dataset.

## Project Overview

The pipeline includes:

- Data loading and stratified train-test splitting
- Random Forest and Gradient Boosting models
- 5-fold Stratified Cross-Validation
- MLflow experiment tracking
- MLflow Model Registry
- Champion model alias
- Model quality gate
- Makefile automation
- GitHub Actions CI/CD
- Git branching and merge conflict resolution

## Dataset

The project uses the Wine dataset from scikit-learn.

- Samples: 178
- Features: 13
- Classes: 3
- Train set: 142 samples
- Test set: 36 samples
- Split: 80/20
- Random state: 42
- Stratified split: Yes

## Models

Six configurations are evaluated:

- Random Forest 1
- Random Forest 2
- Random Forest 3
- Gradient Boosting 1
- Gradient Boosting 2
- Gradient Boosting 3

Models are evaluated using:

- Macro F1
- Accuracy
- Log Loss

The best model is selected using validation Macro F1. Validation Log Loss is used as the tie-breaker.

## MLflow

MLflow is used to track:

- Model parameters
- Training metrics
- Validation metrics
- Model artifacts
- Model signatures
- Input examples

The selected model is registered as:

`WineClassifier`

The selected model is assigned the:

`champion`

alias.

## Quality Gate

The champion model must satisfy:

- Test Macro F1 >= 0.88
- Batch inference time <= 30 ms
- Predictions must only contain classes 0, 1, and 2

## Makefile Commands

```bash
make install
make lint
make test
make train
make clean
```
## CI/CD

GitHub Actions runs on:

- Push to `main`
- Pull requests to `main`

The workflow:

1. Installs dependencies
2. Runs linting
3. Trains and registers the model
4. Runs tests and the quality gate

## Git Workflow

A feature branch was used to update the README. A merge conflict was intentionally created by modifying the same file on both `main` and the feature branch. The conflict was resolved and the branches were merged successfully.