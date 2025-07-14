---
{}
---

- language: en
- license: cc-by-4.0
- tags:
- text-classification
- natural-language-inference
- repo: NA

---

# Model Card for Solution B

This is a Natural Language Inference (NLI) classification model developed to predict whether a hypothesis logically follows from a given premise. The model takes a premise and a hypothesis, and predicts whether the hypothesis is entailed by the premise. It is based on a BiMPM-style architecture using frozen RoBERTa embeddings.

## Model Details

### Model Description

This model encodes frozen RoBERTa embeddings for both premise and hypothesis, then performs multi-perspective matching using a custom BiMPMMatching layer. The matching vectors are aggregated through a second BiLSTM layer, followed by global pooling and dense layers to output a 2-class softmax prediction.

- **Developed by:** Joudi Saeidan
- **Language(s):** English
- **Model type:** Supervised
- **Model architecture:** BiMPM (non-transformer)

### References & Resources

- **Pretrained embeddings used:** [roberta-base](https://huggingface.co/roberta-base)
- **Model architecture inspiration:** [BiMPM paper (Wang et al., 2017)](https://arxiv.org/abs/1702.03814)
- **RoBERTa paper:** [Liu et al., 2019](https://arxiv.org/abs/1907.11692)

## Training Details

### Training Data

Train set: 27,481 examples from train.csv; Dev set: 6,867 examples from dev.csv. Labels are binary: entailment (1) or not-entailment (0).

### Training Procedure

#### Training Hyperparameters

- hidden_size: 64
- num_perspectives: 20
- dropout_rate: 0.4003646807673865
- learning_rate: 0.0004953648788355064
- batch_size: 16

#### Speeds, Sizes, Times

- Model size: ~6MB (best_bimpm_model.keras)
- Embedding computation time: ~6–10 minutes on Colab GPU
- Training time:
  - With Optuna (15 trials): ~2 hours
  - Final training (best config): ~20–25 minutes

## Evaluation

### Testing Data & Metrics

#### Testing Data

dev.csv used as evaluation set with gold labels.

#### Metrics

- Accuracy: ~75-76
- Precision: ~76
- Recall: ~76
- F1: ~75-76
- Classification report included in evaluation notebook

### Results

The model consistently achieved ~75–76% accuracy on the dev set after Optuna tuning.

## Technical Specifications

### Hardware

- Google Colab GPU (e.g., Tesla T4, A100)
- RAM: 12–16 GB
- Storage: ~1 GB

### Software

- Python 3.8+
- TensorFlow 2.x
- Transformers 4.18.0
- Optuna
- Scikit-learn
- Pandas, NumPy

## Bias, Risks, and Limitations

The model uses frozen RoBERTa embeddings and is not fine-tuned end-to-end. Performance may not generalize to domains or languages beyond those present in the training data.

## Additional Information

The model was inspired by the BiMPM paper (Wang, Hamza and Florian, 2017), implementing cosine-based full and max-pooling matching. Attentive matching was not implemented. Hyperparameters were optimized using Optuna.
