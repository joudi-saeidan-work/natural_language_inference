# Model Instructions

## Natural Language Inference (NLI) Task

Given a premise and a hypothesis, determine if the hypothesis is true based on the premise. 27K+ premise-hypothesis pairs were used as training data and more than 6K pairs as validation data.

---

## Solution B: Deep learning-based approach without transformer architectures

Our final model is a **BiMPM-style architecture** that uses **frozen RoBERTa embeddings** and performs **multi-perspective matching**. The premise and hypothesis embeddings are compared via full, max, and cosine-based multi-perspective matching, aggregated through a secondary BiLSTM, then passed through dense layers to output a binary prediction.

We used **Optuna** to perform hyperparameter tuning across:

- `hidden_size`, `num_perspectives`, `dropout_rate`, `learning_rate`, and `batch_size`.

The final model achieved ~75–76% accuracy on the dev set.

---

## Demo Code

- Solution B Demo: `NLI_demo_bimpm_roberta.ipynb`

To run either notebook:

1. Open it in **Google Colab**
2. Upload the appropriate `test.csv` file to the runtime
3. Navigate to `Runtime > Run All` to generate predictions

For **Solution B**, the notebook expects the following files to be present:

- the trained model to be located at:
  `SolutionB/savedModels/best_bimpm_model.keras`
- Any input files (such as `test.csv`) inside the default folder: `Data/`

> If the trained models are **not present**, the demo notebook will fail to load the model.  
> You must train the model and ensure it is saved to the path above.
> Similarly, if your data files are stored somewhere other than `Data/`, make sure to **update the paths** in the notebook accordingly.

---

## Training Code

- Solution B Training: `NLI_training_bimpm_roberta.ipynb`

For **Solution B**: the training notebook includes all necessary steps to train and tune the model, including:

- Embedding generation using frozen RoBERTa
- Model construction using a custom `BiMPMMatching` layer
- Optuna hyperparameter search
- Saving the best `.keras` model

## Evaluation Code

Evaluation code can be found in the following notebooks:

- `NLI_evaluation_bimpm_roberta.ipynb` for **Solution B**

For **Solution B**: the evaluation notebook loads the trained `.keras` model and reports performance on the dev set, including:

- Accuracy
- Precision, Recall, F1-Score
- Classification report from `sklearn`

---

## Resources

- **Access the project poster here**: [Project Poster (PDF)](https://drive.google.com/file/d/10tm2ybZ1oj4jkGvHdD2QbyoUNf_vqNQU/view?usp=sharing)

## Links to References & Code Bases

- **RoBERTa Base (Hugging Face):** https://huggingface.co/roberta-base
- **BiMPM Architecture Paper:** https://arxiv.org/abs/1702.03814
- **Optuna Hyperparameter Tuning:** https://optuna.org/

---
