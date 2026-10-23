# FairCredit AI

**Responsible Credit Default Prediction and Fairness Analysis**

FairCredit AI is a machine learning project that predicts the likelihood of credit card default and explores model performance across demographic groups. It includes model evaluation, fairness metrics, confidence intervals, and a Streamlit dashboard with individual prediction explanations.

## Project objectives

- Predict credit card default using historical customer data.
- Compare classification models using standard evaluation metrics.
- Examine prediction rates, recall, and false-positive rates across sex and age groups.
- Use 95% Wilson confidence intervals to communicate uncertainty in group-level metrics.
- Provide an interactive dashboard for risk assessment and model explanations.

## Dataset

The project uses the **Default of Credit Card Clients** dataset.

Place the dataset at:

`data/default of credit card clients.xls`

The target variable is `DEFAULT`, where `1` represents default and `0` represents no default. The `ID` column is excluded from model features.

## Models and evaluation

The project explores Logistic Regression, Decision Tree, Random Forest, and K-Nearest Neighbors.

The Logistic Regression baseline achieved approximately:

| Metric | Result |
|---|---:|
| Accuracy | 81.7% |
| Precision | 66.1% |
| Recall | 35.2% |
| F1-score | 0.459 |
| ROC-AUC | 0.758 |

These results are from a stratified 80/20 train-test split. Recall is relatively low, meaning the model misses many actual defaulters at its current classification threshold.

## Fairness analysis

The analysis compares predicted default rates, recall, and false-positive rates across sex and age groups. Wilson 95% confidence intervals help show the uncertainty in estimated rates.

Differences between groups are indicators for further investigation, not proof of discrimination or causation. This project is educational and is not a validated credit-decision system.

## Project structure

```text
FairCredit-AI/
├── data/
├── notebooks/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── fairness.py
├── models/
├── outputs/
├── app.py
├── requirements.txt
└── README.md
```

## Setup

Create and activate a virtual environment, then install dependencies:

```bash
pip install -r requirements.txt
```

Ensure the dataset is available at the path specified above.

## Run the project

Train the model:

```bash
python src/train.py
```

Evaluate the saved model:

```bash
python src/evaluate.py
```

Run the fairness analysis:

```bash
python src/fairness.py
```

Launch the dashboard:

```bash
streamlit run app.py
```

## Limitations and responsible use

- Model performance depends on the dataset and classification threshold.
- Group-level metrics do not establish the presence or absence of discrimination on their own.
- Small demographic groups can have uncertain estimates.
- SHAP explanations describe how model features contribute to a prediction; they do not establish causality.
- This project is for learning and demonstration, not real-world lending decisions.

## License

See the repository's `LICENSE` file.
