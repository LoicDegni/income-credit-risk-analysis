# income-credit-risk-analysis

## Project Overview

This project applies end-to-end data mining and machine learning techniques to solve two real-world classification problems related to income prediction and credit risk assessment.

Two datasets are used:

- **Income dataset** (~48,000 individuals) : Predict whether a person earns more or less than $50,000 per year based on demographic, professional, and financial attributes.
[View dataset](https://github.com/LoicDegni/income-credit-risk-analysis/blob/main/data/revenu.csv)<br>

- **Credit dataset** (690 records): Predict whether a credit application should be approved based on financial and personal characteristics.
[View dataset](https://github.com/LoicDegni/income-credit-risk-analysis/blob/main/data/credit.csv)<br>

The project follows a complete data science workflow:

- Data preprocessing and handling of missing values (encoded as ?)
- Exploratory data analysis (EDA)
- Feature engineering and encoding
- Class imbalance handling
- Model training and evaluation <br>

A total of seven classification algorithms are implemented and compared using:

- Train/test split (70% / 30%)
- Cross-validation (5, 7, and 10 folds)

The goal is to build robust predictive models while analyzing overfitting, model generalization, and performance trade-offs.

## Business Relevance
- **Income prediction** enables customer segmentation and targeted marketing strategies
- **Credit approval prediction** supports risk management and decision-making in financial institutions

## Problem Statement
This project addresses several key challenges commonly encountered in real-world data science workflows:

1. Data Quality
The datasets contain missing values encoded as "?", primarily affecting categorical features such as workclass, occupation, and native-country.

- How can missing values be properly handled without distorting the original data distribution?
- How do these missing values impact model reliability and prediction quality?<br>

2. Feature Encoding

The datasets include multiple categorical variables with different characteristics:
- Nominal variables (e.g., occupation, workclass)
- Ordinal variables (e.g., education level)

Key question:
- Which encoding techniques (One-Hot vs Label Encoding) best preserve information and improve model performance?<br>

3. Model Selection

Seven classification algorithms are evaluated:

- Logistic Regression
- Decision Tree
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Naive Bayes
- Gradient Boosting

Key question:

- Which models offer the best trade-off between performance and interpretability?<br>

4. Validation & Reliability

Model performance is evaluated using:

- Train/test split (70/30)
- Cross-validation (5, 7, and 10 folds)

Key question:

- How stable and reliable are the results across different validation strategies?<br>

5. Generalization & Overfitting

Some models show significantly higher performance on training data than on test data.

Key question:

- How can overfitting be detected and mitigated to ensure robust generalization?

## Methodology

1. Data Preprocessing

### Income Dataset
- Missing Values Handling
Missing values (encoded as "?") were identified in categorical features (workclass, occupation, native-country).
    - workclass and occupation missing values were considered informative (e.g., unemployed individuals) and encoded as "unknown"
    - native-country missing values were kept as-is
- Feature Selection
    - Removed fnlwgt (non-informative for prediction)
    - Dropped redundant feature education (kept educational-num instead)
- Categorical Encoding
    -  Applied One-Hot Encoding to all nominal variables
    - Target variable (income) encoded as binary (0 / 1)
- Numerical Scaling
    - Applied standardization (StandardScaler)
    - Special focus on skewed features (capital-gain, capital-loss)
    - Multiple strategies tested → standardization yielded best performance

### Credit Dataset
- Data Cleaning & Imputation
    - No explicit missing values detected
    - Identified inconsistent or unreliable features:
        - Removed zipcode due to high cardinality and low predictive value
        - Detected anomalies in income (e.g., employed individuals with zero income) → imputed using median
- Feature Validation
    - Analyzed distributions and corrected incoherent patterns (e.g., credit score behavior, income inconsistencies)<br>

2. Exploratory Data Analysis (EDA)
- Analyzed distributions, outliers, and feature relationships
- Identified:
    - Strong class imbalance in several categorical features (e.g., race, native-country)
    - Highly skewed numerical variables (e.g., capital-gain, income)
- Highlighted potential redundancies (e.g., marital-status vs relationship)<br>

3. Class Imbalance Handling
### Income Dataset
- Strong imbalance (~75% ≤50K vs ~25% >50K)
- Applied RandomUnderSampler to balance classes
- Chosen due to large dataset size (minimal information loss)
### Credit Dataset
- Moderate imbalance (~77% ratio)
- Applied RandomOverSampler
- Also tested SMOTE → similar performance
- Oversampling preferred due to small dataset size<br>

4. Model Training

Seven classification models were trained and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- KNN
- SVM
- Naive Bayes
- Gradient Boosting<br>

5. Model Evaluation
- Train/Test split: 70% / 30%
- Cross-validation: 5, 7, and 10 folds
- Performance comparison across models
- Analysis of overfitting and generalization gaps

## Results - Income Dataset (Final version)

### Model performance comparison 

| Model               | Accuracy | Precision | Recall   | F1-score | AUC-ROC  | Overfitting |
|---------------------|----------|-----------|----------|----------|----------|-------------|
| Logistic Regression | ~81%     | ~57%      | ~85%     | ~68%     | ~82%     | Low         |
| Decision Tree       | ~77%     | ~52%      | ~81%     | ~63%     | ~78%     | 🔴 High     |
| Random Forest       | ~80.5%   | ~57%      | ~83%     | ~67%     | ~81%     | 🟠 Moderate |
| KNN                 | ~77–78%  | ~52%      | ~75%     | ~62%     | ~77%     | Low         |
| SVM                 | ~79%     | ~55%      | ~85–86%  | ~66%     | ~81%     | Low         |
| Naive Bayes         | ~64–65%  | ~40%      | ~92%     | ~55%     | ~74%     | Low         |
| Gradient Boosting   | **~82%** | **~59%**  | **~87%** | **~70%** | **~84%** | Low         |

### Best Model: Gradient Boosting

### Key Insights
1. Ensemble models outperform others
- Gradient Boosting and Random Forest outperform simpler models
- They better capture complex relationships in the data<br>

2. Overfitting in Decision Tree
- Extremely high training scores (~97%) vs much lower test performance
- Indicates strong overfitting → poor generalization<br>

3. Recall vs Precision trade-off
- Most models show:
    - high recall (good detection of >50K income)
    - lower precision (more false positives)
The models tend to over-predict high income<br>

4. Naive Bayes behavior
- Very high recall (~92%) but extremely low precision (~40%)
- Not suitable for balanced decision-making<br>

5. Stability across validation methods
- Cross-validation results are consistent across:
    - 5, 7, and 10 folds
- Confirms robustness of the models

**Business Interpretation**
- Models are effective for identifying high-income individuals (high recall)
- However, lower precision means:
    - risk of misclassifying lower-income individuals as high-income

Best suited for:
- marketing targeting (acceptable false positives)
Less suited for:
- strict financial decision-making (needs higher precision)

## Results - Credit Dataset (Final version)

### Model Performance Comparison

| Model               | Accuracy    | Precision | Recall      | F1-score    | AUC-ROC | Overfitting  |
| ------------------- | ----------- | --------- | ----------- | ----------- | ------- | ------------ |
| Logistic Regression | ~85–86%     | ~82%      | ~88%        | ~84–85%     | ~85–86% | Low          |
| Decision Tree       | ~80%        | ~79%      | ~75%        | ~77%        | ~80%    | 🔴 Very High |
| Random Forest       | ~86–87%     | ~85–86%   | ~84%        | ~85%        | ~86%    | 🔴 High      |
| KNN                 | ~71%        | ~68%      | ~66%        | ~67%        | ~70%    | Moderate     |
| SVM                 | ~65–66%     | ~80%      | ~30–40%     | ~45%        | ~62–63% | Low          |
| Naive Bayes         | ~81–82%     | ~85%      | ~70–72%     | ~77–78%     | ~80–81% | Low          |
| Gradient Boosting   | **~85–86%** | ~82%      | **~86–88%** | **~83–85%** | ~85%    | 🟠 Moderate  |

### Best Model: Random Forest / Gradient Boosting

- **Random Forest**
    - Highest overall accuracy (~87%)
    - Strong balance across all metrics
    - However: clear overfitting (train = 100%)
- **Gradient Boosting**
    - Slightly lower accuracy but:
    - Better generalization
    - More stable across validation folds
**Final choice: Gradient Boosting (more robust)**

### Key Insights
1. Strong performance overall
    - Most models (LR, RF, GB) achieve >85% accuracy
    - Indicates dataset is **predictive despite small size**<br>
2. Severe overfitting in tree-based models
- Decision Tree and Random Forest:
    - Train score = 100%
    - Significant gap with test performance
-> This issue is related to the small size of the dataset<br>

3. Logistic Regression performs surprisingly well
Very stable across all validations
Strong baseline with:
high recall
good F1-score<br>

4. SVM imbalance issue
- Very high precision (~80%)
- Very low recall (~30–40%)
-> Model is too conservative -> misses many positive cases<br>

5. Impact of small dataset
- Results are more sensitive to:
    - sampling
    - imbalance
    - model complexity

### Business Interpretation 
- Models are effective for **credit approval prediction**
- High recall -> good at identifying valid applications
- But:
    - Overfitting risk reduces reliability in production

-> Best suited for:
- decision support systems
- require : 
    - further validation before real deployment

## Final Section - Conclusion 

This project demonstrates the application of machine learning techniques to two distinct but related financial prediction problems: **income classification** and **credit approval**.

### Key Takeaways
- Data quality and preprocessing are critical
Handling missing values and feature engineering had a significant impact on model performance
- Model choice depends on the dataset
    - Gradient Boosting performed best on both datasets
    - Simpler models (Logistic Regression) provided strong and stable baselines
- Overfitting is a major challenge
Especially for complex models on smaller datasets (e.g., Credit dataset)
- Evaluation strategy matters
Cross-validation provided more reliable insights than a simple train/test split

### Income vs Credit Dataset

| Aspect     | Income Dataset    | Credit Dataset    |
| ---------- | ----------------- | ----------------- |
| Size       | Large (~48k)      | Small (~690)      |
| Challenge  | Class imbalance   | Data variability  |
| Best Model | Gradient Boosting | Gradient Boosting |
| Key Issue  | Precision         | Overfitting       |
