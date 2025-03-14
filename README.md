# UC Berkeley | Professional Certificate in Machine Learning and Artificial Intelligence
## Practical Application 3 (Module 17)


This project aims to demonstrate and compare the capabilities of different classification models, including Logistic Regression, Decision Trees, Support Vector Machines (SVC), and K-Nearest Neighbors (KNN). Using the Bank Marketing dataset from the UCI Machine Learning Repository, we analyze how well these models predict whether a customer will subscribe to a long-term deposit based on demographic, financial, and marketing interaction features. The goal is to evaluate each model’s performance in terms of accuracy, precision, recall, and training efficiency, providing insights into their strengths and trade-offs for real-world applications.

---

## Project Overview
### This project implements the CRISP-DM methodology.

- **Objective**: To identify and confirm the key factors that make customers subscribe a long-term bank account.
- **Scope**: The analysis focuses on common variables such as campaign efforts, phone calls, durationof the call, employment status, age, education and others.
- **Outcome**: This project will highlight the effectivenes, accuracy, performance and processing efficiency (run drurations) of the 4 classifiers in Scikit-Learn - `Logistic Regression`, `Decision Trees`, `Support Vector Machines (SVC)`, and `K-Nearest Neighbors (KNN)`.

---

## Key Features and Methodology

- **Feature Engineering**:
  - Numerical and categorical features were indentified 
  - One-hot encoding is used for categorical features such as `job`, `education`, `age`, `marital status` and others.
  - StandardScaler was applied on numerical features

- **Model Implementation**:
  - `Logistic Regression`
  - `Decision Trees`
  - `Support Vector Machines (SVC)`
  - `K-Nearest Neighbors (KNN)`.

- **Result Reports**:
  - Accuracies
  - Runtime
  - ROC Curves
  - Confusion Matrices


---

## Visualizations
- **Histogram Plot**:
  - Used in initial data analysis and understanding

- **Scatter Plot**:
  - Used in initial data analysis and understanding

- **ROC**:
- **Confusion Matrix**
  - USed in visualizing results


---

## Key Results

- **Model Performance**:
  - The No-skill comparisons reports vs. a simple Logistic Regression Model
  - The 4 models `Logistic Regression`, `Decision Trees`, `Support Vector Machines (SVC)`, and `K-Nearest Neighbors (KNN)` are comapared showing perfoamnce, accuracy and processing times.


---

## Conclusion

The comparison tables highlight the significant improvement of predictive models over the No-Skill model, which simply predicts the majority class. 

Logistic Regression serves as a strong baseline, achieving a balanced trade-off between precision and recall, making it a reliable choice for structured datasets. 

Decision Trees offer slightly better recall for the minority class while maintaining high accuracy, but they can be prone to overfitting. 

Support Vector Machines (SVC) show similar performance to Logistic Regression but take significantly longer to train, making them less efficient for large-scale applications. 

K-Nearest Neighbors (KNN) also performs well, especially in terms of recall for the majority class, though it struggles with the minority class and requires substantial computational power for large datasets. 

The choice of the best model depends on the trade-off between accuracy, interpretability, and training time. While these models perform well, there are several ways to further improve them, including hyperparameter tuning, feature engineering.

Given more time, fine-tuning parameters and incorporating more advanced models could further enhance predictive accuracy and efficiency.


---

## Next Steps

- **Continuous Data Updates**:
  - Integrate new data on bank campaigns.

- **Advanced Feature Engineering**:
  - Explore additional features like Polynomials, PCA for performance boost for SVC.

- **Model Refinement**:
  - Try additional, various sets of hyperparameters
  - Use Kernel Trick

---

## Repository Structure

- `data/`: Contains the cleaned dataset used for analysis.
- `README.md`: Project documentation.

---

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/christianfajardo/ML_AI_PracticalApplication_3.git
   ```

2. Install Python, NumPy, Pandas, Scikit Learn, Matplotlib, and Seaborn Plot dependencies.
   
3. Load and run all `prompt_III.ipynb` cells in VSCode or any Jupyter editor of your choice.




