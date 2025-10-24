# Machine-Learning-Projects
This project showcases end-to-end machine learning workflow using synthetic data: data generation, train-test splitting, training multiple models (Logistic Regression, Decision Tree, Random Forest), evaluating with ROC/AUC, and visualizing results—demonstrating model comparison, performance analysis, and binary classification insights.



##  Project: Employee Salary Prediction using Linear & Ridge Regression

###  Project Description:
This project models employee salary based on synthetic data representing experience, education, job role, and location. It compares **Linear Regression** and **Ridge Regression** to evaluate prediction accuracy and regularization effects.

###  Objectives:
- Generate synthetic employee data with realistic salary patterns
- Train and compare Linear vs Ridge regression models
- Predict salary for new employee input
- Visualize actual vs predicted salaries using bar charts

###  Tools & Technologies:
- **Python**: Core language
- **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn
- **Models**: Linear Regression, Ridge Regression

###  Key Features:
- Synthetic data generation with noise
- Salary prediction for new employee profile
- Coefficient comparison between models
- Bar chart visualization of prediction accuracy

###  Outcome:
Demonstrates regression modeling and visualization techniques for salary prediction, with insights into model behavior and feature impact.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/3017bc5d-377d-4306-9c52-2855bf818df2" />


---

##  Project: Student Final Score Prediction using Regression

###  Project Description:
This project predicts students' final scores using study hours, attendance, previous scores, and participation. It compares **Linear Regression** and **Ridge Regression** to assess model performance.

###  Objectives:
- Simulate academic performance data
- Train regression models to predict final scores
- Evaluate predictions for new student input
- Visualize actual vs predicted scores

###  Tools & Technologies:
- **Python**: Core language
- **Libraries**: Pandas, NumPy, Matplotlib, Scikit-learn
- **Models**: Linear Regression, Ridge Regression

###  Key Features:
- Feature engineering from academic indicators
- Prediction for new student profile
- Scatter plot of actual vs predicted scores
- Model coefficient analysis

###  Outcome:
Showcases regression modeling for educational analytics, with clear visualization and performance comparison.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/82ce7252-cf79-494d-bac4-bc9dbe45e90d" />


---

##  Project: Logistic Regression with and without Feature Scaling

###  Project Description:
This project explores the impact of **feature scaling** on logistic regression coefficients using a simple pass/fail dataset based on study and sleep hours.

###  Objectives:
- Train logistic regression models with and without scaling
- Compare model coefficients
- Highlight the importance of preprocessing

###  Tools & Technologies:
- **Python**: Core language
- **Libraries**: Pandas, Scikit-learn
- **Models**: Logistic Regression
- **Preprocessing**: StandardScaler

###  Key Features:
- Coefficient comparison table
- Demonstrates scaling impact on model interpretation

###  Outcome:
Reinforces the importance of feature scaling in classification tasks and model interpretability.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/30c67457-6f21-436f-bbf2-d7d48e3200c1" />


---

##  Project: ROC Curve Comparison of Classification Models

###  Project Description:
This project compares the performance of **Logistic Regression**, **Decision Tree**, and **Random Forest** classifiers using ROC curves and AUC scores.

###  Objectives:
- Train multiple classifiers on synthetic data
- Evaluate models using ROC curves
- Compare AUC scores for model selection

###  Tools & Technologies:
- **Python**: Core language
- **Libraries**: Pandas, Matplotlib, Scikit-learn
- **Models**: Logistic Regression, Decision Tree, Random Forest

###  Key Features:
- ROC curve visualization for each model
- AUC score comparison table
- Clear performance benchmarking

###  Outcome:
Provides a visual and quantitative comparison of classification models, aiding in model selection for binary tasks.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/b2f4b919-0c2d-43f4-87bc-4a5631afd0f8" />


---

##  Project: Ridge Coefficient Analysis on California Housing Data

###  Project Description:
This project investigates how Ridge Regression coefficients behave across varying regularization strengths using the **California Housing dataset**. It demonstrates the impact of the **alpha parameter** on model weights, helping visualize the trade-off between bias and variance in linear models.

###  Objectives:
- Load and preprocess California housing data
- Apply **feature scaling** for consistent coefficient comparison
- Train Ridge Regression models across a range of alpha values
- Visualize how coefficients shrink as regularization increases

###  Tools & Technologies:
- **Python**: Core language
- **Libraries**: NumPy, Matplotlib, Scikit-learn
- **Model**: Ridge Regression
- **Dataset**: `fetch_california_housing` from Scikit-learn

###  Key Features of the Workflow:
- **Data Loading**: Retrieved housing features and target values
- **Scaling**: Standardized features for fair coefficient comparison
- **Alpha Sweep**: Trained Ridge models across 100 alpha values from 0.001 to 1000
- **Visualization**:
  - Line plot of coefficients vs alpha
  - Log-scaled x-axis to highlight regularization effects

###  Outcome:
This project offers a clear visual understanding of how Ridge Regression penalizes large coefficients as alpha increases. It reinforces the concept of regularization in linear models and supports model tuning for better generalization.

<img width="1920" height="1080" alt="image" src="https://github.com/user-attachments/assets/f171a633-d729-48cc-86d0-a815be314ee6" />




