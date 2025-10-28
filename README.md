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


---

#  Employer Salary Prediction using Machine Learning

##  Project Overview
This project aims to analyze employer-related data and **predict employee salaries** using machine learning techniques. By leveraging features such as experience, job title, education level, and location, the model helps HR departments make **data-driven decisions** regarding compensation and workforce planning.



##  Problem Statement
The goal is to build a regression model that can accurately estimate an employee’s salary based on various attributes. This enables:
- Fair and consistent salary benchmarking
- Identification of salary outliers
- Strategic workforce budgeting



##  Dataset Description
- **Source**: `Employers_data.csv`
- **Size**: 10,000 records × 10 columns
- **Features**:
  - `Employee_ID`, `Name`, `Age`, `Gender`, `Department`, `Job_Title`, `Experience_Years`, `Education_Level`, `Location`
- **Target**:
  - `Salary`



##  Data Preprocessing
- Checked for **missing values** and **duplicates** (none found)
- Removed **outliers** using IQR method for numeric columns
- Applied **one-hot encoding** to categorical features (`Gender`, `Department`, etc.)
- Final encoded dataset shape: **(10000, 9887)**



##  Exploratory Data Analysis (EDA)
Visualizations included:
- **Boxplots** for numeric features to detect outliers
- **Heatmap** to analyze feature correlations
- **Histograms** and **violin plots** for salary distribution
- **Scatter plots** for relationships like `Experience vs Salary`
- **Pairplot** for `Age`, `Experience_Years`, and `Salary`


##  Model Building
Two regression models were trained and evaluated:
- **Linear Regression**
- **Random Forest Regressor**

###  Evaluation Metrics
| Model              | R² Score | MAE     | MSE        | RMSE    |
|-------------------|----------|---------|------------|---------|
| Linear Regression | 0.9861   | 4294.91 | 29.2M      | 5404.39 |
| Random Forest     | 0.9907   | 3519.85 | 19.5M      | 4419.22 |

✅ **Random Forest** performed better and was saved as the final model.

---

##  Model Deployment
The best-performing model was serialized using `joblib`:
```python
joblib.dump(best_model, 'best_salary_model.pkl')
```



##  Project Structure
```
├── Employers_data.csv
├── salary_prediction.ipynb / .py
├── best_salary_model.pkl
├── README.md
```



##  Future Improvements
- Integrate with a web dashboard (e.g., Streamlit or Flask)
- Add feature selection and hyperparameter tuning
- Deploy as an API for real-time salary prediction


<img width="1018" height="694" alt="Screenshot 2025-10-22 230945" src="https://github.com/user-attachments/assets/c756a0b5-83cd-42ea-9dcb-71ddfc973a97" />


<img width="743" height="476" alt="Screenshot 2025-10-22 231120" src="https://github.com/user-attachments/assets/8b7b8f94-fd4e-4000-8601-ed1da8052078" />


<img width="909" height="484" alt="Screenshot 2025-10-22 231137" src="https://github.com/user-attachments/assets/67116b32-bedc-46a4-8be4-92398eae100d" />


<img width="739" height="493" alt="Screenshot 2025-10-22 231217" src="https://github.com/user-attachments/assets/486ebd39-54fc-4e58-b215-7bbfbac08e7c" />


<img width="934" height="538" alt="Screenshot 2025-10-22 231236" src="https://github.com/user-attachments/assets/1c004d7c-765e-45c9-9d59-6f23f00d53b3" />


<img width="922" height="892" alt="Screenshot 2025-10-22 231334" src="https://github.com/user-attachments/assets/134aac4e-ddce-4b95-9c8d-dc922a5b0538" />

<img width="435" height="837" alt="Screenshot 2025-10-22 231742" src="https://github.com/user-attachments/assets/0e714701-b8b5-4d90-ac18-930877c90175" />

<img width="428" height="333" alt="Screenshot 2025-10-22 231907" src="https://github.com/user-attachments/assets/5a7fbd81-0e86-498d-a110-5afc7b00f493" />

