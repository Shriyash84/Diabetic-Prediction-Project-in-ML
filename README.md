# Diabetic-Prediction-Project Using ML

## Project Overview
The Diabetic Prediction Project aims to develop a machine learning model that predicts the likelihood of a patient having diabetes based on various health metrics. The model is trained on a dataset containing patient information, and it utilizes classification algorithms to make predictions.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib/Seaborn (for data visualization)
- Jupyter Notebook

## Dataset
The dataset used in this project is sourced from (https://www.kaggle.com/uciml/pima-indians-diabetes-database) (or specify another source). It contains the following features:
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- Body Mass Index (BMI)
- Diabetes Pedigree Function
- Age
- Outcome (1 for diabetes, 0 for no diabetes)

# Usage
Run the Jupyter Notebook to explore the dataset and view model performance:
jupyter notebook House_Price_Prediction.ipynb

# Model Training
The model is trained using various regression algorithms such as:
- Linear Regression
- Decision Trees
- Random Forest
The training process involves:

1 Data preprocessing (handling missing values, encoding categorical variables, normalization).
2 Splitting the dataset into training and testing sets.
3 Training the model and evaluating its performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

# Results
The final model achieved an RMSE of X on the test dataset. Detailed analysis and visualizations can be found in the notebook, showcasing feature importance and prediction distributions
