Iris Species Classification using Decision Tree
<br>
# Iris Species Classification using Decision Tree  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) 
<br>
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)  
![Streamlit](https://img.shields.io/badge/Streamlit-App-brightgreen)  
![License](https://img.shields.io/badge/License-MIT-yellow)  


📌 Project Overview

This project implements a Decision Tree Classifier to predict the species of iris flowers based on their morphological features. The pipeline covers:

1) Exploratory Data Analysis (EDA)
2) Outlier removal
3) Feature correlation analysis
4) Hyperparameter tuning with cross-validation
5) Cost Complexity Pruning (CCP)
6) Model evaluation (confusion matrix, accuracy, learning curve)
7) Model explainability using SHAP
8) Interactive Streamlit application for predictions

The project uses the classic Iris dataset.

Iris Project/
<br>
├── predictor.py            # Streamlit app for predictions
<br>
├── iris_model.ipynb        # Main training + evaluation script
<br>
├── Iris.csv                # Dataset
<br>
└── README.md               # Documentation

⚙️ Methodology

1) Exploratory Data Analysis (EDA)

Histograms and boxplots to explore distributions
<br>
Correlation heatmap

2) Data Cleaning

Outlier removal using IQR method
<br>
Balanced species distribution

3) Model Training

Decision Tree with GridSearchCV for hyperparameter tuning
<br>
Cost Complexity Pruning (CCP Alpha optimization)

4) Evaluation Metrics

i) Accuracy, confusion matrix, classification report
<br>
ii) Learning curve for bias-variance tradeoff
<br>
iii) Feature importance plots
<br>
iv) SHAP summary plots for interpretability

5) Deployment

Trained model saved with joblib
<br>
Streamlit app for user interaction and predictions

🚀 How to Run
Prerequisites:
1) Python 3.8+
2) Install dependencies:
   pip install -r requirements.txt
   <br>
Training the Model
<br>
Run the training script / notebook:
<br>
   python iris_model.py

This will:
1) Train and prune the Decision Tree
2) Generate evaluation plots
3) Save the trained model in Outputs/
   Running the Streamlit App
   <br>
           streamlit run predictor.py
   <br>
You can then enter flower measurements and get the predicted species with confidence scores.

📊 Results
1) Achieved high test accuracy with pruned Decision Tree
2) Visualizations of decision boundaries, confusion matrix, feature importance
3) SHAP explainability demonstrates feature contributions.
4) classifier is designed to predict the correct class for new data inputs, fulfilling the project’s purpose.

📚 Dependencies:

1) pandas, numpy
2) matplotlib, seaborn
3) scikit-learn
4) shap
5) joblib
6) streamlit

Screenshot of APP
<br>
![App Screenshot](Iris_predictor_SS.png)

✍️ Authors

Maryam Shaikh
<br>
M.Sc. Applied Statistics
