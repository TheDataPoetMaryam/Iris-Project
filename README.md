Iris Species Classification using Decision Tree
📌 Project Overview

This project implements a Decision Tree Classifier to predict the species of iris flowers based on their morphological features. The pipeline covers:

Exploratory Data Analysis (EDA)
Outlier removal
Feature correlation analysis
Hyperparameter tuning with cross-validation
Cost Complexity Pruning (CCP)
Model evaluation (confusion matrix, accuracy, learning curve)
Model explainability using SHAP
Interactive Streamlit application for predictions

The project uses the classic Iris dataset.

Iris Project/
├── predictor.py            # Streamlit app for predictions
├── iris_model.ipynb        # Main training + evaluation script
├── Iris.csv                # Dataset
└── README.md               # Documentation

⚙️ Methodology

1) Exploratory Data Analysis (EDA)

Histograms and boxplots to explore distributions
Correlation heatmap

2) Data Cleaning

Outlier removal using IQR method
Balanced species distribution

3) Model Training

Decision Tree with GridSearchCV for hyperparameter tuning
Cost Complexity Pruning (CCP Alpha optimization)

4) Evaluation Metrics

Accuracy, confusion matrix, classification report
Learning curve for bias-variance tradeoff
Feature importance plots
SHAP summary plots for interpretability

5) Deployment

Trained model saved with joblib
Streamlit app for user interaction and predictions

🚀 How to Run
Prerequisites:
1) Python 3.8+
2) Install dependencies:
   pip install -r requirements.txt
Training the Model
Run the training script / notebook:
   python iris_model.py

This will:
1) Train and prune the Decision Tree
2) Generate evaluation plots
3) Save the trained model in Outputs/
   Running the Streamlit App
           streamlit run predictor.py
You can then enter flower measurements and get the predicted species with confidence scores.

📊 Results
1) Achieved high test accuracy with pruned Decision Tree
2) Visualizations of decision boundaries, confusion matrix, feature importance
3) SHAP explainability demonstrates feature contributions.

📚 Dependencies:

1) pandas, numpy
2) matplotlib, seaborn
3) scikit-learn
4) shap
5) joblib
6) streamlit

✍️ Authors

Maryam Shaikh
M.Sc. Applied Statistics
