🌸 Iris Dataset Analysis and Machine Learning Pipeline
This project provides a complete data science and machine learning pipeline using the classic Iris dataset. It includes data exploration, visualization, unsupervised learning (clustering and outlier detection), supervised classification, model evaluation, hyperparameter tuning, and ensemble learning.

📁 Project Structure
1. Data Loading and Preparation
Load the Iris dataset using scikit-learn.

Convert to a Pandas DataFrame.

Add class labels and check for missing values.

2. Exploratory Data Analysis (EDA)
Pairwise plots using seaborn.pairplot.

Correlation heatmap.

Feature-wise histograms.

3. Unsupervised Learning
K-Means Clustering:

Clusters the data into 3 groups (expected species).

Visual comparison with true species labels.

Outlier Detection:

Using Isolation Forest and DBSCAN.

Results visualized in PCA-reduced 2D space.

Clustering Evaluation:

Confusion matrix between true species and K-Means labels.

Silhouette Score for cluster cohesion.

4. Supervised Learning
Baseline Model: Logistic Regression

Train/test split (80/20 stratified).

Accuracy, classification report, and confusion matrix.

Model Comparison:

Evaluates SVM, Decision Tree, Random Forest, and Gradient Boosting using 10-fold cross-validation.

Model Tuning:

Grid search for best Random Forest parameters.

Ensemble Learning:

Voting Classifier combining SVM, tuned Random Forest, and Gradient Boosting (soft voting).

🚀 How to Run
Requirements
Install dependencies using pip:

==========================================================
pip install pandas scikit-learn matplotlib seaborn
==========================================================

Run the Script
python iris_analysis.py


This will execute the full pipeline and generate all plots and evaluation metrics.

📊 Output
The script provides:

Pairplot and heatmap visualizations.

Histograms of each feature.

K-Means clustering visualization.

PCA scatterplots highlighting outliers.

Confusion matrices and accuracy reports.

Silhouette Score for clustering quality.

Cross-validation comparison of classifiers.

Best hyperparameters from grid search.

Ensemble model performance.

📦 Technologies Used
Python

pandas, matplotlib, seaborn

scikit-learn (EDA, ML models, clustering, evaluation)

📚 Dataset
The Iris dataset is a classical dataset containing 150 samples of iris flowers across 3 species (setosa, versicolor, virginica) with 4 features: sepal length, sepal width, petal length, and petal width.

✅ License
This project is licensed under the MIT License
