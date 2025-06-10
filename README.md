# DecisionTree Learning Summary

Today, I explored and implemented a Decision Tree Classifier to predict loan repayment outcomes. Here’s a summary of the theoretical concepts and insights I gained during the process:

Understanding Decision Trees:

A Decision Tree is a tree-structured model used for decision-making and classification. Each internal node represents a test on a feature, each branch represents an outcome of the test, and each leaf node represents a class label (decision).

Advantages of Decision Trees:
	1.	Easy to understand and visualize: Decision trees are intuitive and less challenging to interpret.
	2.	Minimal data preparation required: They can work well without the need for normalization or scaling.
	3.	Handles both numerical and categorical data: Versatile across different data types.
	4.	Not sensitive to non-linear relationships: Their performance isn’t affected by non-linearity in data.

Disadvantages of Decision Trees:
	1.	Overfitting: They can create complex trees that perfectly fit training data, reducing generalization.
	2.	High variance: Small changes in data can lead to very different trees.
	3.	Low bias: Trees can model very complex relationships, sometimes too well (prone to overfitting).

Key Concepts:
	•	Entropy: A measure of randomness or impurity in a dataset. High entropy means the data is more mixed or uncertain.
	•	Information Gain: Measures the reduction in entropy after splitting a dataset.
        Formula ⇒ Gain = Entropy(before split) - Entropy(after split)
	•	Root Node: The topmost node of the tree, representing the first decision.
	•	Leaf Node: Terminal nodes that carry the final classification or prediction.
	•	Entropy Formula ⇒ Entropy = ∑(Pᵢ * log₂(Pᵢ)), Where Pᵢ is the probability of class i.


Use Case Practiced: Loan Repayment Prediction

Below is a step-by-step outline of the practical implementation I completed:

1. Data Acquisition
    Dataset:
	•	Source: https://github.com/pedromaiapatinha/Loan_Repayment_Prediction/blob/main/Loans_Dataset.csv
	•	Shape: 1000 rows, 6 columns
	•	Target: result(yes or no)

2. Library Imports
	I imported necessary libraries, including:
	•	pandas and numpy for data handling
	•	sklearn modules for model training and evaluation

3. Data Loading and Exploration
	•	Loaded the dataset using pd.read_csv() into a variable named loans.
	Performed exploratory checks using:
	•	loans.shape: to check the number of rows and columns
	•	len(loans): to confirm row count
	•	loans.head(): to preview the first few records

4. Feature and Target Separation
	•	Identified the target column (results, containing ‘yes’ or ‘no’) for classification.
	For features (X), I dropped:
	•	The results column
	•	A column named sum (which was artificially created and a direct function of other features)
*Including sum initially caused the model to output 100% accuracy, which indicated data leakage.

5. Train-Test Split
	•	Split the dataset into training and testing sets using train_test_split from sklearn.model_selection.
	Used:
	•	test_size=0.3
	•	random_state=100
	•	Experimented with different test sizes and random states to observe changes in accuracy.

6. Model Training
	•	Imported DecisionTreeClassifier from sklearn.tree.
	•	Created a classifier instance using entropy
    •   Trained the model on the training set 

7. Model Prediction
	•	Made predictions on the test set

8. Model Evaluation
	•	Imported and used accuracy_score from sklearn.metrics.
	•	Final accuracy score achieved: 93.67%
