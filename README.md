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


# Use Case Practiced: Loan Repayment Prediction

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



# Heart Disease Prediction Using Decision Tree

##  Problem Statement
The goal of this project is to build a classification model using a Decision Tree to predict whether a patient is likely to have heart disease based on medical and diagnostic features.


##  Dataset
- **Source**: Kaggle Heart Disease UCI Dataset(https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- **Target variable**: `target`  indicates presence of heart disease:
  - `0` —> No disease
  - `1` —> Disease
- **Total samples**: 1025
- **Features**: age, sex, cp (chest pain type), trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal


##  Preprocessing
- No missing values in the dataset
- Features were used as is (dataset is already numeric and clean)
- Split into 80% training and 20% test set using `train_test_split`


##  Model Summary
- **Model used**: Decision Tree Classifier
- **Why Decision Tree?** Easy to interpret and visualize, performs well with structured data
- Training was fast and interpretable


##  Evaluation Results
- **Accuracy**: 80.5%
- **Confusion Matrix**:
  - True Positives (Disease predicted correctly): 138
  - True Negatives (No disease predicted correctly): 110
  - False Positives: 49
  - False Negatives: 11

###  Classification Report

| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| No Disease | 0.91      | 0.69   | 0.79     |
| Disease    | 0.74      | 0.93   | 0.82     |


##  Post-Model Visualizations
- **Decision Tree Plot**: Displays the full flow of how the model splits data
- **Feature Importance Plot**: Top 10 features that influenced predictions the most
- **Confusion Matrix Heatmap**: Visual view of correct vs incorrect predictions


##  Insights
- The model performs very well in detecting heart disease (recall = 93%), which is important in healthcare scenarios.
- A few false positives exist, but better to flag at-risk patients than miss dangerous cases.
- Decision Trees provide strong interpretability, especially helpful in understanding what factors lead to high risk.




#  Iris Flower Classification

##  Problem Statement
The goal of this project is to classify iris flowers into one of three species based on the measurements of their petals and sepals. A Decision Tree classifier is used to learn the decision boundaries between species.


##  Dataset
- **Source**: `sklearn.datasets.load_iris()`
- **Target variable**: `species`
  - `setosa`
  - `versicolor`
  - `virginica`
- **Total samples**: 150
- **Features**:
  - `sepal length (cm)`
  - `sepal width (cm)`
  - `petal length (cm)`
  - `petal width (cm)`


##  Preprocessing
- The dataset was already clean and balanced.
- A new column `species` was created by mapping the target (0, 1, 2) to species names.
- Class distribution was visualized using `countplot`, all three classes had equal count (50 each).
- Data was split into train and test sets using `train_test_split` with `random_state=100`.


##  Model Summary
- **Model used**: Decision Tree Classifier
- **Why Decision Tree?** Intuitive and performs well on structured data
- **Random State**: Initially 42 (accuracy = 1.0), changed to 100 for a more realistic split
- **Final Accuracy**: `96.7%`


##  Evaluation Results
- **Accuracy**: 96.7%
- **Confusion Matrix**:
  - Class 0 (Setosa): All correctly predicted
  - Class 1 (Versicolor): 1 misclassified
  - Class 2 (Virginica): All correctly predicted

###  Classification Report

| Class      | Precision | Recall | F1-score |
|------------|-----------|--------|----------|
| Setosa     | 1.00      | 1.00   | 1.00     |
| Versicolor | 1.00      | 0.83   | 0.91     |
| Virginica  | 0.93      | 1.00   | 0.96     |


##  Post-Model Visualizations
- **Decision Tree Diagram**: Displays the logical path used to classify each flower
- **Feature Importance Bar Chart**: Shows which measurements contributed most to predictions
- **Confusion Matrix Heatmap**: Visualizes correct and incorrect predictions across species


##  Insights
- The Iris dataset is linearly separable, especially for Setosa, which is easy to classify.
- The model performs excellently even when shuffled with a different random state.
- Petal length and petal width were the most important features based on the decision tree's learned splits.






#  Titanic Survival Prediction 

This project uses a Decision Tree Classifier to predict which passengers survived the Titanic disaster. It walks through the full machine learning workflow, from data cleaning and EDA to building a pipeline and visualizing model outputs.


##  Dataset

The dataset used is the Titanic dataset, typically available from Kaggle(https://www.kaggle.com/competitions/titanic/data).

Each row contains information about a passenger including age, sex, fare, passenger class, and survival status.


##  Problem Statement

The objective is binary classification:  
**Predict whether a passenger survived (`1`) or not (`0`)** using the available features.


##  Features Used

After cleaning and preprocessing, the features used were:

| Feature    | Description                             |
|------------|-----------------------------------------|
| `Pclass`   | Passenger Class (1st, 2nd, 3rd)          |
| `Sex`      | Gender (male/female)                    |
| `Age`      | Age of the passenger                    |
| `Fare`     | Ticket fare                             |
| `Embarked` | Port of Embarkation (C, Q, S)           |

**Dropped columns**:  
- `Cabin` –> too many missing values  
- `Name` –> mostly for identification  
- `Ticket` –> not meaningful for prediction


##  Data Preprocessing

- **Missing Values**:
  - Filled `Age` with the median
  - Filled `Embarked` with the mode (Southampton)
- **Encoding**:
  - `Sex` → Ordinal Encoding
  - `Pclass` → Treated as ordinal, used Ordinal Encoding
  - `Embarked` → One-Hot Encoding
- **Scaling**:
  - Scaled numerical features: `Age`, `Fare` using `StandardScaler`
- **Pipeline**:
  - Combined encoders and scaler using `ColumnTransformer`
  - Wrapped with a `Pipeline` to train the `DecisionTreeClassifier`


##  Exploratory Data Analysis (EDA)

### 1. Survival by Gender
- Majority of females survived.
- Majority of males did not survive.

### 2. Survival by Passenger Class
- 1st Class passengers had higher survival rates.
- 3rd Class passengers had the highest death rate.

### 3. Age vs Survival (Histogram)
- Highest survival between ages **26–30**
- Most people were in the **20–40** age range
- Very elderly had low survival

### 4. Fare vs Survival (Boxplot)
- Survivors had higher fare values and more outliers.
- Non-survivors mostly paid lower fares.


##  Model

- Model Used: DecisionTreeClassifier
- Train-Test Split: 80-20
- Random State: `42` (for reproducibility)


##  Model Evaluation

**Final Accuracy**: `0.7598`

###  Confusion Matrix
[[83 22]
[21 53]]

###  Classification Report

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| 0     | 0.80      | 0.79   | 0.79     | 105     |
| 1     | 0.71      | 0.72   | 0.71     | 74      |

- Accuracy: 0.76  
- Macro Avg: 0.75  
- Weighted Avg: 0.76


##  Post-Model Visualizations

### 1. Confusion Matrix Display
Visualizes the number of true/false positives and negatives in a colored matrix.

### 2. Feature Importance Plot
Displays the most important features for survival prediction.  
Top features included:
- `Sex`
- `Fare`
- `Pclass`

### 3. Decision Tree Visualization
Used `plot_tree()` to visualize the internal decision-making process of the classifier.

Key takeaways:
- The tree splits early on `Sex` and `Fare`
- Many branches confirm females and high fare passengers had better survival chances


##  Insights

- Sex is the strongest predictor. Most females survived.
- Passenger Class impacts survival: 1st class fared better.
- Fare influences survival, higher fares had more survivors.
- Decision Tree Visualization gives interpretable rules for survival predictions.







#  Wine Quality Classification (Red Wine Dataset)

This project is a machine learning classification task that predicts the quality of red wine using a Decision Tree Classifier. It includes full data preprocessing, exploratory data analysis (EDA), model training, hyperparameter tuning, and post-model interpretability visualizations.


##  Dataset

- **Source**: UCI Machine Learning Repository  
- **Filename**: `winequality-red.csv`  
- **Samples**: 1599  
- **Features**: 11 physicochemical attributes + quality label  
- **Target**: `quality_label` (0 = poor quality, 1 = good quality)


##  Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn


##  Data Preparation

1. Loaded dataset with `;` as the separator.
2. Mapped quality scores into a binary classification:
   - `quality >= 6`: Good Quality (`1`)
   - `quality < 6`: Poor Quality (`0`)
3. Separated features (`X`) and target (`y`).
4. Standardized the features using `StandardScaler`.
5. Split the data: 80% training / 20% testing


##  Exploratory Data Analysis (EDA)

1. **Countplot**:
   - Compared distribution of poor vs good wine.
   - Adjusted threshold from 7→6 to handle class imbalance.

2. **Correlation Heatmap**:
   - Alcohol showed moderate positive correlation with quality.
   - Volatile acidity had weak negative correlation.
   - Sulphates and citric acid showed weak positive relationships.

3. **Boxplots**:
   - Alcohol vs Quality: Good quality wines had higher alcohol content and greater variability.
   - Volatile Acidity vs Quality: Poor quality wines had higher volatile acidity.


##  Model Training

- Algorithm: Decision Tree Classifier  
- Criterion: Entropy  
- Initial Parameters:
  - `max_depth = 3`
  - `min_samples_leaf = 5`
  - `random_state = 100`


## ️ Hyperparameter Tuning

- Tuned hyperparameters manually:
  - Improved model accuracy to **0.78**
  - Found that increasing `max_depth` and controlling `min_samples_leaf` helped generalization.

- Tried class balancing with:
  - `class_weight = 'balanced'`: Accuracy dropped to **0.75**
  - SMOTE: Accuracy around **0.77**


##  Evaluation Results

- Final Accuracy: `0.7812`
- Confusion Matrix:

  |               | Predicted Poor | Predicted Good |
  |---------------|----------------|----------------|
  | Actual Poor   | 114            | 34             |
  | Actual Good   | 36             | 136            |

- Classification Report:

  | Class         | Precision | Recall | F1-score | Support |
  |---------------|-----------|--------|----------|---------|
  | Poor Quality  | 0.76      | 0.77   | 0.77     | 148     |
  | Good Quality  | 0.80      | 0.79   | 0.80     | 172     |
  | **Accuracy**  |           |        | **0.78** | 320     |


##  Post-Model Visualizations

1. Confusion Matrix Heatmap: Clear visualization of true/false predictions.
2. Feature Importance (Good Quality):
   - Alcohol
   - Volatile Acidity
   - Sulphates
   - Density
   - pH

3. Feature Importance (Poor Quality):
   - Highlighted least impactful features for good quality classification.

4. Decision Tree Plot:
   - Visualized tree structure with decision thresholds, sample counts, and classes.


##  Conclusion

- Model performed well after tuning with an accuracy of ~78%.
- Key features like alcohol and volatile acidity were instrumental in predicting quality.
- Best result came from tuning without explicit resampling or reweighting.






