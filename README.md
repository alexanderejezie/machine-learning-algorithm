# 🎯 Microsoft Malware Dataset Project.

## Dataset Description
This dataset was derived from the original copy and simplified for learning purposes. It contains a set of machines, which run Microsoft Windows OS. The goal of this exercise is to predict a Windows machine’s probability of getting infected by various families of malware, based on different properties of that machine.

➡️ [Dataset link](https://drive.google.com/file/d/13hQ-46e6Q7zvLgx8jmQ2R_HwcvKjhpCA/view?usp=sharing)

➡️ [Columns Explanation](https://docs.google.com/spreadsheets/d/1fB5antF52_hfbjRh4fRpV9q2PVLaQVTeerA5e4YrD0A/edit?usp=drive_link)

## ℹ️ Instructions

### Part 1: Supervised Learning

#### Import Data and Perform Basic Data Exploration
1. Import your data and perform basic data exploration.
2. Display general information about the dataset.
3. Create a pandas profiling report to gain insights into the dataset.
4. Handle missing and corrupted values.
5. Remove duplicates, if they exist.
6. Handle outliers, if they exist.
7. Encode categorical features.
8. Prepare your dataset for the modelling phase.

#### Apply Decision Tree and Evaluate
1. Apply a Decision tree model.
2. Plot its ROC curve.
3. Try to improve your model performance by changing the model hyperparameters.

### Part 2: Unsupervised Learning

1. Drop the target variable.
2. Apply K-means clustering and plot the clusters.
3. Find the optimal K parameter.
4. Interpret the results.

# 🛠️ Tools and Libraries
- Python: Core programming language
- Pandas: Data manipulation and analysis
- NumPy: Numerical computations
- Matplotlib: Data visualization
- Seaborn: Statistical data visualization
- SciPy: Scientific computing
- Scikit-learn: Machine learning algorithms

## Example Code
Here is a snippet of code to get you started with data exploration:

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv('path_to_dataset.csv')

# Handle missing values
df = df.dropna()

# Encode categorical features
le = LabelEncoder()
df['column_name'] = le.fit_transform(df['column_name'])

# Split the dataset into training and test sets
X = df.drop('target_variable', axis=1)
y = df['target_variable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Apply Decision Tree
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
```
