import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load the data
df = pd.read_csv("patient_feedback_data.csv")
print('First 5 rows:')
print(df.head())

# 2. Handle missing and duplicate values
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# Fill missing side effects with 'None'
df['Side_Effect_List'] = df['Side_Effect_List'].fillna('None')
print(df.isnull().sum())


# Save cleaned data
"""df.to_csv("patient_feedback_cleaned.csv", index=False)

# 3. Data Visualization

# Distribution of drug names
plt.figure(figsize=(10,4))
sns.countplot(x='Drug_Name', data=df)
plt.xticks(rotation=45)
plt.title('Drug Name Distribution')
plt.tight_layout()
plt.show()

# Distribution of side effects
plt.figure(figsize=(10,4))
sns.countplot(x='Side_Effect_List', data=df)
plt.xticks(rotation=45)
plt.title('Side Effect Distribution')
plt.tight_layout()
plt.show()

# Distribution of effectiveness ratings
plt.figure(figsize=(8,4))
sns.histplot(df['Effectiveness_Rating'], bins=10, kde=True)
plt.title('Effectiveness Rating Distribution')
plt.xlabel('Effectiveness Rating')
plt.tight_layout()
plt.show()

# Correlation heatmap (numeric columns only)
plt.figure(figsize=(10,8))
numeric_cols = df.select_dtypes(include=[np.number]).columns
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (All Numeric Columns)')
plt.tight_layout()
plt.show()

# Boxplot: Effectiveness by Drug
plt.figure(figsize=(10,5))
sns.boxplot(x='Drug_Name', y='Effectiveness_Rating', data=df)
plt.xticks(rotation=45)
plt.title('Effectiveness Rating by Drug')
plt.tight_layout()
plt.show()

# Boxplot: Effectiveness by Gender
plt.figure(figsize=(6,4))
sns.boxplot(x='Gender', y='Effectiveness_Rating', data=df)
plt.title('Effectiveness Rating by Gender')
plt.tight_layout()
plt.show()

# Boxplot: Effectiveness by Age Group
age_bins = [0, 18, 30, 45, 60, 100]
age_labels = ['<18', '18-30', '31-45', '46-60', '60+']
df['age_group'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)

plt.figure(figsize=(8,4))
sns.boxplot(x='age_group', y='Effectiveness_Rating', data=df)
plt.title('Effectiveness Rating by Age Group')
plt.tight_layout()
plt.show()



# Pairplot for key numeric features
sns.pairplot(df[['Age', 'Effectiveness_Rating']], diag_kind='kde')
plt.suptitle('Pairplot of Age and Effectiveness Rating', y=1.02)
plt.show()






# Encode categorical feature: Drug_Name
X = pd.get_dummies(df[['Drug_Name']], drop_first=True)
y = df['Effectiveness_Rating']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# 5. Model Evaluation

print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Plot Actual vs Predicted
plt.figure(figsize=(8, 4))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Effectiveness Rating")
plt.ylabel("Predicted Effectiveness Rating")
plt.title("Actual vs Predicted Effectiveness (Linear Regression)")
plt.show()"""
