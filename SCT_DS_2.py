 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load Titanic dataset (Excel or CSV)
df = pd.read_excel('titanic.xlsx')

# -----------------------------
# Data Cleaning
# -----------------------------
# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop rows where Embarked is missing
df.dropna(subset=['Embarked'], inplace=True)

# -----------------------------
# Exploratory Data Analysis
# -----------------------------
# Gender-wise survival visualization
sns.countplot(x='Survived', hue='Sex', data=df)
plt.title('Survival Count by Gender')
plt.show()

# Class-wise survival visualization
sns.countplot(x='Survived', hue='Pclass', data=df)
plt.title('Survival Count by Passenger Class')
plt.show()

# Correlation Heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
