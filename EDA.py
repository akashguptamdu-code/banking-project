import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load dataset
df = pd.read_csv("European_Bank.csv")

# 2. cleaning dataset
df.head()
df.shape
df.info()
df.describe()

# 3. Check for missing values
df.isnull().sum()
df.duplicated().sum()

# 4. target variable
df['Exited'].value_counts()

df['Exited'].value_counts(normalize=True) * 100

# 5.graph 
sns.countplot(x='Exited', data=df)
plt.title('Customer Churn Distribution')
plt.xlabel('Exited (0 = Retained, 1 = Churned)')
plt.ylabel('Number of Customers')
plt.show()

# geography vs churn
geo_churn = df.groupby('Geography')['Exited'].mean() * 100
print(geo_churn)

geo_churn.plot(kind='bar')
plt.title('Churn Rate by Geography')
plt.xlabel('Geography')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=0)
plt.show()

# active members vs churn
activity_churn = df.groupby('IsActiveMember')['Exited'].mean() * 100
print(activity_churn)

activity_churn.plot(kind='bar')
plt.title('Churn Rate by Active Membership')
plt.xlabel('Active Member (0 = No, 1 = Yes)')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=0)
plt.show()

# age vs churn
sns.boxplot(x='Exited', y='Age', data=df)
plt.title('Age Distribution by Churn Status')
plt.xlabel('Exited')
plt.ylabel('Age')
plt.show()

print(df.groupby('Exited')['Age'].mean())

# correlation heatmap
correlation = df.corr(numeric_only=True)
print(correlation)

plt.figure(figsize=(12, 8))

sns.heatmap(
    correlation,
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')
plt.show()

# gender vs churn
gender_churn = df.groupby('Gender')['Exited'].mean() * 100
print(gender_churn)

gender_churn.plot(kind='bar')
plt.title('Churn Rate by Gender')
plt.xlabel('Gender')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=0)
plt.show()

# credit score vs churn
sns.boxplot(x='Exited', y='CreditScore', data=df)
plt.title('Credit Score Distribution by Churn Status')
plt.xlabel('Exited')
plt.ylabel('Credit Score')
plt.show()

print(df.groupby('Exited')['CreditScore'].mean())

# tenure vs churn
tenure_churn = df.groupby('Tenure')['Exited'].mean() * 100
print(tenure_churn)

tenure_churn.plot(kind='bar')
plt.title('Churn Rate by Tenure')
plt.xlabel('Tenure')
plt.ylabel('Churn Rate (%)')
plt.show()

# number of products vs churn
product_churn = df.groupby('NumOfProducts')['Exited'].mean() * 100
print(product_churn)

product_churn.plot(kind='bar')
plt.title('Churn Rate by Number of Products')
plt.xlabel('Number of Products')
plt.ylabel('Churn Rate (%)')
plt.show()

# balance vs churn
sns.boxplot(x='Exited', y='Balance', data=df)
plt.title('Balance by Churn Status')
plt.show()

# has credit card vs churn
credit_card_churn = df.groupby('HasCrCard')['Exited'].mean() * 100
print(credit_card_churn)

credit_card_churn.plot(kind='bar')
plt.title('Churn Rate by Credit Card Ownership')
plt.xlabel('Has Credit Card (0 = No, 1 = Yes)')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=0)
plt.show()

# estimated salary vs churn
sns.boxplot(x='Exited', y='EstimatedSalary', data=df)
plt.title('Estimated Salary by Churn Status')
plt.show()