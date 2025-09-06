import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("First five rows of the dataset:")
print(df.head())

print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())

print("\nDescriptive statistics:")
print(df.describe())

grouped_means = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped_means)

plt.figure(figsize=(8, 5))
sns.lineplot(data=grouped_means['petal length (cm)'], marker='o')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.grid(True)
plt.tight_layout()
plt.savefig('line_chart.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.barplot(x=grouped_means.index, y=grouped_means['sepal width (cm)'])
plt.title('Average Sepal Width per Species')
plt.xlabel('Species')
plt.ylabel('Sepal Width (cm)')
plt.tight_layout()
plt.savefig('bar_chart.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.histplot(df['petal length (cm)'], bins=20, kde=True)
plt.title('Distribution of Petal Length')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.savefig('scatter_plot.png')
plt.close()
