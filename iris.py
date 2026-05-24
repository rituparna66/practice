from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

iris = load_iris()

df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)

df_iris["species"] = iris.target_names[iris.target]

sns.pairplot(df_iris, hue="species", diag_kind="kde")
plt.suptitle("Iris Pairplot", y=1.02)
plt.show()
print(df_iris.head())
