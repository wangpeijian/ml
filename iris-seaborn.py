import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("./dataset/iris.data", names=['ec', 'ek', 'bc', 'bk', 'class'])

sns.pairplot(df, hue='class', diag_kind='hist')
sns.pairplot(df, hue='class')

plt.show()

