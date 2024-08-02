import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
df = sns.load_dataset('titanic')
df

sns.countplot(data=df, x='sex')
plt.show()
plt.clf()

sns.countplot(data=df , x='class', hue='alive')
plt.show()

import sklearn import metrics
metrics.accuracy_score()

