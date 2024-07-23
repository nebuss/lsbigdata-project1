import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
mpg = pd.read_csv('data/mpg.csv')

plt.clf()
plt.figure(figsize=(5, 4))
sns.scatterplot(data = mpg, 
                x = 'displ', y = 'hwy',
                hue ='drv') \
   .set(xlim =[3, 6], ylim=[10, 30])
plt.show()

# 막대그래프
plt.clf()
df_mpg = mpg.groupby('drv', as_index = False) \
   .agg(mean_hwy = ('hwy', 'mean')) 

df_mpg

sns.barplot(data = df_mpg, x = 'drv', y='mean_hwy')
plt.show()


df_mpg = df_mpg.sort_values('mean_hwy', ascending='False')

sns.barplot(data=df_mpg, x= 'drv', y='mean_hwy')
plt.show()
-------------------------------------------------------

df_mpg = mpg.groupby('drv', as_index = False) \
            .agg(n = ('drv', 'count'))

df_mpg             

sns.barplot(data= df_mpg, x = 'drv', y='n')
plt.show()
plt.clf()
sns.countplot(data= mpg, x='drv')
plt.show()


import plotly.express as px
px.scatter(data_frame =mpg, x = 'cty', y= 'hwy', color = 'drv')
