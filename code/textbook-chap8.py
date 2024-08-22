import pandas as pd

mpg=pd.read_csv("data/mpg.csv")
mpg.shape

# !pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt

plt.clf()
plt.figure(figsize=(5, 4)) # 사이즈 조정
sns.scatterplot(data=mpg, 
                x="displ", y="hwy",
                hue="drv") \
   .set(xlim=[3, 6], ylim=[10, 30])
plt.show()

# 막대그래프
# mpg["drv"].unique()
df_mpg=mpg.groupby("drv", as_index=False) \
          .agg(mean_hwy=('hwy', 'mean'))
df_mpg
plt.clf()
sns.barplot(data=df_mpg.sort_values("mean_hwy"),
            x = "drv", y = "mean_hwy",
            hue = "drv")
plt.show()

mpg

# 교재 8장, p.212
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

economics=pd.read_csv("./data/economics.csv")
economics.head()

economics.info()
sns.lineplot(data=economics, x="date", y="unemploy")
plt.show()
plt.clf()

economics["date2"]=pd.to_datetime(economics["date"])
economics
economics.info()

economics[["date", "date2"]]
economics["date2"].dt.year
economics["date2"].dt.month
economics["date2"].dt.day
economics["date2"].dt.month_name()
economics["date2"].dt.quarter
economics["quarter"]=economics["date2"].dt.quarter
economics[["date2", "quarter"]]
# 각 날짜는 무슨 요일인가?
economics["date2"].dt.day_name()
economics["date2"] + pd.DateOffset(days=30)
economics["date2"] + pd.DateOffset(months=1)
economics["date2"].dt.is_leap_year # 윤년 체크
economics["year"]=economics["date2"].dt.year

sns.lineplot(data=economics, 
             x='year', y='unemploy',
             errorbar=None)
sns.scatterplot(data=economics, 
             x='year', y='unemploy', s=2)
plt.show()
plt.clf()
economics.head(10)

my_df=economics.groupby("year", as_index=False) \
         .agg(
            mon_mean=("unemploy", "mean"),
            mon_std=("unemploy", "std"),
            mon_n=("unemploy", "count")
         )
my_df
mean + 1.96*std/sqrt(12)
my_df["left_ci"]=my_df["mon_mean"] - 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df["right_ci"]=my_df["mon_mean"] + 1.96 * my_df["mon_std"] / np.sqrt(my_df["mon_n"])
my_df.head()

import matplotlib.pyplot as plt

x = my_df["year"]
y = my_df["mon_mean"]
# plt.scatter(x, y, s=3)
plt.plot(x, y, color="black")
plt.scatter(x, my_df["left_ci"], color="blue", s=1)
plt.scatter(x, my_df["right_ci"], color="blue", s=1)
plt.show()
plt.clf()

# 학교
# 박사과정 마지막 (퀄 시험)
# 졸업시험 (1년 1회 실패, 1달 )
# (2번째, 세모 통과 - 인터뷰) 금, 월요일 아침
# 월요일
# 9시 인터뷰, 9시 30분 수업
# 10시
