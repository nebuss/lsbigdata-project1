# 선형회귀 모델이용해서


house_train = pd.read_csv("data/train.csv")
my_df = house_train[["BedroomAbvGr", "SalePrice"]]

# 선형 회귀 모델 생성
model = LinearRegression()

x = np.array(my_df["BedroomAbvGr"]).reshape(-1,1)
y = my_df["SalePrice"]
# 모델 학습
model.fit(x, y) #자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x)

# 데이터와 회귀 직선 시각화
model.coef_
model.intercept_

plt.scatter(x, y, color='blue', label='실제 데이터')
plt.plot(x, y_pred, color='red', label='회귀 직선')
plt.xlabel('방의 개수')
plt.ylabel('가격')
plt.legend()
plt.show()
plt.clf()
# my_df['SalePrice'] =  y_pred
# sub = pd.read_csv("data/sample_submission.csv")
# sub["SalePrice"] = my_df['SalePrice']

# sub.to_csv("sub_prediction085.csv", index=False)
