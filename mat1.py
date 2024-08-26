import numpy as np
# 5행 2열의 행렬 생성
x = np.arange(1, 11).reshape((5, 2)) * 2
print("원래 행렬 x:\n", x)
# 행렬을 전치
transposed_x = x.transpose()
print("전치된 행렬 x:\n", transposed_x)
