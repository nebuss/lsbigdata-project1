import numpy  as np
import matplotlib.pyplot as plt

a=2
b=3
c=5

x = np.linspace(-8, 8, 100)
y= a * x**2 + b* x + c

plt.plot(x, y, color="black")
