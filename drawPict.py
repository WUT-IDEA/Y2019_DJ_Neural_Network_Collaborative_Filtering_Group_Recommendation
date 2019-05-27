#!/etc/bin/python
#coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(2000)
y = np.random.standard_normal((10, 2))
plt.figure(figsize=(7,5))
plt.plot(y[:,0], lw = 1.5,label = '1st')
plt.plot(y[:,1], lw = 1.5, label = '2st')
plt.plot(y, 'ro')
plt.grid(True)
plt.legend(loc = 0) #图例位置自动
plt.axis('tight')
plt.axis('tight')
plt.xlabel('Number of latent factors')
plt.ylabel('RMSE')
plt.show()