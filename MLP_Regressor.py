from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston

boston = load_boston()

X = boston.data
y = boston.target

reg = MLPRegressor(hidden_layer_sizes=(10,),  activation='logistic', solver='adam', max_iter=1000, verbose=False)

reg = reg.fit(X, y)
p = reg.predict(X)

ax1 = np.arange(0, len(X[:,0])).reshape(-1,1)
ax2 = p

plt.scatter(ax1, y, color = 'blue')
plt.plot(ax1,p, color = 'red')
plt.show()
