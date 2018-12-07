from sklearn.neural_network import MLPRegressor
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_absolute_error, mean_squared_error

boston = load_boston()

X = boston.data
y = boston.target

reg = MLPRegressor(hidden_layer_sizes=(10,),  activation='logistic', solver='adam', max_iter=1000, verbose=False)

reg = reg.fit(X, y)
p = reg.predict(X)
print(p)
ax1 = np.arange(0, len(X[:,0])).reshape(-1,1)
ax2 = p

plt.scatter(ax1, y, color = 'blue')
plt.plot(ax1,p, color = 'red')
plt.show()


print(mean_absolute_error(y, p))
print(mean_squared_error(y, p))
print(np.sqrt(mean_absolute_error(y, p)))
