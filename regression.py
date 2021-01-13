import numpy as np
#import scipy as sc
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

# Cargamos la librería

boston = load_boston()


#Estás como son matrices (vectores) no son 5 regresiones ecuaciones iterativas

x = np.array(boston.data[:, 5]) #aquí se transforma en matriz
y = np.array(boston.target)

plt.scatter(x, y, alpha=0.3)

# Añadimos columna de unos para término independiente.

x = np.array([np.ones(x.shape),x]).T

#Formula para minimizar el error cuadrático medio (MDC):

    #matriz transpuesta de x que es lo mismo que hacer (x.T * X)
    #@ mutiplicación matricial
    #inversa de la matriz x  = np.linalg.inv()

B = np.linalg.inv(np.transpose(x) @ x) @ np.transpose(x) @ y

plt.plot([4, 9], [ B[0] + B[1] * 4, B[0] + B[1] * 9  ], c="red")
plt.show()
