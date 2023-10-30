pip install scikit-learn

from sklearn.datasets import load_iris
from sklearn.feature_selection import mutual_info_classif
import numpy as np

# Cargar el conjunto de datos Iris
iris = load_iris()
X = iris.data
y = iris.target

# Calcular la ganancia de información para cada característica
info_gain = mutual_info_classif(X, y)

# Encontrar la característica más informativa
most_informative_feature = np.argmax(info_gain)

print(f"Ganancia de Información para cada característica: {info_gain}")
print(f"La característica más informativa es la {most_informative_feature}-ésima.")
