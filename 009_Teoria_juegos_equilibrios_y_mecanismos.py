pip install nashpy

import nashpy as nash
import numpy as np

# Define las matrices de pagos para dos jugadores en un juego de coordinación
# Cada matriz representa los pagos para el jugador 1 y el jugador 2 en función de sus acciones
matrix_player_1 = np.array([[3, 0], [2, 2]])
matrix_player_2 = np.array([[3, 2], [0, 2]])

# Crea el juego en forma normal
game = nash.Game(matrix_player_1, matrix_player_2)

# Encuentra y muestra los equilibrios de Nash
equilibria = game.support_enumeration()
for eq in equilibria:
    print("Equilibrio de Nash:", eq)
