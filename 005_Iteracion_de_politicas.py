import numpy as np

# Definición de un MDP simple
# Estado 0: Estado de inicio
# Estado 1: Estado final con recompensa +1
# Estado 2: Estado final con recompensa -1
# Acción 0: Moverse a la izquierda
# Acción 1: Moverse a la derecha
# Probabilidades de transición: 100% determinista
mdp = {
    0: {0: [(1.0, 0, 0)], 1: [(1.0, 0, 0)]},
    1: {0: [(1.0, 1, 1)], 1: [(1.0, 1, 1)]},
    2: {0: [(1.0, 2, -1)], 1: [(1.0, 2, -1)]}
}

# Inicializar valores de la función de valor V(s) arbitrariamente
V = {0: 0, 1: 0, 2: 0}

# Política inicial: Moverse siempre a la derecha
policy = {0: 1, 1: 1, 2: 1}

# Parámetro de descuento gamma
gamma = 0.9

# Tolerancia para la convergencia
tolerance = 1e-6

# Algoritmo de evaluación de política
while True:
    delta = 0
    for s in mdp:
        v = V[s]
        new_v = 0
        for a in mdp[s][policy[s]]:
            for (prob, next_state, reward) in a:
                new_v += prob * (reward + gamma * V[next_state])
        V[s] = new_v
        delta = max(delta, abs(v - new_v))
    if delta < tolerance:
        break

# Resultados
print("Valores de la función de valor (V):")
for s in V:
    print(f"V({s}) = {V[s]:.4f}")
