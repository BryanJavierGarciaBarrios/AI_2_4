pip install gym

import gym
from gym import spaces
import numpy as np

class MDPEnvironment(gym.Env):
    def __init__(self):
        super(MDPEnvironment, self).__init__()

        # Definir el espacio de observación y acción
        self.observation_space = spaces.Discrete(3)  # Espacio de estados {0, 1, 2}
        self.action_space = spaces.Discrete(2)  # Espacio de acciones {0, 1}

        # Definir parámetros del MDP
        self.transition_probabilities = np.array([
            [0.9, 0.1, 0.0],
            [0.0, 0.9, 0.1],
            [0.0, 0.0, 1.0]
        ])
        self.rewards = np.array([1, 0, 0])
        self.current_state = 0

    def step(self, action):
        # Realizar una transición en el MDP
        next_state = np.random.choice(self.observation_space.n, p=self.transition_probabilities[self.current_state])
        reward = self.rewards[next_state]
        self.current_state = next_state

        # Determinar si el episodio ha terminado
        done = self.current_state == 2

        return next_state, reward, done, {}

    def reset(self):
        # Restablecer el estado del MDP
        self.current_state = 0
        return self.current_state

# Crear una instancia del entorno MDP
env = MDPEnvironment()

# Realizar episodios de prueba
for _ in range(5):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = env.action_space.sample()  # Política aleatoria
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    print(f"Episodio terminado con recompensa total: {total_reward}")
