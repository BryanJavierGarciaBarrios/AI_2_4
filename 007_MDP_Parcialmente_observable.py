pip install gym gym_pomdp

import numpy as np
import gym
from gym_pomdp.envs import Tiger

# Definir un agente POMDP simple
class SimplePOMDPAgent:
    def __init__(self, action_space, observation_space):
        self.action_space = action_space
        self.observation_space = observation_space
        self.belief = np.ones(observation_space.n) / observation_space.n
        self.policy = np.zeros(action_space.n)

    def act(self):
        return np.random.choice(self.action_space.n, p=self.policy)

    def update_belief(self, action, observation):
        # Actualizar la creencia (belief) basada en la acción y la observación
        self.belief = self.belief * observation
        self.belief = self.belief / self.belief.sum()

    def update_policy(self):
        # Actualizar la política en función de la creencia actual
        self.policy = self.belief

# Crear un entorno POMDP simple
env = gym.make('Tiger-v0')

# Crear un agente POMDP
agent = SimplePOMDPAgent(env.action_space, env.observation_space)

# Ejecutar la interacción agente-entorno en un número de pasos
num_steps = 10
for _ in range(num_steps):
    action = agent.act()
    observation, reward, done, _ = env.step(action)
    agent.update_belief(action, observation)
    agent.update_policy()
    print(f"Action: {action}, Observation: {observation}, Belief: {agent.belief}, Policy: {agent.policy}")

env.close()
