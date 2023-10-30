pip install pomdp-py

from pomdp_py.models.pomdp import POMDP
from pomdp_py.models.pomdp import ObservationModel
from pomdp_py.models.pomdp import TransitionModel
from pomdp_py.models.pomdp import RewardModel

class MyPOMDP(POMDP):
    def __init__(self):
        # Define los estados, acciones y observaciones
        states = [0, 1, 2]  # Estados {0, 1, 2}
        actions = [0, 1]  # Acciones {0, 1}
        observations = [0, 1, 2]  # Observaciones {0, 1, 2}

        super().__init__(states, actions, observations)

        # Define el modelo de transición
        self.transition_model = TransitionModel(self)
        self.transition_model.set_t_probs(
            ((0, 0, 0), (0.1, 0.7, 0.2), (0.4, 0.2, 0.4)),
            ((0, 1, 2), (0.1, 0.4, 0.5), (0.2, 0.7, 0.1))
        )

        # Define el modelo de observación
        self.observation_model = ObservationModel(self)
        self.observation_model.set_o_probs(
            ((0, 1, 2), (0.3, 0.5, 0.2), (0.1, 0.2, 0.7)),
            ((0, 1, 2), (0.7, 0.1, 0.2), (0.4, 0.5, 0.1))
        )

        # Define el modelo de recompensa
        self.reward_model = RewardModel(self)
        self.reward_model.set_rewards(
            ((0, 0, 0), (0.0, 0.0, 0.0), (1.0, -1.0, 0.0)),
            ((0, 1, 2), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        )

# Crear una instancia del POMDP
my_pomdp = MyPOMDP()
