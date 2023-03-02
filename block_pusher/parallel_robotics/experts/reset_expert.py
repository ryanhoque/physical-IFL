"""
A simple expert that initiates a reset of the environment (handled by parallel_experiment.py)
"""
from .base_expert import ParallelExpert 

class ResetExpert(ParallelExpert):
    def __init__(self, envs, cfg):
        pass

    def get_action(self, state, env_idx=0):
        # actual reset calls are handled by parallel_experiment
        return "reset"


