"""
A wrapper for envs that define the expert functions themselves
"""
from .base_expert import ParallelExpert 

class AnalyticExpert(ParallelExpert):
    def __init__(self, envs, cfg):
        self.expert_fns = [env.human_action for env in envs]

    def get_action(self, state, env_idx=0):
        return self.expert_fns[env_idx](state)


