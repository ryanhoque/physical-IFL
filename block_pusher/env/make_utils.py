import gym
from gym.envs.registration import register

ENV_ID = {
    'Navigation': 'Navigation-v0',
    'Physical': 'BlockPushing-v0'
}

ENV_CLASS = {
    'Navigation': 'Navigation',
    'Physical':  'BlockPushing'
}

def register_env(env_name):
    assert env_name in ENV_ID, "unknown environment"
    env_id = ENV_ID[env_name]
    env_class = ENV_CLASS[env_name]
    register(id=env_id, entry_point='block_pusher.env.' + env_name.lower() + ":" + env_class)

def make_env(env_name, idx=0):
    env_id = ENV_ID[env_name]
    return gym.make(env_id, idx=idx)