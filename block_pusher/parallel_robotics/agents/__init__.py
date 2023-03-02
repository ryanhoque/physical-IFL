from .il_agent import SingleTaskParallelILAgent

# Mappings from CLI option strings to agents
agent_map = {
    "IL": SingleTaskParallelILAgent
}

agent_cfg_map = {
    "IL": "il_agent.yaml"
}