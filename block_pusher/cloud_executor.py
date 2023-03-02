from .arg_utils import get_parser
import os.path as osp
from .parallel_robotics.parallel_experiment import ParallelExperiment
from .parallel_robotics.agents import *
from .parallel_robotics.experts import *
from .parallel_robotics.allocations import *
from dotmap import DotMap
import rclpy
import yaml

def main(args=None):
    rclpy.init(args=args)
    # Get user arguments and construct config
    parser = get_parser()
    exp_cfg, _ = parser.parse_known_args()

    # Create experiment and run it
    # load agent
    agent = agent_map[exp_cfg.agent]
    dirname = '/frsg_ws/src/parallel_rl/block_pusher' # Ryan: modified for FogrosG
    filepath = osp.join(dirname, 'parallel_robotics/agents/cfg/{}'.format(agent_cfg_map.get(exp_cfg.agent, 'base_agent.yaml')))
    with open(filepath, "r") as fh:
        agent_cfg = yaml.safe_load(fh)
    for key in agent_cfg:
        if type(agent_cfg[key]) == bool:
            parser.add_argument('--{}'.format(key), action='store_true', default=agent_cfg[key])
            parser.add_argument('--no_{}'.format(key), action='store_false', dest='{}'.format(key))
        else:
            parser.add_argument('--{}'.format(key), type=type(agent_cfg[key]), default=agent_cfg[key])
    
    # load expert
    expert = expert_map[exp_cfg.expert]
    filepath = osp.join(dirname, 'parallel_robotics/experts/cfg/{}'.format(expert_cfg_map.get(exp_cfg.expert, 'base_expert.yaml')))
    with open(filepath, "r") as fh:
        expert_cfg = yaml.safe_load(fh)
    for key in expert_cfg:
        if type(expert_cfg[key]) == bool:
            parser.add_argument('--{}'.format(key), action='store_true', default=expert_cfg[key])
            parser.add_argument('--no_{}'.format(key), action='store_false', dest='{}'.format(key))
        else:
            parser.add_argument('--{}'.format(key), type=type(expert_cfg[key]), default=expert_cfg[key])
    
    # load allocation
    allocation = allocation_map[exp_cfg.allocation]
    filepath = osp.join(dirname, 'parallel_robotics/allocations/cfg/{}'.format(allocation_cfg_map.get(exp_cfg.allocation, 'base_allocation.yaml')))
    with open(filepath, "r") as fh:
        allocation_cfg = yaml.safe_load(fh)
    for key in allocation_cfg:
        if type(allocation_cfg[key]) == bool:
            parser.add_argument('--{}'.format(key), action='store_true', default=allocation_cfg[key])
            parser.add_argument('--no_{}'.format(key), action='store_false', dest='{}'.format(key))
        else:
            parser.add_argument('--{}'.format(key), type=type(allocation_cfg[key]), default=allocation_cfg[key])

    exp_cfg = vars(parser.parse_known_args()[0]) # get CLI args
    exp_cfg = DotMap(exp_cfg)
    exp_cfg.agent_cfg = DotMap()
    exp_cfg.expert_cfg = DotMap()
    exp_cfg.allocation_cfg = DotMap()

    # assumes the keys don't overlap among cfg files
    for key in agent_cfg:
        exp_cfg.agent_cfg[key] = exp_cfg[key]
        del exp_cfg[key]
    for key in expert_cfg:
        exp_cfg.expert_cfg[key] = exp_cfg[key]
        del exp_cfg[key]
    for key in allocation_cfg:
        exp_cfg.allocation_cfg[key] = exp_cfg[key]
        del exp_cfg[key]

    print(exp_cfg)
    experiment = ParallelExperiment(exp_cfg, agent, expert, allocation)
    experiment.run()
