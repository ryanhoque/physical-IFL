import argparse
'''
Util to compile command line arguments for core script to run experiments
for Parallel Robotics (main.py)
'''


def get_parser():
    # Global Parameters
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@',
                                    description='Fleet DAgger Arguments')
    parser.add_argument('--env_name', default='Physical',
                        help='Choice of environment')
    parser.add_argument('--logdir', default='logs',
                        help='exterior log directory')
    parser.add_argument('--logdir_suffix', default='tmp',
                        help='log directory suffix')
    parser.add_argument('--cuda', action='store_true',
                        help='run on CUDA (default: False)')
    parser.add_argument('--cnn', action='store_true',
                        help='visual observations')
    parser.add_argument('--resume', default='',
                        help='if resuming a previous run, this is the logdir from which to load info')
    parser.add_argument('--seed', type=int, default=100,
                        help='random seed (default: 100)')

    # Parallel experiment
    parser.add_argument('--agent', type=str, default="IL", help="Type of parallel agent; options are in parallel_robotics/agents/__init__.py")
    parser.add_argument('--expert', type=str, default="Analytic", help="Type of parallel expert; options are in parallel_robotics/experts/__init__.py")
    parser.add_argument('--allocation', type=str, default="CRU", help="Type of allocation strategy; options are in parallel_robotics/allocations/__init__.py")
    parser.add_argument('--num_envs', type=int, default=2, help="number of robots")
    parser.add_argument('--num_humans', type=int, default=1, help="number of humans")
    parser.add_argument('--min_int_time', type=int, default=1, help="minimum intervention time")
    parser.add_argument('--hard_reset_time', type=int, default=0, help="number of steps waited before reset completes")
    parser.add_argument('--log_freq', type=int, default=10, help="log frequency")
    parser.add_argument('--vec_env', action="store_true", help="whether or not to use vectorized Isaac Gym environments")
    parser.add_argument('--async_env', action="store_true", help="whether or not to use asynchronous nonblocking environments")
    parser.add_argument('--render', action="store_true", help="whether or not to render an Isaac Gym env")
    parser.add_argument('--num_steps', type=int, default=100, help='maximum number of parallel timesteps (default: 100000)')
    parser.add_argument('--noise', type=float, default=0.0, help='standard deviation for independent zero-mean gaussian noise injection into actions')
    parser.add_argument('--augmentation', type=int, default=0, help='data augmentation factor for visual observations')
    parser.add_argument('--discrete', action="store_true", help="if true, use a discrete act space")
    parser.add_argument('--fogrosg', action='store_true', help="whether or not to use Fog ROS G communication")

    return parser
