"""
An implementation of a parallel Imitation Learning agent
"""
import numpy as np
import cv2
import torch
from .base_agent import ParallelAgent
from .impl.bc import BC
from .impl.replay_memory import ReplayMemory
from .impl.utils import augment
from ...env.make_utils import make_env
import pickle
import random

def torchify(x, device): return torch.tensor(x, dtype=torch.float32).to(device)

class SingleTaskParallelILAgent(ParallelAgent):
    def __init__(self, envs, exp_cfg, logdir):
        self.exp_cfg = exp_cfg
        self.cfg = exp_cfg.agent_cfg
        if self.cfg.updates_per_step == -1:
            self.cfg.updates_per_step = self.exp_cfg.num_humans
        self.envs = envs
        self.logdir = logdir
        self.device = torch.device("cuda" if self.exp_cfg.cuda else "cpu")

        # Experiment setup
        self.experiment_setup()

        # Shared memory across all env steps
        self.human_memory = ReplayMemory(self.cfg.replay_size, exp_cfg.seed)
        # Each ensemble member's memory samples with replacement from main memory when constructed
        self.ensemble_memories = [ReplayMemory(self.cfg.replay_size, exp_cfg.seed+i) for i in range(self.cfg.num_policies)]
        self.recovery_memory = ReplayMemory(self.cfg.replay_size, exp_cfg.seed)

        self.total_numsteps = 0
        self.num_constraint_violations = 0
        self.num_goal_reached = 0
        self.num_unsafe_transitions = 0
        self.last_actions = None

    def experiment_setup(self):
        agent = self.agent_setup()
        self.forward_agent = agent 

    def agent_setup(self):
        if self.exp_cfg.vec_env:
            obs_space = self.envs.observation_space
            act_space = self.envs.action_space
        else:
            obs_space = self.envs[0].observation_space
            act_space = self.envs[0].action_space
        agent = BC(obs_space,
            act_space,
            self.exp_cfg,
            self.logdir)
        return agent

    def pretrain_critic_recovery(self, constraint_demo_data):
        # Get data for recovery policy and safety critic training
        self.num_unsafe_transitions = 0
        for transition in constraint_demo_data:
            self.recovery_memory.push(*transition)
            self.num_constraint_violations += int(transition[2])
            self.num_unsafe_transitions += 1
            if self.num_unsafe_transitions == self.cfg.num_unsafe_transitions:
                break
        print("Number of Constraint Transitions: ",
              self.num_unsafe_transitions)
        print("Number of Constraint Violations: ",
              self.num_constraint_violations)
        batch_size = self.cfg.batch_size
        if self.cfg.pos_fraction > 0:
            batch_size = min(self.cfg.batch_size, int(self.num_constraint_violations / self.cfg.pos_fraction))
        for i in range(self.cfg.critic_safe_pretraining_steps):
            if i % 100 == 0:
                print("CRITIC SAFE UPDATE STEP: ", i)
            self.forward_agent.safety_critic.update_parameters(
                memory=self.recovery_memory,
                agent=self.forward_agent,
                batch_size=batch_size)

    def _compute_validation_loss(self):
        # compute validation loss on the initial offline heldout set
        validation = []
        for j in range(len(self.held_out_data)):
            a_pred = self.forward_agent.get_actions(np.expand_dims(self.held_out_data[j][0], axis=0))[0]
            a_sup = self.held_out_data[j][1]
            if self.exp_cfg.discrete:
                if a_pred == a_sup:
                    validation.append(0)
                else:
                    validation.append(1)
            else:
                validation.append(sum(a_pred - a_sup) ** 2)
        print("ValidLoss: ", sum(validation)/len(validation))

    def _only_make_validation_set(self, task_demo_data, train_size=0.9):
        train_size = int(len(task_demo_data) * train_size)
        self.held_out_data = task_demo_data[:len(task_demo_data)-train_size]

    def pretrain_with_task_data(self, task_demo_data, train_size=0.9):
        self.num_task_transitions = 0
        #random.shuffle(task_demo_data)
        train_size = int(len(task_demo_data) * train_size)
        self.held_out_data = task_demo_data[:len(task_demo_data)-train_size]
        for transition in task_demo_data[len(task_demo_data)-train_size:]:
            self.human_memory.push(*transition)
            self.num_goal_reached += int(transition[2])
            self.num_task_transitions += 1
            if self.num_task_transitions == self.cfg.num_task_transitions:
                break
        if not self.exp_cfg.discrete:
            for i in range(self.cfg.num_policies):
                for _ in range(self.human_memory.size):
                    elem = self.human_memory.buffer[np.random.randint(self.human_memory.size)]
                    self.ensemble_memories[i].push(elem[0].copy(), elem[1].copy(), elem[2], elem[3].copy(), elem[4])
        print("Number of Task Transitions: ", self.num_task_transitions)

        # Pretrain BC policy
        print("Pretraining BC!")
        for i in range(self.cfg.policy_pretraining_steps):
            if self.exp_cfg.discrete:
                loss = self.forward_agent.train_discrete(
                    memory=self.human_memory,
                    batch_size=min(self.cfg.batch_size, self.human_memory.size)
                )
            else:
                loss = self.forward_agent.train(
                    memory=self.ensemble_memories,
                    batch_size=min(self.cfg.batch_size, self.human_memory.size)
                )
            if i % 100 == 0:
                print("TrainLoss: ", loss.item())
                self._compute_validation_loss()

    def add_transitions(self, transitions, act_info=None, augmentation=0):
        def add_transition(memory, state, action, reward, next_state, mask):
            memory.push(state.copy(), action, reward, next_state.copy(), mask)
        def add_augmented_transitions(memory, state, action, reward, next_state, mask):
            imgstack = np.hstack(((state*255).astype(np.uint8), (next_state*255).astype(np.uint8)))
            for _ in range(augmentation):
                augmented = augment(imgstack)
                state_mod = (augmented[:,:state.shape[1],:] / 255.).astype(np.float32)
                next_state_mod = (augmented[:,state.shape[1]:,:] / 255.).astype(np.float32)
                add_transition(memory, state_mod, action, reward, next_state_mod, mask)
        if augmentation:
            assert self.exp_cfg.cnn
        for t in transitions:
            if t is not None:
                state, action, reward, next_state, done, info = t
                mask = float(not done)
                if augmentation:
                    add_augmented_transitions(self.recovery_memory, state, action, info['constraint'], next_state, mask)
                else:
                    add_transition(self.recovery_memory, state, action, info['constraint'], next_state, mask)
                if info['human']:
                    if augmentation:
                        add_augmented_transitions(self.human_memory, state, action, reward, next_state, mask)
                        aug_factor = augmentation
                    else:
                        add_transition(self.human_memory, state, action, reward, next_state, mask)
                        aug_factor = 1
                    if not self.exp_cfg.discrete:
                        for i in range(self.cfg.num_policies):
                            for _ in range(aug_factor):
                                elem = self.human_memory.buffer[np.random.randint(self.human_memory.size)]
                                if not self.exp_cfg.discrete:
                                    add_transition(self.ensemble_memories[i], elem[0].copy(), elem[1].copy(), elem[2], elem[3].copy(), elem[4])
                if info['constraint']:
                    self.num_constraint_violations += 1

    def train(self, t): 
        print('human memory len', len(self.human_memory))
        if len(self.human_memory) > self.cfg.batch_size:
            # Number of updates per step in environment
            for i in range(self.cfg.updates_per_step):
                if len(self.human_memory) > self.cfg.batch_size:
                    if self.exp_cfg.discrete:
                        loss = self.forward_agent.train_discrete(
                            memory=self.human_memory,
                            batch_size=self.cfg.batch_size
                        )
                    else:
                        self.forward_agent.train(
                            memory=self.ensemble_memories,
                            batch_size=self.cfg.batch_size
                        )
                if not self.cfg.disable_online_updates and len(
                        self.recovery_memory) > self.cfg.batch_size \
                        and self.num_constraint_violations / self.cfg.batch_size > self.cfg.pos_fraction:
                    self.forward_agent.safety_critic.update_parameters(
                        memory=self.recovery_memory,
                        agent=self.forward_agent,
                        batch_size=self.cfg.batch_size)
            if self.cfg.updates_per_step > 0:
                print('Online Train Loss', loss.item())
                if t % 10 == 0:
                    self._compute_validation_loss() # measure whether we overfit to new data

    def get_actions(self, states, t):
        self.last_actions = self.forward_agent.get_actions(states)
        return self.last_actions, None

    def save(self):
        pickle.dump(self.human_memory, open(self.logdir+'/human_memory.pkl', 'wb'))
        #pickle.dump(self.ensemble_memories, open(self.exp_cfg.logdir+'/ensemble_memories.pkl', 'wb'))
        pickle.dump(self.recovery_memory, open(self.logdir+'/recovery_memory.pkl', 'wb'))
        self.forward_agent.save()

    def load(self, logdir):
        self.human_memory = pickle.load(open(logdir+'/human_memory.pkl', 'rb'))
        #self.ensemble_memories = pickle.load(open(logdir+'/ensemble_memories.pkl', 'rb'))
        self.recovery_memory = pickle.load(open(logdir+'/recovery_memory.pkl', 'rb'))
        self.forward_agent.load(logdir)

    def get_allocation_metrics(self, states, t):
        actions = self.last_actions
        if self.exp_cfg.vec_env:
            constraint_violation = self.envs.constraint_buf.cpu().numpy()
        else:
            constraint_violation = [env.constraint for env in self.envs]
        uncertainty = self.forward_agent.get_policy_uncertainty(states)
        safety = self.forward_agent.safety_critic.get_value(
            torchify(states, self.device), 
            torchify(actions, self.device)).cpu().numpy()
        return {'constraint_violation': constraint_violation, 'uncertainty': uncertainty, 'risk': safety}

