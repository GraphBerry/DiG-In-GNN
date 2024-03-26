import random
import time
from utils.utils import writeHeader, writeRow
import torch
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
device = T.device("cpu")
from RL_model.RL_utils import plot_learning_curve


class PolicyNetwork(nn.Module):
    """
    Policy neural network: used to determine policy probabilities based on state.
    """
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim):
        super(PolicyNetwork, self).__init__()
        '''
        Three-layer neural network that maps state to action selection probabilities.
        '''
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.prob = nn.Linear(fc2_dim, action_dim)
        self.instancenorm = nn.LayerNorm(state_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=0.001)
        self.to(device)

    def forward(self, state):
        """
        Forward propagation, calculates the probability of taking each policy based on the state.
        :param state: The state.
        :return: The probability of each action.
        """

        x = T.relu(self.fc1(state))
        x = T.relu(self.fc2(x))
        prob = T.softmax(self.prob(x), dim=-1)

        return prob

    def save_checkpoint(self, checkpoint_file):
        """
        Save the reinforcement learning model.
        :param checkpoint_file:
        :return:
        """
        T.save(self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False)

    def load_checkpoint(self, checkpoint_file):
        """
        Load the model.
        :param checkpoint_file:
        :return:
        """
        self.load_state_dict(T.load(checkpoint_file))


class Reinforce:
    """
    Reinforcement learning model.
    """
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, sample_rate, ckpt_dir, tolerance, gamma=0.99):
        self.RL3 = None
        self.gamma = gamma
        self.checkpoint_dir = ckpt_dir
        self.sample_rate = sample_rate # Records the probability of being selected for learning.
        self.reward_memory = [] # Records rewards.
        self.log_prob_memory = [] # Records each policy.
        self.action_memory = [] # Records actions.
        self.rate = 0
        self.rate_f = 0
        self.rate_b = 0
        self.pre_reward = 0.5
        self.num = 1
        self.tolerance = tolerance

        self.policy = PolicyNetwork(alpha=alpha, state_dim=state_dim, action_dim=action_dim,
                                    fc1_dim=fc1_dim, fc2_dim=fc2_dim)

        # self.csv_filename = './record/yelp_RL' + str(self.sample_rate) + '_reward_5.csv'
        # writeHeader(self.csv_filename, header= ['reward', 'pre'], mode= 'w')


    def choose_action(self, observation, list):
        """
        Choose an action based on the observed state.
        :param observation: The current observed state.
        :return: The chosen action [0,1], i.e., whether to aggregate the node or not.
        """
        #state = T.tensor([observation], dtype=T.float).to(device)
        state = observation

        # Get the probabilities of each action.
        probabilities = self.policy.forward(state)
        # Choose an action based on the probabilities.
        dist = Categorical(probabilities)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        self.log_prob_memory.append(log_prob[list])
        self.action_memory.append(action[list])

        return action

    def store_reward(self, reward):
        """
        Record the reward for the agent's learning.
        :param reward: The reward for the current policy selection.
        :return:
        """
        self.reward_memory.append(reward)

    def memory_clear(self):
       self.reward_memory.clear()
       self.log_prob_memory.clear()
       self.action_memory.clear()

    def learn(self, batch_labels):
        """
        Reinforcement learning parameter update, updated once per batch.
        :return:
        """

        loss = 0
        reward = 0
        reward_pos = 0
        log_memory = torch.cat(self.log_prob_memory)
        action = torch.cat(self.action_memory)
        self.policy.optimizer.zero_grad()
        for g, log_prob, label, act in zip(self.reward_memory[0], log_memory, batch_labels, action):
            if act.item() == 1:
                # r = max(g[label] - (self.pre_reward - self.tolerance / self.rate),0)\
                # The following method is more stable than the above one, 
                # because if all values are greater than 0, 
                # they will only optimize towards expanding the probability of this action. 
                # Although the magnitude is different, 
                # there will never be a direct decrease in probability.
                r = g[label] - (self.pre_reward - self.tolerance / self.rate)
            else:
                # r = max((self.pre_reward - self.tolerance / self.rate) - g[label],0)
                r = (self.pre_reward - self.tolerance / self.rate) - g[label]
            reward += r
            reward_pos += g[label]
            # The reward multiplied by the probability of selection is the so-called loss, 
            # but in fact, there is no loss in policy gradient.
            loss += r * (- log_prob)

        loss /= len(self.reward_memory[0])
        reward /= len(self.reward_memory[0])
        reward_pos /= len(self.reward_memory[0])
        self.pre_reward = (self.pre_reward * self.num + reward_pos.item()) / (self.num + 1)
        self.num += 1

        loss.backward(retain_graph=True)
        self.policy.optimizer.step()


        size = len(action)

        # writeRow(self.csv_filename, row= [reward.item(), self.pre_reward], mode= 'a')

        print(F'RL:{self.sample_rate},loss:{loss:.5f},rate:{self.rate:.5f},rate_f:{self.rate_f:.5f},,rate_b:{self.rate_b:.5f}reward:{reward:.5f},reward_pos:{reward_pos:.5f},size:{size} ')


        self.reward_memory.clear()
        self.log_prob_memory.clear()
        self.action_memory.clear()


    def save_models(self, episode):
        self.policy.save_checkpoint(self.checkpoint_dir + 'Reinforce_policy_{}.pth'.format(episode))
        print('Saved the policy network successfully!')

    def load_models(self, episode):
        self.policy.load_checkpoint(self.checkpoint_dir + 'Reinforce_policy_{}.pth'.format(episode))
        print('Loaded the policy network successfully!')