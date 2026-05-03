import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import namedtuple, deque
import numpy as np
import torch.nn.functional as F
from torch.distributions import Normal

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)


class DQNAgent:
    def __init__(self, state_dim, action_dim, device, gamma=0.9, lr=5e-5, buffer_size=10000, batch_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.device = device
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayBuffer(buffer_size)
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

    def select_action(self, state, use_greedy=False):
        is_batch = isinstance(state, torch.Tensor)

        if not use_greedy and random.random() < self.epsilon:
            if is_batch:
                batch_size = state.shape[0]
                return torch.tensor([random.randrange(self.action_dim) for _ in range(batch_size)], device=self.device,
                                    dtype=torch.long)
            else:
                return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                if not is_batch:
                    state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                else:
                    state = state.to(self.device)

                q_values = self.policy_net(state)

                if is_batch:
                    return q_values.max(1)[1]
                else:
                    return q_values.max(1)[1].view(1, 1)

    def store_experience(self, state, action, reward, next_state, done):
        state_tensor = torch.from_numpy(state).float()
        action_tensor = torch.tensor([[action]], dtype=torch.long)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        next_state_tensor = torch.from_numpy(next_state).float()
        done_tensor = torch.tensor([done], dtype=torch.bool)
        self.memory.push(state_tensor, action_tensor, reward_tensor, next_state_tensor, done_tensor)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        state_batch = torch.cat([s.unsqueeze(0) for s in batch.state]).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_state_batch = torch.cat([s.unsqueeze(0) for s in batch.next_state]).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        next_q_values[done_batch] = 0.0
        expected_q_values = reward_batch + (self.gamma * next_q_values)
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q_values, expected_q_values.unsqueeze(1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


class ActorCriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetwork, self).__init__()
        self.shared_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )
        self.actor_head = nn.Linear(256, action_dim)
        self.critic_head = nn.Linear(256, 1)

    def forward(self, state):
        x = self.shared_layer(state)
        action_mean = torch.tanh(self.actor_head(x))
        state_value = self.critic_head(x)
        return action_mean, state_value


class PPOAgent:
    def __init__(self, state_dim, action_dim, device, gamma=0.9, lr=5e-5, gae_lambda=0.95, policy_clip=0.2):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.device = device
        self.policy = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.memory = []

    def store_experience(self, state, action, reward, log_prob, value, done):
        self.memory.append((
            torch.from_numpy(state).float(),
            torch.from_numpy(action),
            torch.tensor([reward], dtype=torch.float32),
            torch.tensor([log_prob], dtype=torch.float32),
            torch.tensor([value], dtype=torch.float32),
            torch.tensor([done], dtype=torch.bool)
        ))

    def clear_memory(self):
        self.memory = []

    def select_action(self, state):
        with torch.no_grad():
            is_batch = state.ndim > 1
            state_tensor = torch.from_numpy(state).float().to(self.device)
            if not is_batch:
                state_tensor = state_tensor.unsqueeze(0)

            action_mean, state_value = self.policy(state_tensor)
            action_std = torch.tensor(0.5, device=self.device)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            action_log_prob = dist.log_prob(action).sum(dim=-1)

            if not is_batch:
                return action.cpu().numpy().flatten(), action_log_prob.item(), state_value.item()
            else:
                return action.cpu().numpy(), action_log_prob.cpu().numpy(), state_value.cpu().numpy().flatten()

    def learn(self, epochs=10, entropy_coeff=0.01):
        states = torch.stack([t[0] for t in self.memory]).squeeze(1).to(self.device)
        actions = torch.stack([t[1] for t in self.memory]).to(self.device)
        rewards = torch.stack([t[2] for t in self.memory]).view(-1).to(self.device)
        old_log_probs = torch.stack([t[3] for t in self.memory]).view(-1).to(self.device)
        values = torch.stack([t[4] for t in self.memory]).view(-1).to(self.device)
        dones = torch.stack([t[5] for t in self.memory]).view(-1).to(self.device)
        advantages = torch.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t].float()
                next_values = 0
            else:
                next_non_terminal = 1.0 - dones[t].float()
                next_values = values[t + 1]
            delta = rewards[t] + self.gamma * next_values * next_non_terminal - values[t]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            advantages[t] = last_gae_lam
        returns = advantages + values
        for _ in range(epochs):
            new_means, new_values = self.policy(states)
            new_values = new_values.view(-1)
            action_std = torch.tensor(0.5, device=self.device)
            dist = Normal(new_means, action_std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(new_values, returns)
            loss = policy_loss + 0.5 * value_loss - entropy_coeff * entropy
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()
        self.clear_memory()
