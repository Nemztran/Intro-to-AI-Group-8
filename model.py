import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import os

# Hằng số cho log của độ lệch chuẩn 
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

def weights_init_(m):
    """
    Khởi tạo trọng số cho các layer Linear
    - Xavier Uniform: phân phối trọng số để tránh vanishing/exploding gradients
    - Bias: khởi tạo bằng 0
    """
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain = 1)
        torch.nn.init.constant_(m.bias, 0)

class Critic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, checkpoint_dir='checkpoints', name= 'critic_network'):
        super(Critic, self).__init__()
        # Critic 1
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # Critic 2
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)

        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.apply(weights_init_)

    def forward(self, state, action):
        """
        Forward pass: Tính Q-value cho cặp (state, action)
        Args:
            state: Observation tensor (batch_size, num_inputs)
            action: Action tensor (batch_size, num_actions)
            
        Returns:
            x1: Q-value từ Critic 1
            x2: Q-value từ Critic 2
        """
        # Nối state và action thành 1 vector
        xu = torch.cat([state, action], 1)

        # ===== CRITIC 1 =====
        x1 = F.relu(self.linear1(xu))      # Linear + ReLU activation
        x1 = F.relu(self.linear2(x1))      # Hidden layer + ReLU
        x1 = self.linear3(x1)              # Output layer (Q-value)
        
        # ===== CRITIC 2 =====
        x2 = F.relu(self.linear4(xu))      # Linear + ReLU activation
        x2 = F.relu(self.linear5(x2))      # Hidden layer + ReLU
        x2 = self.linear6(x2)              # Output layer (Q-value)

        return x1, x2
    
    def save_checkpoint(self):
        """Lưu trọng số model vào file checkpoint"""
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        """Tải trọng số model từ file checkpoint"""
        self.load_state_dict(torch.load(self.checkpoint_file))

class Actor(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space = None, checkpoint_dir = 'checkpoints', name = 'actor_network'):
        super(Actor, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        
        self.mean_linear = nn.Linear(hidden_dim, num_actions)  # Output cho mean
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)  # Output cho log std
        
        self.name = name
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_sac')
        self.apply(weights_init_)

        if action_space is None: 
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor((action_space.high - action_space.low) / 2.) 
            self.action_bias = torch.FloatTensor((action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))  # Linear + ReLU activation
        x = F.relu(self.linear2(x))      # Hidden layer + ReLU
        mean = self.mean_linear(x)       # Output layer (mean)
        log_std = self.log_std_linear(x) # Output layer (log std)
        log_std = torch.clamp(log_std, min = LOG_SIG_MIN, max = LOG_SIG_MAX)  # Giới hạn log std
        return mean, log_std
    
    def sample(self, state):
        mean, self.log_std = self.forward(state)  # Lấy mean và log std từ forward pass
        std = torch.exp(self.log_std)  # Tính độ lệch chuẩn
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Lấy mẫu từ phân phối Normal (reparameterization trick)
        y_t = torch.tanh(x_t)  # Áp dụng tanh để giới hạn hành động trong [-1, 1]
        action = y_t * self.action_scale + self.action_bias  # Áp dụng scale và bias
        log_prob = normal.log_prob(x_t)  # Tính log-probability của hành động

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)  # Điều chỉnh log-probability sau khi áp dụng tanh
        log_prob = log_prob.sum(1, keepdim=True)  # Tổng log-probability cho tất cả các action dimensions
        mean = torch .tanh(mean) * self.action_scale + self.action_bias  # Tính mean sau khi áp dụng tanh và scale/bias
        return action, log_prob, mean
    def save_checkpoint(self):
        """Lưu trọng số model vào file checkpoint"""
        torch.save(self.state_dict(), self.checkpoint_file)
    def load_checkpoint(self):
        """Tải trọng số model từ file checkpoint"""
        self.load_state_dict(torch.load(self.checkpoint_file))
    def to(self, device):
        """
        Chuyển model và các tensor liên quan sang CPU hoặc GPU
        """
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super(Actor, self).to(device)
