import os
import torch
import torch.nn.function as F
from torch.optiom import Adam
from model import *
from buffer import ReplayBuffer

def hard_update(target, source):
    """Copy thẳng trọng số từ source sang target"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def soft_update(target, source, tau):
    """Copy 1 phần trọng số từ sourrce sang target theo công thức:
    θ_target = τ*θ_source + (1 - τ)*θ_target
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau*param.data + (1.0-tau)*target_param.data)

class Agent:
    def __init__(self, num_inputs, action_space, gamma, tau, alpha, target_update_interval, hidden_size, learning_rate, exploration_scaling_factor):
        """
        Args:
            num_inputs: Kích thước của state space
            action_space: Action space của môi trường (ví dụ: gym.spaces.Box)
            gamma: Hệ số chiết khấu (discount factor)
            tau: Hệ số mềm để cập nhật target networks
            alpha: Hệ số entropy (độ ngẫu nhiên của policy)
            policy: Loại policy (ví dụ: 'Gaussian')
            target_update_interval: Số bước để cập nhật target networks
            hidden_size: Số lượng neuron trong các hidden layers
            learning_rate: Tốc độ học cho cả actor và critic
            exploration_scaling_factor: Hệ số để điều chỉnh độ ngẫu nhiên của policy theo thời gian
        """
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = target_update_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Sử dung GPU nếu có, ko thì CPU
        print(f"Running on {self.device}")

        self.critic = Critic(num_inputs, action_space.shape[0], hidden_size).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), learning_rate)

        self.critic_target = Critic(num_inputs=num_inputs, num_actions=action_space.shape[0], hidden_dim=hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)  # Copy weights from critic to critic_target

        self.policy = Actor(num_inputs=num_inputs, num_actions=action_space.shape[0], hidden_dim=hidden_size, action_space=action_space).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), learning_rate)

    def select_action(self, state, evaluate=False):
        """Quyết đinh action dựa trên status hiện tại
        """
        state = torch.FLoatTensor(state).to(self.device).unsqueeze(0)  # Thêm batch dimension
        if evaluate is False: # Nếu eva đanh là flase -> đang training, lấy action có yêu tố khám phá
            action, _, _ = self.policy.sample(state)
        else: 
            _, _, action = self.policy.sample(state) # nếu eva là true -> đang evaluate, lấy action có giá trị cao nhất (không có yếu tố khám phá)
        return action.detach().cpu().numpy()[0]  # Trả về action dưới dạng numpy array, bỏ batch dimension
    
    def update_parameters(self, memory: ReplayBuffer, batch_size: int, updates: int):
        """
        Ý tưởng: Lấy 1 lô dữ liệu từ ReplayBuffer, bao gồm: state, action, rewward, next_state, mask(episode đã kết thúc ch) để cập nhật critic và policy networks.
        Args:
            memory: ReplayBuffer chứa các trải nghiệm
            batch_size: Kích thước của lô dữ liệu để cập nhật
            updates: Số lần cập nhật đã thực hiện (để quyết định khi nào cập nhật target networks)
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample_buffer(batch_size=batch_size)

        #Chuyển dữ liệu sang tensor và đưa lên thiết bị (GPU hoặc CPU)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # Cập nhật critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch)  # Lấy giá trị Q hiện tại từ critic
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf2_loss = qf1_loss + qf2_loss

        #Cập nhật critic
        self.critic_optimizer.zero_grad()
        qf2_loss.backward()
        self.critic_optimizer.step()

        p1, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, p1)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = torch.tensor(0.).to(self.device)  # 
        alpha_tlogs = torch.tensor(self.alpha)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()


    def save_checkpoint(self):
        if not os.path.exists(self.critic.checkpoint_dir): 
            os.makedirs(self.critic.checkpoint_dir)
        print('Saving models...')
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.policy.save_checkpoint()
    
    def load_checkpointl(self, evaluate=False):
        try:
            print('Loading models...')
            self.critic.load_checkpoint()
            self.critic_target.load_checkpoint()
            self.policy.load_checkpoint()
        except:
            if evaluate:
                raise FileNotFoundError("Unable to evaluate a model with out a checkpoint ?1?1?1.")
            else:
                print("No checkpoint found, starting from scratch.")
        
        if evaluate:
            self.critic.eval()
            self.policy.eval()
            self.critic_target.eval()
        else:
            self.critic.train()
            self.policy.train()
            self.critic_target.train()

