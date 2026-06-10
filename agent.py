import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import *
from buffer import ReplayBuffer
from evaluation import evaluate_policy, success_from_info
from torch.utils.tensorboard import SummaryWriter


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
    def __init__(
        self,
        num_inputs,
        action_space,
        gamma,
        tau,
        alpha,
        target_update_interval,
        hidden_size,
        learning_rate,
        exploration_scaling_factor,
        intrinsic_reward_clip=1.0,
        use_curiosity=True,
        checkpoint_dir='checkpoints',
    ):
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
            intrinsic_reward_clip: Giới hạn reward nội tại sau khi normalize theo batch
            use_curiosity: Bật/tắt intrinsic curiosity reward
            checkpoint_dir: Thư mục lưu checkpoint
        """
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.target_update_interval = target_update_interval

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Sử dung GPU nếu có, ko thì CPU
        print(f"Running on {self.device}")

        self.critic = Critic(num_inputs, action_space.shape[0], hidden_size, checkpoint_dir=checkpoint_dir).to(self.device)
        self.critic_optimizer = Adam(self.critic.parameters(), learning_rate)

        self.critic_target = Critic(num_inputs=num_inputs, num_actions=action_space.shape[0], hidden_dim=hidden_size, checkpoint_dir=checkpoint_dir, name='critic_target_network').to(self.device)
        hard_update(self.critic_target, self.critic)  # Copy weights from critic to critic_target

        self.policy = Actor(num_inputs=num_inputs, num_actions=action_space.shape[0], hidden_dim=hidden_size, action_space=action_space, checkpoint_dir=checkpoint_dir).to(self.device)
        self.policy_optimizer = Adam(self.policy.parameters(), learning_rate)

        # Initialize the predictive model
        self.predictive_model = PredictiveModel(num_inputs, num_actions=action_space.shape[0], hidden_dim=hidden_size, checkpoint_dir=checkpoint_dir).to(self.device)
        self.predictive_model_optim = Adam(self.predictive_model.parameters(), learning_rate)
        self.exploration_scaling_factor = exploration_scaling_factor
        self.intrinsic_reward_clip = intrinsic_reward_clip
        self.use_curiosity = use_curiosity and exploration_scaling_factor > 0
    def select_action(self, state, evaluate=False):
        """Quyết đinh action dựa trên status hiện tại
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)  # Thêm batch dimension
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
        prediction_loss = torch.tensor(0.0, device=self.device)
        if self.use_curiosity:
            predicted_next_state = self.predictive_model(state_batch, action_batch)
            prediction_loss = F.mse_loss(predicted_next_state, next_state_batch)
            predictive_error_no_reduction = F.mse_loss(predicted_next_state, next_state_batch, reduction='none')

            intrinsic_reward = predictive_error_no_reduction.detach().mean(dim=1, keepdim=True)
            intrinsic_reward = intrinsic_reward / intrinsic_reward.mean().clamp_min(1e-8)
            if self.intrinsic_reward_clip is not None:
                intrinsic_reward = torch.clamp(intrinsic_reward, max=self.intrinsic_reward_clip)
            intrinsic_reward = self.exploration_scaling_factor * intrinsic_reward
            reward_batch += intrinsic_reward

            self.predictive_model_optim.zero_grad()
            prediction_loss.backward()
            self.predictive_model_optim.step()

        # Cập nhật critic
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch)  # Lấy giá trị Q hiện tại từ critic
        qf1_loss = F.mse_loss(qf1, next_q_value) 
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss

        #Cập nhật critic
        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.policy.sample(state_batch)
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_loss = torch.tensor(0.).to(self.device)  # 
        alpha_tlogs = torch.tensor(self.alpha, device=self.device)

        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), prediction_loss.item(), alpha_tlogs.item()


    def train(self, env, env_name, memory : ReplayBuffer, episodes = 1000, batch_size=64, updates_per_step=1, summary_writer_name="", max_episode_steps=100, warmup_episodes=20, seed=None):
        from datetime import datetime
        if seed is not None:
            env.action_space.seed(seed)
        #Tensorboard
        summary_writer_name= f"runs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_" + summary_writer_name
        writer= SummaryWriter(summary_writer_name)

        #Training Loop 
        total_numsteps= 0
        update= 0

        for i_episode in range(episodes):
            episode_reward= 0
            episode_steps= 0
            done= False
            episode_seed = seed + i_episode if seed is not None else None
            state, info = env.reset(seed=episode_seed)
            episode_success = success_from_info(info)

            while not done and episode_steps < max_episode_steps:
                if warmup_episodes > i_episode:
                    action = env.action_space.sample()
                else:
                    action = self.select_action(state)
                if memory.can_sample(batch_size=batch_size):
                    for _ in range(updates_per_step):
                        critic_1_loss, critic_2_loss, policy_loss, ent_loss, prediction_loss, alpha_val = self.update_parameters(memory, batch_size, update)
                        #Tensorboard
                        writer.add_scalar('loss/critic_1', critic_1_loss, update)
                        writer.add_scalar('loss/critic_2', critic_2_loss, update)
                        writer.add_scalar('loss/policy', policy_loss, update)
                        writer.add_scalar('loss/entropy', ent_loss, update)
                        writer.add_scalar('loss/prediction_loss', prediction_loss, update)
                        writer.add_scalar('parameters/alpha', alpha_val, update)
                        update += 1
                next_state, reward, terminated, truncated, info= env.step(action)
                step_success = success_from_info(info)
                episode_success = episode_success or step_success
                done = terminated or truncated or step_success

                episode_steps += 1
                total_numsteps += 1
                episode_reward += reward
            
                mask = 0.0 if terminated or step_success else 1.0

                memory.store_transition(state, action, reward, next_state, mask)

                state = next_state
            writer.add_scalar('reward/train', episode_reward, i_episode)
            writer.add_scalar('success/train', float(episode_success), i_episode)
            writer.add_scalar('steps/train', episode_steps, i_episode)
            print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, success: {episode_success}, reward: {round(episode_reward, 2)}")
            if i_episode % 10 == 0:
                self.save_checkpoint()
        writer.close()

    def test(self, env, episodes=10, max_episode_steps=100, name="SAC Agent", print_episodes=True, seed=None):
        """
        Evaluate the trained agent on the environment
        Args:
            env: Gymnasium environment
            episodes: Number of episodes to evaluate
            max_episode_steps: Maximum steps per episode
        """
        return evaluate_policy(
            env=env,
            select_action=lambda state: self.select_action(state, evaluate=True),
            episodes=episodes,
            max_episode_steps=max_episode_steps,
            name=name,
            print_episodes=print_episodes,
            seed=seed,
        )
                
    def save_checkpoint(self):
        if not os.path.exists(self.critic.checkpoint_dir): 
            os.makedirs(self.critic.checkpoint_dir)
        print('Saving models...')
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()
        self.policy.save_checkpoint()
        if self.use_curiosity:
            self.predictive_model.save_checkpoint()
    
    def load_checkpoint(self, evaluate=False):
        try:
            print('Loading models...')
            self.critic.load_checkpoint(map_location=self.device)
            self.policy.load_checkpoint(map_location=self.device)
        except FileNotFoundError as exc:
            if evaluate:
                raise FileNotFoundError("Unable to evaluate a model without actor and critic checkpoints.") from exc
            else:
                print("No checkpoint found, starting from scratch.")
        else:
            try:
                self.critic_target.load_checkpoint(map_location=self.device)
            except FileNotFoundError:
                print("No target critic checkpoint found, copying critic weights.")
                hard_update(self.critic_target, self.critic)

            if self.use_curiosity:
                try:
                    self.predictive_model.load_checkpoint(map_location=self.device)
                except FileNotFoundError:
                    print("No predictive model checkpoint found, leaving initialized weights.")
        
        if evaluate:
            self.critic.eval()
            self.policy.eval()
            self.critic_target.eval()
            self.predictive_model.eval()
        else:
            self.critic.train()
            self.policy.train()
            self.critic_target.train()
            self.predictive_model.train()

