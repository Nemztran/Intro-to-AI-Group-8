import numpy as np
import os

class ReplayBuffer():
    """
    Lớp dùng để lưu trữ kinh nghiệm để mô hình có thể dùng lại để học lại nhiều lần
    """

    def __init__(self, max_size, input_size, n_actions):
        """
        :param max_size: the maximum number of experiences the buffer can hold
        "param n_actions: the number of possible actions 
        """
        self.mem_size = max_size
        self.mem_ctr = 0
        self.state_memory = np.zeros((self.mem_size, input_size))
        self.new_state_memory = np.zeros((self.mem_size, input_size))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.mask_memory = np.zeros(self.mem_size, dtype=np.float32)
    
    def can_sample(self, batch_size):
        """
        Kiểm tra bộ nhớ có đủ dữ liệu để học hay chưa
        Vd: Muốn lấy 36 mẫu để học thì cần phải có ít nhất 180 mẫu trong bộ nhớ để đảm bảo đa dạng và hiệu quả trong quá trình học
        """
        if self.mem_ctr >  (batch_size * 5):
            return True
        else:
            return False
    
    def store_transition(self, state, action, reward, next_state, mask):
        """
        Nhận dữ liệu từ môi trường và lưu trữ nó vào bộ nhớ
        """
        index = self.mem_ctr % self.mem_size #Khi bộ nhớ đầy thì ghi đè lên dữ liệu cũ

        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action  
        self.reward_memory[index] = reward
        self.mask_memory[index] = mask

        self.mem_ctr += 1

    def sample_buffer(self, batch_size):
        """
        Lấy ngẫu nhiên 1 lô dữ liệu để bỏ vào train
        """
        max_mem = min(self.mem_ctr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        next_states = self.new_state_memory[batch]
        action = self.action_memory[batch]
        reward = self.reward_memory[batch]
        masks = self.mask_memory[batch]

        return states, action, reward, next_states, masks

    def save(self, path):
        n = min(self.mem_ctr, self.mem_size)
        np.savez_compressed(path,
            states=self.state_memory[:n],
            next_states=self.new_state_memory[:n],
            actions=self.action_memory[:n],
            rewards=self.reward_memory[:n],
            masks=self.mask_memory[:n],
            mem_ctr=np.array([self.mem_ctr]))
        print(f"Buffer saved ({n} transitions)")

    def load(self, path):
        if not os.path.exists(path):
            print(f"No buffer at {path}, starting empty")
            return
        data = np.load(path)
        n = len(data['states'])
        self.state_memory[:n] = data['states']
        self.new_state_memory[:n] = data['next_states']
        self.action_memory[:n] = data['actions']
        self.reward_memory[:n] = data['rewards']
        self.mask_memory[:n] = data['masks']
        self.mem_ctr = int(data['mem_ctr'][0])
        print(f"Buffer loaded ({n} transitions)")

