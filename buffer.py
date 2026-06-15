import numpy as np

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




