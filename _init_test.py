import numpy as np
from buffer import ReplayBuffer

buffer_size = 9
loop_size = 18

memory = ReplayBuffer(buffer_size, input_size=8, n_actions=1)

for i in range (loop_size):
    state = np.array([i]*8)
    next_state = np.array([i]*8)
    action = np.array([i])
    reward = i
    mask = 0.0 if i % 2 == 0 else 1.0
    memory.store_transition(state, action, reward, next_state, mask)

print("Testing to ensure the first state is correct")
assert np.all(memory.state_memory[0] == 9)
print("Test Successfully :D, T.T")

print("Testing to ensure the last state is correct")
assert np.all(memory.state_memory[-1] == 17)
print("Test Successfully :D, T.T")

print("Testing to ensure masks are stored as floats")
assert memory.mask_memory.dtype == np.float32
print("Test Successfully :D, T.T")
