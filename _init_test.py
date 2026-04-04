from buffer import ReplayBuffer

buffer_size = 9
loop_size = 18

memory = ReplayBuffer(buffer_size, input_size=8, n_actions=1)

for i in range (loop_size):
    memory.store_transition(i, i, i, i)

print("Testing to ensure the first state is correct")
assert memory.state_memory[0] == 9
print("Test Successfully :D, T.T")

print("Testing to ensure the last state is correct")
assert memory.state_memory[-1] == 17
print("Test Successfully :D, T.T")