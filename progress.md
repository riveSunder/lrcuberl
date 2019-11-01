# Progress update 2019/11/01 commit e926623317d7154e151c8f72abc110914f5d2ea1

Thee environment is built and a DQN training framework established. There is still a bad memory leak that occurs during DQN training, but it has been attenuated. At least part of the leak was due to converting lists to tensors. E.g. instead of 

<code>my_tensor = torch.Tensor([my_float])</code>

we need to  use

<code>my_tensor = torch.Tensor(np.array(my_float))</code>.

That fix didn't solve everything but it did enable DQN training to persist for more than 100 episodes (this depends on the size of the MLP policy network). An ugly workaround repeatedly resumes DQN training through command-line calls for now. The remaining memory leak probably has something to do with autograd and failing to free up memory used to store gradients after irrelevant episodes have been flushed.

Although DQN training quickly learns to solve single-move scrambles and can get pretty good up to a difficulty of ~5 (the difficulty being the number of random moves used to establish a scramble), performance falls off rapidly after that. The Rubik's Cube problem is very similar to the bit-flipping problem described in the [Hindsight Experience Replay paper](https://papers.nips.cc/paper/7090-hindsight-experience-replay.html). The bit-flipping problem was one where an environment's state space consists of a string of bits initialized with each bit randomly taking a value of 0 or 1. Agents can flip any number of bits at each time step, and if they change the state to an unknown target configuration they get a positive reward. For random target configurations this is impossible for DQN to learn with any appreciable number of bits (they say 40 or more) and even if the target is always the same (all 1s or all 0s, for example), the task will be very difficult for long bit-strings. A Rubik's Cube has about 43 quintillion different possible states, which is just slightly more than the roughly 36 quintillion different states describable by a string of 55 bits. 

Hindsight Experience Replay (HER) is simple in concept but gave a big boost to learning a set of 3 robot tasks in the original paper. It allows the agent to truly believe "I meant to do that" when it fails to reach a target state. This is achieved by replacing the actual target state (in the Rubik's Cube task this is matching all colors on each face) with whatever the agent ended up with at the end of each episode during training with experience replay. The method works even better if several fake targets are met for each episode. 
