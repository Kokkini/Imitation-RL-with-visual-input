import gym
from gym import spaces
import numpy as np
import sys

class Room(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, room_size, num_tasks, work_per_task, max_steps=100):
    super(Room, self).__init__()
    # Define action and observation space
    # They must be gym.spaces objects
    # Example when using discrete actions:
    n_actions = 5
    self.action_space = spaces.Discrete(5)
    # observation: (room_size, agent_x, agent_y, progress_bar, (task_x, task_y, task_done)*num_tasks
    self.observation_dim = 4 + 3*num_tasks
    obs_high = max([room_size-1, work_per_task])
    # Example for using image as input:
    self.observation_space = spaces.Box(low=0, high=obs_high, shape=(self.observation_dim,), dtype=np.int32)
    self.state = None
    self.num_tasks = num_tasks
    self.room_size = room_size
    self.current_task = 0
    self.work_per_task = work_per_task
    self.t = 0
    self.t_limit = max_steps
    self.reset()

  def step(self, action):
    # action: up, down, left, right, stay
    if action == 0:
        self.state[2] = min([self.state[2]+1, self.room_size-1])
    elif action == 1:
        self.state[2] = max([self.state[2]-1, 0])
    elif action == 2:
        self.state[1] = max([self.state[1]-1, 0])
    elif action == 3:
        self.state[1] = min([self.state[1]+1, self.room_size-1])
    elif action !=4:
        print(f"invalid action {action}")
        exit(1)
    # increase progress if the agent is at the current task
    if self.current_task < self.num_tasks:
        c_task_x = self.state[4+(3*self.current_task)]
        c_task_y = self.state[5+(3*self.current_task)]
        if self.state[1] == c_task_x and self.state[2] == c_task_y:
            self.state[3] += 1
        else:
            self.state[3] = 0

    reward = 0
    if self.state[3] == self.work_per_task:
        self.state[6+self.current_task*3] = 1
        reward = self.current_task + 1
        self.current_task += 1
        self.state[3] = 0

    self.t += 1
    done = False
    if self.current_task >= self.num_tasks:
        done = True
    if self.t >= self.t_limit:
        done = True

    info = None

    return self.state.copy(), reward, done, info

  def reset(self):
    state = np.zeros(self.observation_dim, dtype=np.int32)
    state[0] = self.room_size
    # randomize the agent's and tasks' positions
    taken_pos = []
    while True:
        pos = tuple(np.random.randint(self.room_size, size=(2)).tolist())
        if pos in taken_pos:
            continue
        taken_pos.append(pos)
        if len(taken_pos) == self.num_tasks + 1:
            break
    state[1], state[2] = taken_pos[-1][0], taken_pos[-1][1]
    for i in range(self.num_tasks):
        state[4+3*i] = taken_pos[i][0]
        state[5+3*i] = taken_pos[i][1]
        state[6+3*i] = 0

    # reset progress
    state[3] = 0

    # reset current task
    self.current_task = 0

    self.state = state

    self.t = 0
    return state.copy()  # reward, done, info can't be included

  def render(self, mode='human'):
    map2d = [["." for i in range(self.room_size)] for i in range(self.room_size)]
    for i in range(self.num_tasks):
        task_x = self.state[4+i*3]
        task_y = self.state[5+i*3]
        task_done = self.state[6+i*3]
        if task_done == 0:
            map2d[self.room_size-1-task_y][task_x] = str(i+1)

    agent_x = self.state[1]
    agent_y = self.state[2]
    map2d[self.room_size-1-agent_y][agent_x] = "A"
    progress = ['_'] * self.work_per_task
    progress[:self.state[3]] = ["#"] * self.state[3]
    print("\n".join(["".join(map2d[i]) for i in range(self.room_size)]))
    print("".join(progress))
    sys.stdout.flush()


  def close (self):
    return