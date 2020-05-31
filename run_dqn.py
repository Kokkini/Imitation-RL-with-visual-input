from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
# from stable_baselines.common.policies import CnnPolicy
from dqn import DQN
import os
from stable_baselines.results_plotter import load_results, ts2xy
from stable_baselines.bench import Monitor
import numpy as np
import gym
import dill

best_mean_reward, n_steps = -np.inf, 0
def callback(_locals, _globals):
  """
  Callback called at each step (for DQN and others) or after n steps (see ACER or PPO2)
  :param _locals: (dict)
  :param _globals: (dict)
  """
  global n_steps, best_mean_reward
  # Print stats every 1000 calls
  if (n_steps + 1) % 1000 == 0:
      # Evaluate policy training performance
      x, y = ts2xy(load_results(log_dir), 'timesteps')
      if len(x) > 0:
          mean_reward = np.mean(y[-100:])
          print(x[-1], 'timesteps')
          print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(best_mean_reward, mean_reward))

          # New best model, you could save the agent here
          if mean_reward > best_mean_reward:
              best_mean_reward = mean_reward
              # Example for saving best model
              print("Saving new best model")
              _locals['self'].save(os.path.join(log_dir,'best_model.pkl'))
  n_steps += 1
  return True

def trajectory_to_exp(trajectories):
    '''
    :param trajectories: [{"obs": [], "act": [], "rew": [], "obs_after": []}]
    :return:  [[obs[i], act[i], rew[i], obs_after[i], done[i]]]
    '''
    result = []
    for traj in trajectories:
        N = len(traj["obs"])
        done = [False for _ in range(N)]
        done[-1] = True
        experiences = list(zip(traj["obs"], traj["act"], traj["rew"], traj["obs_after"], done))
        result += experiences
    return result

def load_expert_trajectory(path):
    with open(path, "rb") as f:
        trajectories = dill.load(f)
    return trajectories

def load_expert_exp(path):
    trajectories = load_expert_trajectory(path)
    return trajectory_to_exp(trajectories)

log_dir = input("log dir: ")
expert_traj_path = input("expert trajectory path")
os.makedirs(log_dir, exist_ok=True)
env = gym.make('BreakoutNoFrameskip-v4')
env = wrap_deepmind(env, frame_stack=True, clip_rewards=False)
env = Monitor(env, log_dir, allow_early_resets=True)
expert_exp = load_expert_exp(expert_traj_path)



model = DQN(CnnPolicy, env, verbose=1, prioritized_replay=True, buffer_size= 200000,
            exploration_final_eps=0.1, train_freq=4, batch_size=128, expert_exp=expert_exp)
# model = ACKTR(CnnPolicy, env)
model.learn(total_timesteps=int(1e6), callback=callback)


# model.save("deepq_breakout")
# model = DQN.load("deepq_breakout")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()