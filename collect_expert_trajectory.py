import numpy as np
from room import Room
import pickle
import gym
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
import os
from IPython import display
import matplotlib.pyplot as plt

def show_state(env, step, info):
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.title(f"Step: {step} | info: {info}")
    plt.axis("off")
    display.clear_output(wait=True)
    display.display(plt.gcf())

def room_get_human_act():
    mapping = {"w": 0, "s": 1, "a": 2, "d": 3, "e": 4}
    act = mapping[input("pick action: ").lower()]
    return act

def mountain_car_get_human_act():
    mapping = {"a": 0, "s": 1, "d": 2}
    act = mapping[input("pick action: ").lower()]
    return act

def get_human_act(env):
  action_meaning = env.unwrapped.get_action_meanings()
  print(list(enumerate(action_meaning)))
  act = None
  possible_acts = [str(i) for i in range(len(action_meaning))]
  while act not in possible_acts:
    act = input("pick action: ")
  return int(act)


def get_trajectories(env, num_trajectories, get_human_act):
    trajectories = []
    for i in range(num_trajectories):
        traj = {"obs": [], "act": [], "rew": [], "obs_after": []}
        obs = env.reset()
        while True:
            traj["obs"].append(obs)
            env.render()
            act = get_human_act()
            traj["act"].append(act)
            obs, reward, done, info = env.step(act)
            traj["obs_after"].append(obs)
            traj["rew"].append(reward)
            if done:
                break
        trajectories.append(traj)
    env.close()
    return trajectories

def get_trajectories_notebook(env, num_trajectories, get_human_act):
    trajectories = []
    for i in range(num_trajectories):
        traj = {"obs": [], "act": [], "rew": [], "obs_after": []}
        obs = env.reset()
        score = 0
        diffScore = 0
        step = 0
        while True:
            show_state(env, step, str(score))
            print(f"reward: {diffScore}, score: {score}")
            traj["obs"].append(obs)
            # env.render()
            act = get_human_act(env)
            traj["act"].append(act)
            obs, reward, done, info = env.step(act)
            traj["obs_after"].append(obs)
            traj["rew"].append(reward)
            step += 1
            if done:
                break
        trajectories.append(traj)
    env.close()
    return trajectories


if __name__ == "__main__":
    # room_size = 10
    # num_tasks = 2
    # work_per_task = 8
    # env = Room(room_size, num_tasks, work_per_task, max_steps=200)
    log_dir = "/content/drive/My Drive/Colab Notebooks/imitation_RL"
    env_name = "BreakoutNoFrameskip-v4"
    env = gym.make(env_name)
    env = wrap_deepmind(env)
    num_trajectories = 1
    trajectories = get_trajectories_notebook(env, num_trajectories, get_human_act)
    print(f"average reward: {np.mean([sum(traj['rew']) for traj in trajectories])}")
    print()

    trajectory_file = os.path.join(log_dir, f"{env_name}_expert.pkl")
    with open(trajectory_file, "wb") as f:
        pickle.dump(trajectories, f)



