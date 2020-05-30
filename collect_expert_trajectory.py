import numpy as np
from room import Room
import pickle
import gym
from stable_baselines.common.atari_wrappers import make_atari, wrap_deepmind
import os
from IPython import display
import matplotlib.pyplot as plt
import time
from pynput.keyboard import Key, Listener, KeyCode
import dill

class KeyListener:

    key_mapping = {KeyCode.from_char('a'): "LEFT", KeyCode.from_char('d'): "RIGHT", KeyCode.from_char('w'): "UP", KeyCode.from_char("s"): "DOWN", Key.space: "FIRE"}

    def __init__(self, env):
        meaning = env.unwrapped.get_action_meanings()
        self.meaning_to_action = dict(zip(meaning, list(range(len(meaning)))))
        self.time_between_frames = 0.1
        self.current_act = self.meaning_to_action["NOOP"]

    def on_press(self, key):
        if key in self.key_mapping and self.key_mapping[key] in self.meaning_to_action:
            self.current_act = self.meaning_to_action[self.key_mapping[key]]
        elif key == KeyCode.from_char("+"):
            self.time_between_frames /= 2
        elif key == KeyCode.from_char("-"):
            self.time_between_frames *= 2

    def on_release(self, key):
        if key in self.key_mapping and self.key_mapping[key] in self.meaning_to_action:
            self.current_act = self.meaning_to_action["NOOP"]
        if key == Key.esc:
            return False

def get_trajectories_continuous(env, num_trajectories, get_human_act):
    trajectories = []
    key_listener = KeyListener(env)
    Listener(on_press=key_listener.on_press, on_release=key_listener.on_release).start()
    # with Listener(on_press=key_listener.on_press,on_release=key_listener.on_release) as listener:
    #     listener.join()
    for i in range(num_trajectories):
        traj = {"obs": [], "act": [], "rew": [], "obs_after": []}
        obs = env.reset()
        reward = 0
        total_reward = 0
        step = 0
        while True:
            time.sleep(key_listener.time_between_frames)
            print(f"reward: {reward}, total reward: {total_reward}")
            traj["obs"].append(obs)
            env.render()
            act = key_listener.current_act
            print(f"action: {act}")
            traj["act"].append(act)
            obs, reward, done, info = env.step(act)
            traj["obs_after"].append(obs)
            traj["rew"].append(reward)
            step += 1
            total_reward += reward
            if done:
                break
        trajectories.append(traj)
    env.close()
    return trajectories



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
        reward = 0
        total_reward = 0
        step = 0
        while True:
            show_state(env, step, str(score))
            print(f"reward: {reward}, total reward: {total_reward}")
            traj["obs"].append(obs)
            # env.render()
            act = get_human_act(env)
            traj["act"].append(act)
            obs, reward, done, info = env.step(act)
            traj["obs_after"].append(obs)
            traj["rew"].append(reward)
            step += 1
            total_reward += reward
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
    # log_dir = "/content/drive/My Drive/Colab Notebooks/imitation_RL"
    log_dir = "."
    env_name = "BreakoutNoFrameskip-v4"
    env = gym.make(env_name)
    env = wrap_deepmind(env)
    num_trajectories = 1
    trajectories = get_trajectories_continuous(env, num_trajectories, get_human_act)
    print(f"average reward: {np.mean([sum(traj['rew']) for traj in trajectories])}")
    print()

    trajectory_file = os.path.join(log_dir, f"{env_name}_expert.pkl")
    with open(trajectory_file, "wb") as f:
        dill.dump(trajectories, f)



