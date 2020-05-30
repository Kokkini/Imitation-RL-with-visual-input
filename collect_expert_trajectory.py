import numpy as np
from room import Room
import pickle
import gym


def room_get_human_act():
    mapping = {"w": 0, "s": 1, "a": 2, "d": 3, "e": 4}
    act = mapping[input("pick action: ").lower()]
    return act

def mountain_car_get_human_act():
    mapping = {"a": 0, "s": 1, "d": 2}
    act = mapping[input("pick action: ").lower()]
    return act


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


if __name__ == "__main__":
    # room_size = 10
    # num_tasks = 2
    # work_per_task = 8
    # env = Room(room_size, num_tasks, work_per_task, max_steps=200)

    env = gym.make("MountainCar-v0")
    num_trajectories = 1
    trajectories = get_trajectories(env, num_trajectories, mountain_car_get_human_act)
    print(f"average reward: {np.mean([sum(traj['rew']) for traj in trajectories])}")
    print()

    with open("mcar.pickle", "wb") as f:
        pickle.dump(trajectories, f)



