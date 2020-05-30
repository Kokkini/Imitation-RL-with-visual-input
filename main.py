import torch
from torch.distributions import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt
import room
import pickle

# room_size = 10
# num_tasks = 2
# work_per_task = 8
# env = room.Room(room_size, num_tasks, work_per_task, max_steps=200)

env = gym.make("MountainCar-v0")
env._max_episode_steps = 200

obs_size = env.observation_space.shape[0]
num_acts = env.action_space.n

def get_logit_net(sizes, activation):
    layers = []
    for i in range(len(sizes) - 2):
        layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
        layers.append(activation())
    layers.append(torch.nn.Linear(sizes[-2], sizes[-1], bias=False))
    return torch.nn.Sequential(*layers)


# nets
policy_net_sizes = [obs_size, 64, 64, num_acts]
value_net_sizes = [obs_size, 64, 64, 1]
forward_net_sizes = [obs_size+num_acts, 64, 64, obs_size]

activation = torch.nn.Tanh
logit_net = get_logit_net(policy_net_sizes, activation)
value_net = get_logit_net(value_net_sizes, activation)
forward_net = get_logit_net(forward_net_sizes, activation)
print(logit_net)
print(value_net)
print(forward_net)


vis_dict = {"reward": [],
            "loss_pi": [],
            "loss_v": []}

def ppo_loss(obs, act, old_log_prob, advatage, epsilon):
    new_log_prob = get_policy(obs).log_prob(act)
    policy_ratio = torch.exp(new_log_prob - old_log_prob)
    clamped_policy_ratio = torch.clamp(policy_ratio, 1-epsilon, 1+epsilon)
    loss = - torch.min(policy_ratio*advatage, clamped_policy_ratio*advatage).mean()

    approx_kl = (old_log_prob - new_log_prob).mean().item()
    info_dict = dict(kl=approx_kl)
    return loss, info_dict

def plot_dict(d, sharex=True):
    fig, axs = plt.subplots(len(d), 1, sharex=sharex, figsize=(10, len(d)*5))
    keys = list(d.keys())
    for i in range(len(keys)):
        ax = axs[i]
        key = keys[i]
        ax.plot(d[key])
        ax.set_title(key)
    plt.show()

def load_expert_traj(file_path):
    with open(file_path, 'rb') as f:
        content = pickle.load(f)
    return content

def get_policy(obs):
    logits = logit_net(obs)
    return Categorical(logits=logits)

def get_action(obs):
    return get_policy(obs).sample().item()

def compute_loss(obs, act, weights):
    logp = get_policy(obs).log_prob(act)
    return -(logp * weights).mean()

def reward_to_go(rews, gamma):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + gamma*(rtgs[i+1] if i+1 < n else 0)
    return list(rtgs)

def n_step_reward(rews, vals, gamma, n):
    result = []
    rew_to_go = reward_to_go(rews, gamma)
    T = len(rews) - 1
    for t in range(T+1):
        if t+n > T:
            r = rew_to_go[t]
        else:
            r = rew_to_go[t] - (gamma**n)*rew_to_go[t+n] + (gamma**n)*vals[t+n]
        result.append(r)
    return result

def n_step_reward_at(t, n, rew_to_go, vals, gamma):
    T = len(rew_to_go) - 1
    if t + n > T:
        r = rew_to_go[t]
    else:
        r = rew_to_go[t] - (gamma ** n) * rew_to_go[t + n] + (gamma ** n) * vals[t + n]
    return r

def GAE_reward(rews, vals, gamma, lam):
    vals_arr = np.array(vals)
    delta_t = rews + gamma * vals_arr[1:] - vals_arr[:-1]
    return reward_to_go(delta_t, gamma*lam)

def get_advantage(reward_to_go, values):
    return reward_to_go - values

def to_onehot(index_arr, num_classes):
    return np.eye(num_classes)[index_arr]

def train_one_epoch(policy_optimizer, value_optimizer, forward_optimizer, batch_size, expert_prob=0.2, expert_trajectories=None, reward_scaling=1, render = True):
    obs = env.reset()
    batch_obs = []
    batch_obs_after = []
    batch_act = []
    batch_weight = []
    batch_reward_to_go = []
    ep_val = []
    ep_reward = []
    ep_true_reward = []
    batch_log_prob = []
    render_this_epoch = False
    gamma = 0.99
    lam = 0.97
    batch_ep_reward = []
    pi_updates_per_epoch = 20
    v_updates_per_epoch = 20
    forward_updates_per_epoch = 1
    curiosity_weight = 0

    while True:
        if not render_this_epoch and render:
            env.render()
        batch_obs.append(obs.copy())
        act_dist = get_policy(torch.as_tensor(obs, dtype=torch.float32))
        val = value_net(torch.as_tensor(obs, dtype=torch.float32)).item()
        ep_val.append(val)
        act = act_dist.sample().item()
        batch_log_prob.append(act_dist.log_prob(torch.as_tensor(act)).item())
        batch_act.append(act)
        act_onehot = np.array(to_onehot([act], num_acts))[0]
        obs_act = np.concatenate([obs, act_onehot])
        pred_obs = forward_net(torch.as_tensor(obs_act, dtype=torch.float32))

        obs, reward, done, info = env.step(act)
        curiosity_reward = torch.mean((pred_obs - torch.as_tensor(obs, dtype=torch.float32))**2).item()
        ep_reward.append(reward + curiosity_weight * curiosity_reward)
        ep_true_reward.append(reward)
        batch_obs_after.append(obs.copy())



        if done:
            # print(f"ep length: {len(ep_reward)}")
            batch_ep_reward.append(sum(ep_true_reward))
            render_this_epoch = True
            # weights = list(reward_to_go(ep_reward, gamma))
            # weights = n_step_reward(ep_reward, ep_val, gamma, 30)
            # last_val = value_net(torch.as_tensor(obs, dtype=torch.float32)).item()
            last_val = 0

            ep_reward = [0] * len(ep_reward)
            if len(ep_reward) < env._max_episode_steps:
                ep_reward[-1] = 1

            ep_reward = np.array(ep_reward) * reward_scaling
            weights = GAE_reward(ep_reward, ep_val + [last_val], gamma, lam)
            batch_weight += weights
            # ep_reward += [last_val] #boostrap
            batch_reward_to_go += reward_to_go(ep_reward, gamma)
            obs = env.reset()
            ep_reward = []
            ep_true_reward = []
            ep_val = []
            done = False
            if len(batch_obs) >= batch_size:
                break


    if np.random.uniform() < expert_prob:
        print("using expert trajectories")
        for traj in expert_trajectories:
            vals = value_net(torch.as_tensor(traj["obs"]+[traj["obs_after"][-1]], dtype=torch.float32)).data.numpy()[:,0]
            vals[-1] = 0

            expert_reward = [0] * len(traj["rew"])
            if len(expert_reward) < env._max_episode_steps:
                expert_reward[-1] = 1

            # expert_reward_scaled = np.array(traj["rew"]) * reward_scaling
            expert_reward_scaled = np.array(expert_reward) * reward_scaling

            expert_weights = GAE_reward(expert_reward_scaled, vals, gamma, lam)
            print(f"average expert advantage: {np.mean(expert_weights)}")
            print(f"average advantage       : {np.mean(batch_weight)}")
            batch_weight += expert_weights
            batch_reward_to_go += reward_to_go(expert_reward_scaled, gamma)
            expert_act_dist = get_policy(torch.as_tensor(traj["obs"], dtype=torch.float32))
            expert_log_prob = expert_act_dist.log_prob(torch.as_tensor(traj["act"], dtype=torch.long)).data.numpy().tolist()
            print(f"average expert log prob: {np.mean(expert_log_prob)}")
            print(f"average log prob: {np.mean(batch_log_prob)}")
            batch_log_prob += expert_log_prob

            batch_obs += traj["obs"]
            batch_obs_after += traj["obs_after"]
            batch_act += traj["act"]

    for i in range(pi_updates_per_epoch):
        policy_optimizer.zero_grad()
        # batch_loss = compute_loss(torch.as_tensor(batch_obs, dtype=torch.float32),
        #                           torch.as_tensor(batch_act, dtype=torch.long),
        #                           torch.as_tensor(batch_weight, dtype=torch.float32))
        batch_loss, pi_info = ppo_loss(torch.as_tensor(batch_obs, dtype=torch.float32),
                                         torch.as_tensor(batch_act, dtype=torch.long),
                                         torch.as_tensor(batch_log_prob, dtype=torch.float32),
                                         torch.as_tensor(batch_weight, dtype=torch.float32),
                                         epsilon=0.2)
        kl = pi_info['kl']
        target_kl = 0.01
        if kl > 1.5 * target_kl:
            print('Early stopping at step %d due to reaching max kl.' % i)
            break

        batch_loss.backward()
        policy_optimizer.step()
    print(f"loss_pi: {batch_loss.item()}")


    # advatages = get_advantage(torch.as_tensor(batch_reward_to_go, dtype=torch.float32), value_net(torch.as_tensor(batch_obs, dtype=torch.float32)))
    # advatages = advatages.data.numpy()
    batch_act_onehot = np.array(to_onehot(batch_act, num_acts))
    batch_obs_act = np.concatenate([batch_obs, batch_act_onehot], axis=1)
    for k in range(forward_updates_per_epoch):
        batch_pred_obs = forward_net(torch.as_tensor(batch_obs_act, dtype=torch.float32))
        forward_loss = torch.mean((batch_pred_obs - torch.as_tensor(batch_obs_after, dtype=torch.float32))**2)
        forward_optimizer.zero_grad()
        forward_loss.backward()
        forward_optimizer.step()
    print(f"loss_forward: {forward_loss.item()}")


    for j in range(v_updates_per_epoch):
        value_optimizer.zero_grad()
        value_loss = value_net(torch.as_tensor(batch_obs, dtype=torch.float32)) - torch.as_tensor(batch_reward_to_go,dtype=torch.float32)
        value_loss = (value_loss ** 2).mean()
        value_loss.backward()
        value_optimizer.step()
    print(f"loss_vf: {value_loss.item()}")


    avg_ep_reward = sum(batch_ep_reward)/len(batch_ep_reward)
    print(f"ep reward: {avg_ep_reward}")
    print()
    vis_dict["reward"].append(avg_ep_reward)
    vis_dict["loss_pi"].append(batch_loss.item())
    vis_dict["loss_v"].append(value_loss.item())



batch_size = 1000
num_epochs = 300
expert_prob = 1
expert_trajectories = load_expert_traj("mcar.pickle")
reward_scaling = 1

policy_optimizer = torch.optim.Adam(logit_net.parameters(), lr=3e-4)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)
forward_optimizer = torch.optim.Adam(forward_net.parameters(), lr=1e-3)

render_every = 10

for epoch in range(num_epochs):
    print(f"epoch: {epoch}")
    render = False
    if epoch % render_every == 0:
        render = True
    if epoch > 100:
        expert_prob = 0
    train_one_epoch(policy_optimizer, value_optimizer, forward_optimizer, batch_size, expert_prob, expert_trajectories, reward_scaling, render)

plot_dict(vis_dict)
env.close()
