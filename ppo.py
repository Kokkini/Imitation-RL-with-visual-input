import torch
from torch.distributions import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt
import room

room_size = 10
num_tasks = 2
work_per_task = 8
env = room.Room(room_size, num_tasks, work_per_task, max_steps=200)

# env = gym.make("CartPole-v0")

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
activation = torch.nn.Tanh
logit_net = get_logit_net(policy_net_sizes, activation)
value_net = get_logit_net(value_net_sizes, activation)
print(logit_net)
print(value_net)


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

# def GAE_reward(rews, vals, gamma, lam):
#     rew_to_go = reward_to_go(rews, gamma)
#     T = len(rews) - 1
#     result = []
#     for t in range(len(rews)):
#         if t < T:
#             r = 0
#             for n in range(1, T-t):
#                 r += n_step_reward_at(t, n, rew_to_go, vals, gamma) * lam**(n-1)
#             r *= (1 - lam)
#             r += (lam**(T-t-1)) * n_step_reward_at(t, T-t, rew_to_go, vals, gamma)
#         else:
#             r = n_step_reward_at(t, 1, rew_to_go, vals, gamma)
#         result.append(r)
#     return result

def GAE_reward(rews, vals, gamma, lam):
    vals_arr = np.array(vals)
    delta_t = rews + gamma * vals_arr[1:] - vals_arr[:-1]
    return reward_to_go(delta_t, gamma*lam)

def get_advantage(reward_to_go, values):
    return reward_to_go - values

def train_one_epoch(policy_optimizer, value_optimizer, batch_size, render = True):
    obs = env.reset()
    batch_obs = []
    batch_act = []
    batch_weight = []
    batch_reward_to_go = []
    ep_val = []
    ep_reward = []
    batch_log_prob = []
    render_this_epoch = False
    gamma = 0.99
    batch_ep_reward = []
    pi_updates_per_epoch = 20
    v_updates_per_epoch = 20

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

        obs, reward, done, info = env.step(act)
        ep_reward.append(reward)

        if done:
            batch_ep_reward.append(sum(ep_reward))
            render_this_epoch = True
            # weights = list(reward_to_go(ep_reward, gamma))
            # weights = n_step_reward(ep_reward, ep_val, gamma, 30)
            last_val = value_net(torch.as_tensor(obs, dtype=torch.float32)).item()
            weights = GAE_reward(ep_reward, ep_val + [last_val], gamma, lam=0.97)
            batch_weight += weights
            # ep_reward += [last_val] #boostrap
            batch_reward_to_go += reward_to_go(ep_reward, gamma)
            obs = env.reset()
            ep_reward = []
            ep_val = []
            done = False
            if len(batch_obs) >= batch_size:
                break

    # advatages = get_advantage(torch.as_tensor(batch_reward_to_go, dtype=torch.float32), value_net(torch.as_tensor(batch_obs, dtype=torch.float32)))
    # advatages = advatages.data.numpy()
    for j in range(v_updates_per_epoch):
        value_optimizer.zero_grad()
        value_loss = value_net(torch.as_tensor(batch_obs, dtype=torch.float32)) - torch.as_tensor(batch_reward_to_go,dtype=torch.float32)
        value_loss = (value_loss ** 2).mean()
        value_loss.backward()
        value_optimizer.step()
    print(f"loss_vf: {value_loss.item()}")

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
    avg_ep_reward = sum(batch_ep_reward)/len(batch_ep_reward)
    print(f"ep reward: {avg_ep_reward}")
    print()
    vis_dict["reward"].append(avg_ep_reward)
    vis_dict["loss_pi"].append(batch_loss.item())
    vis_dict["loss_v"].append(value_loss.item())



batch_size = 1000
num_epochs = 100
policy_optimizer = torch.optim.Adam(logit_net.parameters(), lr=3e-4)
value_optimizer = torch.optim.Adam(value_net.parameters(), lr=1e-3)


for epoch in range(num_epochs):
    print(f"epoch: {epoch}")
    train_one_epoch(policy_optimizer, value_optimizer, batch_size, render=False)

plot_dict(vis_dict)
