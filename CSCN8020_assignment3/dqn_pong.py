import gym
import numpy as np
import random
from collections import deque
import csv
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from assignment3_utils import process_frame, transform_reward

# ============================================================
# 1. Config & Utilities
# ============================================================

ENV_NAME = "PongDeterministic-v4"
IMAGE_SHAPE = (84, 80)      # (height, width) after crop + downsample
NUM_FRAMES = 4              # stack of 4 frames
GAMMA = 0.95                # discount
LEARNING_RATE = 1e-4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


def preprocess_obs(obs, image_shape=IMAGE_SHAPE):
    """
    Uses the prof's utils to:
    - crop
    - downsample (by 2)
    - convert to grayscale
    - normalize to [-1, 1]
    Returns a 2D frame of shape (H, W).
    """
    # process_frame returns shape (1, H, W, 1)
    processed = process_frame(obs, image_shape)
    processed = processed[0, :, :, 0]   # -> (H, W)
    return processed.astype(np.float32)


def make_initial_state(env):
    """
    Resets the env and creates an initial state with 4 identical frames
    stacked along the channel dimension: (C=4, H, W).
    """
    obs, info = env.reset()
    frame = preprocess_obs(obs)
    state = np.stack([frame] * NUM_FRAMES, axis=0)  # (4, 84, 80)
    return state


def update_state(state, new_frame):
    """
    Given current state (4, H, W) and a new frame (H, W),
    returns the next state (4, H, W) by dropping the oldest frame.
    """
    new_state = np.concatenate(
        (state[1:, :, :], np.expand_dims(new_frame, axis=0)), axis=0
    )
    return new_state


# ============================================================
# 2. Replay Buffer
# ============================================================

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        # store as np arrays
        self.buffer.append(
            (np.array(state, copy=False),
             action,
             reward,
             np.array(next_state, copy=False),
             done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = np.stack(states)
        next_states = np.stack(next_states)
        actions = np.array(actions)
        rewards = np.array(rewards, dtype=np.float32)
        dones = np.array(dones, dtype=np.uint8)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 3. DQN Network (CNN)
# ============================================================

class DQN(nn.Module):
    def __init__(self, in_channels, num_actions):
        super(DQN, self).__init__()

        # input: (B, 4, 84, 80)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # compute conv output size manually or with a dummy forward
        # For 84x80, this works out to 64 * 8 * 6 = 3072
        self.fc_input_dim = 64 * 8 * 6

        self.fc1 = nn.Linear(self.fc_input_dim, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        # x: (B, C=4, H, W)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)


# ============================================================
# 4. Epsilon-greedy policy
# ============================================================

def select_action(state, policy_net, epsilon, num_actions):
    """
    state: np.array (4, 84, 80)
    """
    if random.random() < epsilon:
        # Explore
        return random.randrange(num_actions)
    else:
        # Exploit
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)  # (1,4,84,80)
            q_values = policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())


# ============================================================
# 5. Training loop for one configuration
# ============================================================

def train_dqn(
    env_name=ENV_NAME,
    num_episodes=300,           # you can increase later (e.g., 500+)
    batch_size=8,
    target_update_interval=10,  # episodes
    replay_capacity=50000,
    epsilon_init=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.995,
    gamma=GAMMA,
    learning_rate=LEARNING_RATE,
    results_csv="results.csv",
    seed=42
):
    # ------------- Setup --------------
    env = gym.make(env_name)
    env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_actions = env.action_space.n

    policy_net = DQN(NUM_FRAMES, num_actions).to(DEVICE)
    target_net = DQN(NUM_FRAMES, num_actions).to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(replay_capacity)

    epsilon = epsilon_init

    # ------------- Logging --------------
    os.makedirs(os.path.dirname(results_csv), exist_ok=True)
    csv_file = open(results_csv, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "episode",
        "steps",
        "score",
        "total_reward",
        "avg_reward_last5",
        "epsilon"
    ])

    episode_rewards = []

    global_step = 0

    # ------------- Training episodes --------------
    for episode in range(1, num_episodes + 1):
        state = make_initial_state(env)
        done = False
        episode_reward = 0.0
        episode_steps = 0
        score = 0      # Pong score (sum of transformed rewards)

        while not done:
            global_step += 1
            episode_steps += 1

            action = select_action(state, policy_net, epsilon, num_actions)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Optionally transform reward to -1 / 0 / +1 to stabilize learning
            transformed_r = transform_reward(reward)
            episode_reward += transformed_r
            score += transformed_r

            next_frame = preprocess_obs(obs)
            next_state = update_state(state, next_frame)

            replay_buffer.push(state, action, transformed_r, next_state, done)
            state = next_state

            # --------- Optimization step once buffer is warm ----------
            if len(replay_buffer) >= batch_size * 5:
                states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)

                states_b = torch.from_numpy(states_b).float().to(DEVICE)        # (B,4,84,80)
                next_states_b = torch.from_numpy(next_states_b).float().to(DEVICE)
                actions_b = torch.from_numpy(actions_b).long().to(DEVICE)
                rewards_b = torch.from_numpy(rewards_b).to(DEVICE)
                dones_b = torch.from_numpy(dones_b).float().to(DEVICE)

                # Q(s,a)
                q_values = policy_net(states_b)
                q_values = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                # max_a' Q_target(s', a')
                with torch.no_grad():
                    next_q_values = target_net(next_states_b).max(1)[0]
                    target = rewards_b + gamma * next_q_values * (1 - dones_b)

                loss = nn.MSELoss()(q_values, target)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

        # --------- End episode ----------
        episode_rewards.append(episode_reward)

        # Update epsilon per episode
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
            if epsilon < epsilon_min:
                epsilon = epsilon_min

        # Update target network every N episodes
        if episode % target_update_interval == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Average reward of last 5 episodes
        last5 = episode_rewards[-5:]
        avg_last5 = float(np.mean(last5))

        csv_writer.writerow([episode, episode_steps, score, episode_reward, avg_last5, epsilon])
        csv_file.flush()

        print(
            f"Episode {episode:4d} | Steps: {episode_steps:4d} "
            f"| Score: {score:4.1f} | EpReward: {episode_reward:5.1f} "
            f"| AvgLast5: {avg_last5:6.2f} | eps: {epsilon:.3f}"
        )

    env.close()
    csv_file.close()

    return policy_net


if __name__ == "__main__":

    # 1) Baseline: batch_size = 8, target_update = 10
    train_dqn(
        num_episodes=300,          
        batch_size=8,
        target_update_interval=10,
        results_csv="results/baseline_bs8_tu10.csv"
    )

    # 2) Batch size experiment: batch_size = 16, target_update = 10
    train_dqn(
        num_episodes=300,
        batch_size=16,
        target_update_interval=10,
        results_csv="results/batch16_bs16_tu10.csv"
    )

    # 3) Target network experiment: batch_size = 8, target_update = 3
    train_dqn(
        num_episodes=300,
        batch_size=8,
        target_update_interval=3,
        results_csv="results/target3_bs8_tu3.csv"
    )
