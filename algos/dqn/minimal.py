from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LineWorld:
    def __init__(self, length: int = 7) -> None:
        assert length >= 3
        self.length = length
        self.start_state = length // 2
        self.state = self.start_state

    @property
    def num_states(self) -> int:
        return self.length

    @property
    def num_actions(self) -> int:
        return 2

    def encode(self, state: int) -> np.ndarray:
        obs = np.zeros(self.length, dtype=np.float32)
        obs[state] = 1.0
        return obs

    def reset(self) -> np.ndarray:
        self.state = self.start_state
        return self.encode(self.state)

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        move = -1 if action == 0 else 1
        next_state = min(max(self.state + move, 0), self.length - 1)
        self.state = next_state

        if next_state == 0:
            return self.encode(next_state), -1.0, True
        if next_state == self.length - 1:
            return self.encode(next_state), 1.0, True
        return self.encode(next_state), -0.02, False


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(
            maxlen=capacity
        )

    def __len__(self) -> int:
        return len(self.buffer)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.stack(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.stack(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )


class QNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def select_action(model: QNet, state: np.ndarray, epsilon: float) -> int:
    if random.random() < epsilon:
        return random.randrange(2)

    state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        return int(model(state_tensor).argmax(dim=-1).item())


def train_step(
    q_net: QNet,
    target_net: QNet,
    optimizer: torch.optim.Optimizer,
    batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    gamma: float,
) -> float:
    states, actions, rewards, next_states, dones = batch

    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    with torch.no_grad():
        next_q_values = target_net(next_states).max(dim=1).values
        targets = rewards + gamma * (1.0 - dones) * next_q_values

    loss = F.smooth_l1_loss(q_values, targets)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def evaluate(env: LineWorld, model: QNet) -> tuple[list[int], float]:
    obs = env.reset()
    trajectory = [int(np.argmax(obs))]
    total_reward = 0.0
    done = False

    while not done and len(trajectory) < env.length + 2:
        action = select_action(model, obs, epsilon=0.0)
        obs, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(int(np.argmax(obs)))

    return trajectory, total_reward


def main() -> None:
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_printoptions(precision=3, sci_mode=False)

    env = LineWorld(length=7)
    q_net = QNet(input_dim=env.num_states, hidden_dim=32, num_actions=env.num_actions)
    target_net = QNet(
        input_dim=env.num_states, hidden_dim=32, num_actions=env.num_actions
    )
    target_net.load_state_dict(q_net.state_dict())

    optimizer = torch.optim.Adam(q_net.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(capacity=2_000)

    gamma = 0.95
    batch_size = 32
    sync_every = 20
    global_step = 0

    for episode in range(250):
        obs = env.reset()
        done = False
        epsilon = max(0.05, 0.9 - 0.85 * episode / 249)

        while not done:
            action = select_action(q_net, obs, epsilon=epsilon)
            next_obs, reward, done = env.step(action)
            replay_buffer.push(obs, action, reward, next_obs, float(done))

            obs = next_obs
            global_step += 1

            if len(replay_buffer) >= batch_size:
                batch = replay_buffer.sample(batch_size)
                train_step(q_net, target_net, optimizer, batch, gamma=gamma)

            if global_step % sync_every == 0:
                target_net.load_state_dict(q_net.state_dict())

    trajectory, total_reward = evaluate(env, q_net)
    with torch.no_grad():
        q_table = q_net(torch.eye(env.num_states))

    print("Estimated Q-values:")
    print(q_table)
    print("\nGreedy trajectory:")
    print(trajectory)
    print("Greedy return:", round(total_reward, 3))


if __name__ == "__main__":
    main()
