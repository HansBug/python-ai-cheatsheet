from __future__ import annotations

import numpy as np


class LineWorld:
    def __init__(self, length: int = 5) -> None:
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

    def reset(self) -> int:
        self.state = self.start_state
        return self.state

    def step(self, action: int) -> tuple[int, float, bool]:
        move = -1 if action == 0 else 1
        next_state = min(max(self.state + move, 0), self.length - 1)
        self.state = next_state

        if next_state == 0:
            return next_state, -1.0, True
        if next_state == self.length - 1:
            return next_state, 1.0, True
        return next_state, -0.02, False


class QLearningAgent:
    def __init__(
        self,
        num_states: int,
        num_actions: int,
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 0.2,
    ) -> None:
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((num_states, num_actions), dtype=np.float32)

    def choose_action(self, state: int) -> int:
        if np.random.random() < self.epsilon:
            return int(np.random.randint(self.q_table.shape[1]))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> None:
        best_next_q = 0.0 if done else float(np.max(self.q_table[next_state]))
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error


def evaluate_policy(env: LineWorld, agent: QLearningAgent) -> tuple[list[int], float]:
    state = env.reset()
    trajectory = [state]
    total_reward = 0.0
    done = False

    while not done and len(trajectory) < env.length + 2:
        action = int(np.argmax(agent.q_table[state]))
        state, reward, done = env.step(action)
        total_reward += reward
        trajectory.append(state)

    return trajectory, total_reward


def main() -> None:
    np.random.seed(0)
    np.set_printoptions(precision=3, suppress=True)

    env = LineWorld(length=7)
    agent = QLearningAgent(
        num_states=env.num_states,
        num_actions=env.num_actions,
        lr=0.1,
        gamma=0.95,
        epsilon=0.2,
    )

    num_episodes = 400
    for episode in range(num_episodes):
        state = env.reset()
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state

        agent.epsilon = max(0.02, agent.epsilon * 0.995)

    trajectory, total_reward = evaluate_policy(env, agent)
    action_names = np.array(["left", "right"])
    greedy_actions = action_names[np.argmax(agent.q_table, axis=1)]

    print("Learned Q-table:")
    print(agent.q_table)
    print("\nGreedy action per state:")
    print(greedy_actions)
    print("\nGreedy trajectory:")
    print(trajectory)
    print("Greedy return:", round(total_reward, 3))


if __name__ == "__main__":
    main()
