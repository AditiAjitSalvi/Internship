import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import random

# =============================
# HYPERPARAMETERS
# =============================
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
MAX_STEPS = 50
MAX_STATIONS = 50
D_MODEL = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================
# COMMAND VOCAB
# =============================
CMD = {"GET": 0, "PUT": 1, "WAIT": 2}
N_CMDS = len(CMD)

# =============================
# ACTOR-CRITIC MODEL
# =============================
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(MAX_STATIONS + 1, D_MODEL)
        self.fc = nn.Sequential(
            nn.Linear(D_MODEL, 128),
            nn.ReLU(),
        )
        self.actor_cmd = nn.Linear(128, N_CMDS)
        self.actor_stn = nn.Linear(128, MAX_STATIONS)
        self.critic = nn.Linear(128, 1)

    def forward(self, station):
        x = self.embed(station)
        x = self.fc(x)
        return (
            torch.softmax(self.actor_cmd(x), dim=-1),
            torch.softmax(self.actor_stn(x), dim=-1),
            self.critic(x)
        )

# =============================
# PPO AGENT
# =============================
class PPOAgent:
    def __init__(self):
        self.policy = ActorCritic().to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.long).to(device)

        probs_cmd, probs_stn, value = self.policy(state)

        dist_cmd = Categorical(probs_cmd)
        dist_stn = Categorical(probs_stn)

        a_cmd = dist_cmd.sample()
        a_stn = dist_stn.sample()

        log_prob = dist_cmd.log_prob(a_cmd) + dist_stn.log_prob(a_stn)

        return (
            a_cmd.item(),
            a_stn.item(),
            log_prob.detach(),     # 🔥 FIXED
            value.detach()         # 🔥 FIXED
        )

    def update(self, memory):
        states, actions, old_log_probs, rewards, values = zip(*memory)

        # ===== Compute Returns =====
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + GAMMA * G
            returns.insert(0, G)

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        values = torch.cat(values).squeeze()
        advantages = (returns - values).detach()   # 🔥 FIXED

        old_log_probs = torch.stack(old_log_probs).detach()

        # ===== PPO Update =====
        for _ in range(K_EPOCHS):
            new_log_probs = []
            new_values = []

            for s, (a_cmd, a_stn) in zip(states, actions):
                probs_cmd, probs_stn, val = self.policy(
                    torch.tensor(s, dtype=torch.long).to(device)
                )

                dist_cmd = Categorical(probs_cmd)
                dist_stn = Categorical(probs_stn)

                new_log_probs.append(
                    dist_cmd.log_prob(torch.tensor(a_cmd).to(device)) +
                    dist_stn.log_prob(torch.tensor(a_stn).to(device))
                )
                new_values.append(val)

            new_log_probs = torch.stack(new_log_probs)
            new_values = torch.cat(new_values).squeeze()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (returns - new_values).pow(2).mean()

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

# =============================
# ENVIRONMENT
# =============================
class WagonEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.occupied = [0] * MAX_STATIONS
        self.current_station = random.randint(0, MAX_STATIONS - 1)
        self.steps = 0
        return self.current_station

    def step(self, cmd, stn):
        reward = -0.1
        done = False
        self.steps += 1

        if cmd == CMD["GET"]:
            reward += 1.0

        elif cmd == CMD["PUT"]:
            if self.occupied[stn] == 0:
                self.occupied[stn] = 1
                reward += 3.0
            else:
                reward -= 2.0

        elif cmd == CMD["WAIT"]:
            reward += 0.5

        self.current_station = stn

        if self.steps >= MAX_STEPS:
            done = True

        return self.current_station, reward, done

# =============================
# TRAINING LOOP
# =============================
def train():
    agent = PPOAgent()
    env = WagonEnv()

    for ep in range(1, 301):
        state = env.reset()
        memory = []
        ep_reward = 0

        for _ in range(MAX_STEPS):
            a_cmd, a_stn, logp, val = agent.select_action(state)
            next_state, reward, done = env.step(a_cmd, a_stn)

            memory.append((state, (a_cmd, a_stn), logp, reward, val))
            state = next_state
            ep_reward += reward
            if done:
                break

        agent.update(memory)

        if ep % 10 == 0:
            print(f"Episode {ep} | Total Reward: {ep_reward:.2f}")

    torch.save(agent.policy.state_dict(), "ppo_wagon_model.pth")
    print("✅ Training complete. Model saved.")

# =============================
# MAIN
# =============================
if __name__ == "__main__":
    train()
