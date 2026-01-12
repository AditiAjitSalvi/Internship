import os
import csv
import random
import math
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import time

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
wagon_config_csv = r"e:\Internship\code\wagon_config.csv"
tanks_csv_default = r"e:\Internship\code\tanks_csv.csv"
station_master_csv = r"e:\Internship\WayTime-dB.mdb\StationMaster.csv"
csv_programs = r"e:\Internship\WayTime-dB.mdb\AutoSequencePrograms.csv"
csv_zones = r"e:\Internship\WayTime-dB.mdb\CrossTrolleyMaster.csv"

# RL Hyperparameters
LR = 1e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 4
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
MAX_STATIONS = 200 
MAX_STEPS = 50     
BATCH_SIZE = 16
UPDATE_ITERS = 5 # PPO update iterations

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# ==========================================
# DATA LOADING & VERIFICATION
# ==========================================
def load_wagon_config(config_path=wagon_config_csv):
    """
    Loads wagon configuration from CSV.
    """
    config = {}
    if not os.path.exists(config_path):
        return config
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                w_num = (row.get('Transporter Name') or row.get('Wagon Number') or "").strip()
                if not w_num:
                    continue
                
                config[w_num] = {
                    'sf': float(row.get('Superfast Speed', 0)),
                    'f': float(row.get('Fast Speed', 0)),
                    's': float(row.get('Slow Speed', 0)),
                    'lift_time': float(row.get('Lift Time', 0)),
                    'lower_time': float(row.get('Lower Time', 0)),
                    'min_stn': int(row.get('Minimum Station No', 1)),
                    'max_stn': int(row.get('Maximum Station No', 200)),
                    'basic_pos': int(row.get('Basic Position', 0))
                }
    except Exception as e:
        print(f"Error loading wagon config: {e}")
    return config

def verify_stations(tanks, min_stn, max_stn):
    """
    Verifies if the total number of unique stations in the tank list matches 
    the expected count (Maximum Station No).
    """
    present_stations = {tank.get('station_no') for tank in tanks if tank.get('station_no')}
    total_count = len(present_stations)
    expected_count = max_stn

    print(f"\n[STATION VERIFICATION]")
    print(f"Goal: Total stations should be {expected_count}")
    print(f"Status: Found {total_count} stations in the tank data.")
    
    if total_count == expected_count:
        print(f"SUCCESS: Total station count matches Maximum Station No ({expected_count}).")
        return True
    else:
        diff = expected_count - total_count
        status = "missing" if diff > 0 else "extra"
        print(f"FAILURE: Total station count ({total_count}) does not match Maximum Station No ({expected_count}).")
        print(f"There are {abs(diff)} {status} stations.")
        return False

def load_data():
    """ Loads historical sequence data for training RL models. """
    print("Loading historical data for AI training...")
    sequences = defaultdict(list)
    vocab_cmd = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    
    if not os.path.exists(csv_programs):
        print(f"Warning: {csv_programs} not found. AI training will use empty data.")
        return [], {}, vocab_cmd

    with open(csv_programs, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seq_id = row['ProgramNo'] + "_" + row['ProjectID']
                cmd = row['Instruction']
                stn = int(row['InstructionValue']) if row['InstructionValue'] else 0
                stn = min(stn, MAX_STATIONS - 1)
                if cmd not in vocab_cmd:
                    vocab_cmd[cmd] = len(vocab_cmd)
                sequences[seq_id].append((vocab_cmd[cmd], stn))
            except:
                continue

    train_data = list(sequences.values())
    print(f"Loaded {len(train_data)} sequences for training.")
    
    adj_list = defaultdict(list)
    if os.path.exists(csv_zones):
        with open(csv_zones, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    s1 = int(row['Row1StationNo'])
                    s2 = int(row['Row2StationNo'])
                    adj_list[s1].append(s2)
                    adj_list[s2].append(s1)
                except:
                    continue
                
    return train_data, adj_list, vocab_cmd

# ==========================================
# ML MODELS (TRANSFORMER + GNN + PPO)
# ==========================================
class SeqTransformer(nn.Module):
    def __init__(self, n_cmds, n_stations, d_model=D_MODEL):
        super(SeqTransformer, self).__init__()
        self.cmd_emb = nn.Embedding(n_cmds, d_model)
        self.stn_emb = nn.Embedding(n_stations, d_model)
        self.pos_emb = nn.Embedding(MAX_STEPS, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=N_HEADS, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.fc_cmd = nn.Linear(d_model, n_cmds)
        self.fc_stn = nn.Linear(d_model, n_stations)

    def forward(self, cmd_seq, stn_seq):
        seq_len = cmd_seq.size(1)
        pos = torch.arange(seq_len, device=cmd_seq.device).unsqueeze(0).expand(cmd_seq.size(0), -1)
        x = self.cmd_emb(cmd_seq) + self.stn_emb(stn_seq) + self.pos_emb(pos)
        x = self.transformer(x)
        return self.fc_cmd(x), self.fc_stn(x)

class SafetyGNN(nn.Module):
    def __init__(self, n_stations, d_model=D_MODEL):
        super(SafetyGNN, self).__init__()
        self.state_emb = nn.Embedding(2, d_model)
        self.w = nn.Linear(d_model, d_model)

    def forward(self, station_states, adj_matrix):
        x = self.state_emb(station_states)
        out = []
        for i in range(x.size(0)):
            h = x[i]
            aggr = torch.mm(adj_matrix, h) 
            h_next = torch.relu(self.w(aggr))
            out.append(h_next)
        return torch.stack(out)

class ActorCritic(nn.Module):
    def __init__(self, n_cmds, n_stations, d_model=D_MODEL):
        super(ActorCritic, self).__init__()
        self.transformer = SeqTransformer(n_cmds, n_stations, d_model)
        self.gnn = SafetyGNN(n_stations, d_model)
        self.actor_cmd = nn.Linear(d_model * 2, n_cmds)
        self.actor_stn = nn.Linear(d_model * 2, n_stations)
        self.critic = nn.Linear(d_model * 2, 1)

    def forward(self, cmd_seq, stn_seq, station_states, adj_matrix):
        # Extract context from Transformer
        seq_len = cmd_seq.size(1)
        pos = torch.arange(seq_len, device=cmd_seq.device).unsqueeze(0).expand(cmd_seq.size(0), -1)
        x_seq = self.transformer.cmd_emb(cmd_seq) + self.transformer.stn_emb(stn_seq) + self.transformer.pos_emb(pos)
        x_seq = self.transformer.transformer(x_seq)
        ctx_seq = x_seq[:, -1, :] 

        # Extract context from Safety GNN
        x_gnn = self.gnn(station_states, adj_matrix)
        ctx_safe, _ = torch.max(x_gnn, dim=1)
        
        state = torch.cat([ctx_seq, ctx_safe], dim=-1)
        return torch.softmax(self.actor_cmd(state), dim=-1), torch.softmax(self.actor_stn(state), dim=-1), self.critic(state)
class Memory:
    def __init__(self):
        self.actions_cmd = []
        self.actions_stn = []
        self.states_cmd = []
        self.states_stn = []
        self.states_safe = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions_cmd[:]
        del self.actions_stn[:]
        del self.states_cmd[:]
        del self.states_stn[:]
        del self.states_safe[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class PPOAgent:
    def __init__(self, vocab_cmd, n_stations, adj_list):
        self.vocab_cmd = vocab_cmd
        self.n_stations = n_stations
        self.adj_matrix = self._build_adj_matrix(adj_list, n_stations)
        self.policy = ActorCritic(len(vocab_cmd), n_stations).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        self.MseLoss = nn.MSELoss()
        
    def update(self, memory):
        # Basic PPO Update
        old_states_cmd = torch.stack(memory.states_cmd).to(device).detach()
        old_states_stn = torch.stack(memory.states_stn).to(device).detach()
        old_states_safe = torch.stack(memory.states_safe).to(device).detach()
        old_actions_cmd = torch.tensor(memory.actions_cmd).to(device).detach()
        old_actions_stn = torch.tensor(memory.actions_stn).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()
        
        # Calculate Rewards-to-go
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        
        for _ in range(UPDATE_ITERS):
            # Evaluating old actions and values
            # Flatten the batch for the policy
            probs_cmd, probs_stn, state_values = self.policy(old_states_cmd, old_states_stn, old_states_safe, self.adj_matrix)
            
            dist_cmd = Categorical(probs_cmd)
            dist_stn = Categorical(probs_stn)
            
            logprobs = dist_cmd.log_prob(old_actions_cmd) + dist_stn.log_prob(old_actions_stn)
            dist_entropy = dist_cmd.entropy() + dist_stn.entropy()
            state_values = torch.squeeze(state_values)
            
            # ratios = exp(logprobs - old_logprobs)
            ratios = torch.exp(logprobs - old_logprobs)

            # Surrogates
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages

            # Loss
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

    def _build_adj_matrix(self, adj_list, n):
        mat = torch.zeros((n, n), device=device)
        for u, neighbors in adj_list.items():
            if u < n:
                for v in neighbors:
                    if v < n:
                        mat[u, v] = 1.0
        return mat

    def select_action(self, cmd_seq, stn_seq, station_states):
        # Truncate sequences to MAX_STEPS
        cmd_seq = cmd_seq[-MAX_STEPS:]
        stn_seq = stn_seq[-MAX_STEPS:]
        cmd_seq_t = torch.tensor([cmd_seq], dtype=torch.long).to(device)
        stn_seq_t = torch.tensor([stn_seq], dtype=torch.long).to(device)
        station_states = torch.tensor([station_states], dtype=torch.long).to(device)
        with torch.no_grad():
            probs_cmd, probs_stn, val = self.policy(cmd_seq_t, stn_seq_t, station_states, self.adj_matrix)
        dist_cmd = Categorical(probs_cmd)
        dist_stn = Categorical(probs_stn)
        action_cmd = dist_cmd.sample()
        action_stn = dist_stn.sample()
        log_prob = dist_cmd.log_prob(action_cmd) + dist_stn.log_prob(action_stn)
        return action_cmd.item(), action_stn.item(), log_prob, val

class WagonEnv:
    def __init__(self, adj_list, max_stations):
        self.adj_list = adj_list
        self.max_stations = max_stations
        self.reset()
    def reset(self):
        self.station_occupancy = [0] * self.max_stations
        self.step_count = 0
        return self.station_occupancy
    def step(self, cmd, stn):
        reward = 0
        done = False
        self.step_count += 1
        if stn >= self.max_stations: return self.station_occupancy, -10, True
        if self.station_occupancy[stn] == 1: reward -= 10
        elif any(self.station_occupancy[n] == 1 for n in self.adj_list.get(stn, [])): reward -= 5
        self.station_occupancy[stn] = 1 
        reward += 1
        if self.step_count >= MAX_STEPS: done = True
        return self.station_occupancy, reward, done

# ==========================================
# SEQUENCE GENERATION (PHYSICS-BASED)
# ==========================================
def calculate_time_value(distance1, distance2, distance3, sfspeed, fspeed, sspeed):
    try:
        sfs = sfspeed * 16.66 if sfspeed > 0 else 1.0
        fs = fspeed * 16.66 if fspeed > 0 else 1.0
        ss = sspeed * 16.66 if sspeed > 0 else 1.0
        return (distance1 / sfs) + (distance2 / fs) + (distance3 / ss)
    except: return 0

def gap_analysis_sequence(tanks, wagon_config, wagon_id):
    if wagon_id not in wagon_config:
        raise ValueError(f"Wagon configuration for {wagon_id} missing.")
        
    w = wagon_config[wagon_id]
    sequence_data = [["Command", "Value", "TravelTime", "AccumulatedTime"]]
    accumulated_time = 0.0
    
    for i in range(len(tanks) - 1):
        curr_tank = tanks[i]
        next_tank = tanks[i+1]
        s_curr = int(curr_tank.get('station_no', 0))
        s_next = int(next_tank.get('station_no', 0))
        dist_curr = float(curr_tank.get('distance_mm', 0))
        dist_next = float(next_tank.get('distance_mm', 0))
        dist_total = abs(dist_next - dist_curr)
        
        # Physics Parameters
        distance3 = 500.0 # Slow speed approach distance
        distance2 = max(0, dist_total - distance3)
        distance1 = dist_total 
        
        travel_time = calculate_time_value(distance1, distance2, distance3, w['sf'], w['f'], w['s'])
        
        # GET FROM
        sequence_data.append(["GET FROM", s_curr, f"{w['lift_time']:.2f}", f"{accumulated_time:.2f}"])
        accumulated_time += w['lift_time']
        
        # PUT ON
        sequence_data.append(["PUT ON", s_next, f"{(travel_time + w['lower_time']):.2f}", f"{accumulated_time:.2f}"])
        accumulated_time += travel_time + w['lower_time']
        
        # WAIT
        dip_time = float(next_tank.get('dip_time_sec', 0))
        if dip_time > 0:
            sequence_data.append(["WAIT", int(dip_time), "0.00", f"{accumulated_time:.2f}"])
            accumulated_time += dip_time
            
    return sequence_data

def generate_sequence_from_tanks(csv_path, speeds_input=None):
    print(f"Processing sequence for {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    config = load_wagon_config()
    wagon_id = speeds_input if speeds_input else "Wagon 1"
    
    if wagon_id not in config:
        print(f"Error: Wagon '{wagon_id}' not found in configuration.")
        return

    w = config[wagon_id]

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            tanks = list(reader)
        
        # 1. Station Verification
        if not verify_stations(tanks, w['min_stn'], w['max_stn']):
            print("\n[WARNING] Verification failed. Stations in the required range are potentially missing.")
        else:
            print("\n[INFO] Verification passed.")

        # 2. Generate Sequence
        print(f"\nGenerating sequence for {wagon_id}...")
        seq_data = gap_analysis_sequence(tanks, config, wagon_id)
        
        # 3. Print Results
        print("\n" + tabulate(seq_data[1:], headers=seq_data[0], tablefmt="grid"))
        
    except Exception as e:
        print(f"Error generating sequence: {e}")

def evaluate_model(agent, train_data, vocab_cmd):
    """ Evaluates the model on training data and returns true vs predicted labels. """
    y_true = []
    y_pred = []
    rev_vocab = {v: k for k, v in vocab_cmd.items()}
    
    print("\nEvaluating model for Confusion Matrix...")
    for seq in train_data:
        curr_cmd_seq = [vocab_cmd["<SOS>"]]
        curr_stn_seq = [0]
        state = [0] * agent.n_stations
        
        # We predict the next command in the sequence
        for cmd_idx, stn_val in seq:
            a_cmd, a_stn, _, _ = agent.select_action(curr_cmd_seq, curr_stn_seq, state)
            
            y_true.append(rev_vocab.get(cmd_idx, "UNKNOWN"))
            y_pred.append(rev_vocab.get(a_cmd, "UNKNOWN"))
            
            # Update for next step
            curr_cmd_seq.append(cmd_idx)
            safe_stn = min(stn_val, agent.n_stations - 1)
            curr_stn_seq.append(safe_stn)
            if safe_stn < agent.n_stations:
                state[safe_stn] = 1
                
    return y_true, y_pred

def plot_confusion_matrix(y_true, y_pred, labels):
    """ Plots and saves a confusion matrix. """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
    plt.xlabel('Predicted Command')
    plt.ylabel('True Command')
    plt.title('Confusion Matrix: Model Command Predictions')
    plt.tight_layout()
    
    # Use absolute path to ensure saving
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'confusion_matrix.png')
    plt.savefig(save_path)
    plt.close() # Free memory
    print(f"Confusion matrix plot successfully saved at: {save_path}")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    global wagon_config_csv
    parser = argparse.ArgumentParser(description="Sequence Generation and AI Training Tool")
    parser.add_argument("--mode", type=str, choices=["gen", "train"], default="gen", help="Execution mode")
    parser.add_argument("--input", type=str, help="Path to tanks CSV", default=tanks_csv_default)
    parser.add_argument("--speeds", type=str, help="Wagon Name", default="Wagon 1")
    parser.add_argument("--config", type=str, help="Path to wagon config CSV", default=wagon_config_csv)
    
    args = parser.parse_args()
    wagon_config_csv = args.config

    if args.mode == "gen":
        generate_sequence_from_tanks(args.input, args.speeds)
    else:
        # AI Training Mode
        train_data, adj_list, vocab_cmd = load_data()
        if not train_data:
            print("No training data found. Exiting.")
            return

        agent = PPOAgent(vocab_cmd, MAX_STATIONS, adj_list)
        memory = Memory()
        env = WagonEnv(adj_list, MAX_STATIONS)
        
        print("\nStarting RL Training Loop...")
        start_time = time.time()
        for episode in range(1, 11): 
            state = env.reset()
            curr_cmd_seq = [vocab_cmd["<SOS>"]]
            curr_stn_seq = [0]
            ep_reward = 0
            
            for t in range(MAX_STEPS):
                # We need to feed the sequence in a format the model likes (last N steps)
                # For simplicity in this training loop, we use the sequences as they grow
                c_seq = curr_cmd_seq[-MAX_STEPS:]
                s_seq = curr_stn_seq[-MAX_STEPS:]
                
                # Convert list to tensor for memory storage
                c_seq_t = torch.tensor(c_seq, dtype=torch.long)
                # Pad to MAX_STEPS if needed or just handle variable size
                # Here we pad with 0 for simplicity in this batching example
                c_pad = torch.zeros(MAX_STEPS, dtype=torch.long)
                c_pad[:len(c_seq)] = c_seq_t
                
                s_pad = torch.zeros(MAX_STEPS, dtype=torch.long)
                s_pad[:len(s_seq)] = torch.tensor(s_seq, dtype=torch.long)
                
                safe_state_t = torch.tensor(state, dtype=torch.long)

                action_cmd, action_stn, log_prob, val = agent.select_action(curr_cmd_seq, curr_stn_seq, state)
                
                # Store in memory
                memory.states_cmd.append(c_pad)
                memory.states_stn.append(s_pad)
                memory.states_safe.append(safe_state_t)
                memory.actions_cmd.append(action_cmd)
                memory.actions_stn.append(action_stn)
                memory.logprobs.append(log_prob)
                
                next_state, reward, done = env.step(action_cmd, action_stn)
                
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                curr_cmd_seq.append(action_cmd)
                curr_stn_seq.append(action_stn)
                state = next_state
                ep_reward += reward
                if done: break
            
            # Update the agent
            agent.update(memory)
            memory.clear()
            
            print(f"Episode {episode}: Total Reward = {ep_reward:.2f}")
            
        end_time = time.time()
        print(f"Training Finished in {end_time - start_time:.2f}s.")
        
        model_save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_v8.pth")
        torch.save(agent.policy.state_dict(), model_save_path)
        print(f"Model saved as '{model_save_path}'")

        # Confusion Matrix Generation
        y_true, y_pred = evaluate_model(agent, train_data, vocab_cmd)
        labels = sorted(list(vocab_cmd.keys()))
        plot_confusion_matrix(y_true, y_pred, labels)

if __name__ == "__main__":
    main()

#python v7.py --input tanks_csv.csv --speeds "Wagon 1"