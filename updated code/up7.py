import os
import csv
import argparse
import time
from collections import defaultdict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# ==========================================
# CONFIGURATION & HYPERPARAMETERS
# ==========================================
wagon_config_csv = r"e:\Internship\code\wagon_config.csv"
tanks_csv_default = r"e:\Internship\code\tanks_csv.csv"
csv_programs = r"e:\Internship\WayTime-dB.mdb\AutoSequencePrograms.csv"
csv_zones = r"e:\Internship\WayTime-dB.mdb\CrossTrolleyMaster.csv"
station_master_csv = r"e:\Internship\WayTime-dB.mdb\StationMaster.csv"

# RL HYPERPARAMETERS
LR = 3e-4
GAMMA = 0.99
EPS_CLIP = 0.1
K_EPOCHS = 8
UPDATE_ITERS = 10
ENTROPY_BETA = 0.05

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2

MAX_STEPS = 50
MAX_STATIONS = 500

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==========================================
# DATA LOADING & DIAGNOSTICS
# ==========================================
def load_wagon_config(path):
    config = {}
    if not os.path.exists(path):
        return config
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = (r.get('Transporter Name') or r.get('Wagon Number') or "").strip()
            if not name: continue
            config[name] = {
                'sf': float(r.get('Superfast Speed', 0)),
                'f': float(r.get('Fast Speed', 0)),
                's': float(r.get('Slow Speed', 0)),
                'lift': float(r.get('Lift Time', 0)),
                'lower': float(r.get('Lower Time', 0)),
                'min_stn': int(r.get('Minimum Station No', 1)),
                'max_stn': int(r.get('Maximum Station No', 200))
            }
    return config

def load_station_censor_data():
    censor_map = {}
    if not os.path.exists(station_master_csv):
        return censor_map
    try:
        with open(station_master_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pid = row['ProjectID']
                    stn = int(row['StationNumber'])
                    cdist = row.get('CensorDistance', '')
                    if cdist and cdist.strip():
                        censor_map[(pid, stn)] = float(cdist)
                except: continue
    except: pass
    return censor_map

def load_training_data():
    sequences = defaultdict(list)
    vocab = {"<PAD>":0, "<SOS>":1, "<EOS>":2}

    if not os.path.exists(csv_programs):
        return [], vocab, {}

    with open(csv_programs, 'r') as f:
        reader = csv.DictReader(f)
        for r in reader:
            key = r['ProgramNo'] + "_" + r.get('ProjectID', '1')
            cmd = r['Instruction']
            stn = int(r['InstructionValue']) if r['InstructionValue'] else 0
            if cmd not in vocab:
                vocab[cmd] = len(vocab)
            sequences[key].append((vocab[cmd], min(stn, MAX_STATIONS-1)))

    adj = defaultdict(list)
    if os.path.exists(csv_zones):
        with open(csv_zones, 'r') as f:
            reader = csv.DictReader(f)
            for r in reader:
                try:
                    s1, s2 = int(r['Row1StationNo']), int(r['Row2StationNo'])
                    if s1 < MAX_STATIONS and s2 < MAX_STATIONS:
                        adj[s1].append(s2)
                        adj[s2].append(s1)
                except: continue

    return list(sequences.values()), vocab, adj

def root_cause_analysis(error_msg, context):
    print("\n" + "!"*30)
    print("ROOT CAUSE ANALYSIS")
    print(f"Error: {error_msg}")
    tanks = context.get('tanks', [])
    config = context.get('config', {})
    wid = context.get('wagon_id')
    if wid and wid not in config:
        print(f"- CAUSE: Wagon '{wid}' is missing from config.")
    elif wid:
        w = config[wid]
        for t in tanks:
            s = int(t.get('station_no', 0))
            if s < w['min_stn'] or s > w['max_stn']:
                print(f"- CAUSE: Station {s} is out of range ({w['min_stn']}-{w['max_stn']}) for {wid}.")
    if len(tanks) < 2:
        print("- CAUSE: Insufficient tank data (min 2 required).")
    print("!"*30 + "\n")

# ==========================================
# PHYSICS ENGINE
# ==========================================
def calculate_time_value(dist_total, s_speed, f_speed, sf_speed, slow_dist=500.0):
    """
    Accurate 3-speed phased timing.
    Logic: 
    1. Always move at Slow Speed for the last 'slow_dist' mm.
    2. If distance > slow_dist, move the rest at Fast Speed.
    3. If distance is very large, move at Superfast Speed for the middle section.
    Refined Formula below for simplicity while maintaining accuracy:
    """
    try:
        # Conversion: m/min * (1 min / 60 sec) * (1000 mm / 1 m) = 16.666
        to_mms = 16.666
        v_s = s_speed * to_mms if s_speed > 0 else 1.0
        v_f = f_speed * to_mms if f_speed > 0 else 1.0
        v_sf = sf_speed * to_mms if sf_speed > 0 else 1.0

        if dist_total <= slow_dist:
            return dist_total / v_s
        
        # We split the remaining distance between Fast and Superfast
        # Assumption: 30% of remaining is Fast (accel/decel), 70% is Superfast if long enough
        rem_dist = dist_total - slow_dist
        if rem_dist > 2000: # Superfast threshold
            d_sf = rem_dist * 0.7
            d_f = rem_dist * 0.3
            return (d_sf / v_sf) + (d_f / v_f) + (slow_dist / v_s)
        else:
            return (rem_dist / v_f) + (slow_dist / v_s)
    except: return 0

# ==========================================
# MODEL DEFINITIONS
# ==========================================
class TransformerEncoder(nn.Module):
    def __init__(self, n_cmds):
        super().__init__()
        self.cmd_emb = nn.Embedding(n_cmds, D_MODEL)
        self.stn_emb = nn.Embedding(MAX_STATIONS, D_MODEL)
        self.pos_emb = nn.Embedding(MAX_STEPS, D_MODEL)
        layer = nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, N_LAYERS)

    def forward(self, c, s):
        pos = torch.arange(c.size(1), device=c.device)
        x = self.cmd_emb(c) + self.stn_emb(s) + self.pos_emb(pos)
        return self.encoder(x)[:, -1]

class ActorCritic(nn.Module):
    def __init__(self, n_cmds):
        super().__init__()
        self.encoder = TransformerEncoder(n_cmds)
        self.actor_cmd = nn.Linear(D_MODEL, n_cmds)
        self.actor_stn = nn.Linear(D_MODEL, MAX_STATIONS)
        self.critic = nn.Linear(D_MODEL, 1)

    def forward(self, c, s):
        x = self.encoder(c, s)
        return (
            torch.softmax(self.actor_cmd(x), -1),
            torch.softmax(self.actor_stn(x), -1),
            self.critic(x)
        )

class PPOAgent:
    def __init__(self, vocab):
        self.policy = ActorCritic(len(vocab)).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR, weight_decay=1e-4)
        self.mse = nn.MSELoss()

    def update(self, memory):
        c_state = torch.stack(memory.c_states).to(device).detach()
        s_state = torch.stack(memory.s_states).to(device).detach()
        a_cmd = torch.stack(memory.a_cmds).to(device).detach().squeeze()
        a_stn = torch.stack(memory.a_stns).to(device).detach().squeeze()
        old_logp = torch.stack(memory.logprobs).to(device).detach().squeeze()

        rewards = []
        G = 0
        for r, d in zip(reversed(memory.rewards), reversed(memory.dones)):
            if d: G = 0
            G = r + GAMMA * G
            rewards.insert(0, G)
        
        rewards = torch.tensor(rewards).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for _ in range(UPDATE_ITERS):
            p_c, p_s, val = self.policy(c_state, s_state)
            val = val.squeeze()
            dc, ds = Categorical(p_c), Categorical(p_s)
            logp = dc.log_prob(a_cmd) + ds.log_prob(a_stn)
            ent = dc.entropy() + ds.entropy()
            
            adv = rewards - val.detach()
            ratio = torch.exp(logp - old_logp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1-EPS_CLIP, 1+EPS_CLIP) * adv

            loss = -torch.min(s1, s2) + 0.5 * self.mse(val, rewards) - ENTROPY_BETA * ent.mean()
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

class Memory:
    def __init__(self):
        self.c_states, self.s_states = [], []
        self.a_cmds, self.a_stns = [], []
        self.logprobs, self.rewards, self.dones = [], [], []
    def clear(self):
        self.__init__()

class WagonEnv:
    def __init__(self, adj):
        self.adj = adj
        self.reset()
    def reset(self):
        self.occ = [0]*MAX_STATIONS
        self.steps = 0
        return self.occ
    def step(self, stn):
        reward = 0
        self.steps += 1
        if stn >= MAX_STATIONS: return self.occ, -20, True
        if self.occ[stn]: reward -= 15
        elif any(self.occ[n] for n in self.adj.get(stn, [])): reward -= 8
        else: reward += 10
        self.occ[stn] = 1
        done = self.steps >= MAX_STEPS
        if done: reward += 50
        return self.occ, reward, done

# ==========================================
# SEQUENCE GENERATION
# ==========================================
def gap_analysis_sequence(tanks, wagon_config, wagon_id, censor_map):
    w = wagon_config[wagon_id]
    seq_data = [["Command", "Value", "TravelTime", "Lift/Lower", "Accumulated"]]
    curr_t = 0.0
    
    for i in range(len(tanks) - 1):
        curr, nxt = tanks[i], tanks[i+1]
        s_c, s_n = int(curr.get('station_no', 0)), int(nxt.get('station_no', 0))
        d_c, d_n = float(curr.get('distance_mm', 0)), float(nxt.get('distance_mm', 0))
        dist = abs(d_n - d_c)
        
        slow_d = censor_map.get((nxt.get('project_id', '1'), s_n), 500.0)
        trav_t = calculate_time_value(dist, w['s'], w['f'], w['sf'], slow_d)
        
        # GET FROM
        seq_data.append(["GET FROM", s_c, "0.00", f"{w['lift']:.2f}", f"{curr_t:.2f}"])
        curr_t += w['lift']
        
        # PUT ON
        seq_data.append(["PUT ON", s_n, f"{trav_t:.2f}", f"{w['lower']:.2f}", f"{curr_t:.2f}"])
        curr_t += trav_t + w['lower']
        
        # WAIT
        dip = float(nxt.get('dip_time_sec', 0))
        if dip > 0:
            seq_data.append(["WAIT", int(dip), "0.00", "0.00", f"{curr_t:.2f}"])
            curr_t += dip
            
    return seq_data

def evaluate_and_plot(agent, train_data, vocab):
    y_true, y_pred = [], []
    rev = {v: k for k, v in vocab.items()}
    print("\nEvaluating for Confusion Matrix...")
    for seq in train_data:
        c_q, s_q = [vocab["<SOS>"]], [0]
        for c_i, s_v in seq:
            c_p = torch.zeros((1, MAX_STEPS), dtype=torch.long).to(device)
            s_p = torch.zeros((1, MAX_STEPS), dtype=torch.long).to(device)
            c_p[0, :len(c_q)] = torch.tensor(c_q[-MAX_STEPS:])
            s_p[0, :len(s_q)] = torch.tensor(s_q[-MAX_STEPS:])
            with torch.no_grad():
                pc, ps, _ = agent.policy(c_p, s_p)
                y_true.append(rev.get(c_i, "UNK"))
                y_pred.append(rev.get(pc.argmax().item(), "UNK"))
            c_q.append(c_i)
            s_q.append(s_v)
            
    lbls = sorted(list(vocab.keys()))
    cm = confusion_matrix(y_true, y_pred, labels=lbls)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=lbls, yticklabels=lbls, cmap='Blues')
    plt.title('Confusion Matrix - up7 Accurate')
    plt.savefig('confusion_matrix_up7.png')
    plt.close()

# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["gen", "train"], default="train")
    parser.add_argument("--input", type=str, default=tanks_csv_default)
    parser.add_argument("--wagon", type=str, default="Wagon 1", help="Wagon Name from config")
    parser.add_argument("--episodes", type=int, default=200, help="Number of training episodes")
    args = parser.parse_args()

    train_data, vocab, adj = load_training_data()
    
    if args.mode == "train":
        agent = PPOAgent(vocab)
        env, mem = WagonEnv(adj), Memory()
        print(f"Training for {args.episodes} episodes...")
        for ep in range(1, args.episodes + 1):
            env.reset()
            c_q, s_q = [vocab["<SOS>"]], [0]
            ep_r = 0
            for _ in range(MAX_STEPS):
                cp = torch.zeros((1, MAX_STEPS), dtype=torch.long).to(device)
                sp = torch.zeros((1, MAX_STEPS), dtype=torch.long).to(device)
                cp[0, :len(c_q)] = torch.tensor(c_q[-MAX_STEPS:])
                sp[0, :len(s_q)] = torch.tensor(s_q[-MAX_STEPS:])
                
                with torch.no_grad():
                    pc, ps, _ = agent.policy(cp, sp)
                    
                # Action Masking
                mask = torch.ones_like(ps)
                for i, o in enumerate(env.occ):
                    if o: mask[0, i] = 0
                ps = (ps * mask) / (ps * mask).sum()
                
                dc, ds = Categorical(pc), Categorical(ps)
                ac, as_ = dc.sample(), ds.sample()
                _, r, d = env.step(as_.item())
                
                mem.c_states.append(cp.squeeze(0))
                mem.s_states.append(sp.squeeze(0))
                mem.a_cmds.append(ac); mem.a_stns.append(as_)
                mem.logprobs.append(dc.log_prob(ac) + ds.log_prob(as_))
                mem.rewards.append(r); mem.dones.append(d)
                
                c_q.append(ac.item()); s_q.append(as_.item())
                ep_r += r
                if d: break
            agent.update(mem); mem.clear()
            if ep % 50 == 0: print(f"Ep {ep} | Reward: {ep_r:.2f}")
        torch.save(agent.policy.state_dict(), "model_up7_accurate.pth")
        evaluate_and_plot(agent, train_data, vocab)
    
    else:
        if not os.path.exists(args.input):
            print(f"File {args.input} not found."); return
        config = load_wagon_config(wagon_config_csv)
        c_map = load_station_censor_data()
        wid = args.wagon
        if wid not in config:
            print(f"Wagon '{wid}' not found in configuration.")
            return
            
        try:
            with open(args.input, 'r') as f:
                tanks = list(csv.DictReader(f))
            seq = gap_analysis_sequence(tanks, config, wid, c_map)
            print(f"\n[GENERATED SEQUENCE FOR {wid}]")
            print(tabulate(seq[1:], headers=seq[0], tablefmt="grid"))
        except Exception as e:
            root_cause_analysis(str(e), {'tanks': tanks, 'config': config, 'wagon_id': wid})

if __name__ == "__main__":
    main()
