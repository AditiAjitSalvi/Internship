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


# ==========================================
# CONFIGURATION
# ==========================================
csv_programs = r"e:\Internship\WayTime-dB.mdb\AutoSequencePrograms.csv"
csv_zones = r"e:\Internship\WayTime-dB.mdb\CrossTrolleyMaster.csv"
csv_train_seq = r"e:\Internship\Sequnce-trainfinal.csv"
station_master_csv = r"e:\Internship\WayTime-dB.mdb\StationMaster.csv"
wagon_master_csv = r"e:\Internship\WayTime-dB.mdb\WagonMaster.csv"
wagon_config_csv = r"e:\Internship\code\wagon_config.csv"

# Hyperparameters
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# DATA LOADING
# ==========================================
def load_wagon_speeds(wagon_id=None):
    """
    Loads wagon speeds from WagonMaster.csv.
    If wagon_id is provided, returns speeds for that wagon.
    Otherwise returns a default or dictionary.
    """
    speeds = {}
    try:
        with open(wagon_master_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Key can be ID or WagonNumber
                # Using ID as key
                try:
                    # speeds: Superfast, Fast, Slow
                    sf = float(row['SuperfastSpeed'])
                    f_spd = float(row['FastSpeed'])
                    s_spd = float(row['SlowSpeed'])
                    
                    # Assuming WagonNumber or ID is the identifier
                    # User said "i give speeds of wagone in input line", which implies manual override,
                    # but "use @[...] as referance".
                    # We'll store by WagonNumber (e.g., "Wagon 1")
                    w_num = row['WagonNumber'].strip()
                    speeds[w_num] = (sf, f_spd, s_spd)
                    
                    # Also store by simple ID if needed
                    speeds[row['ID']] = (sf, f_spd, s_spd)
                except:
                    continue
    except FileNotFoundError:
        print("WagonMaster.csv not found.")
        return None
        
    if wagon_id and wagon_id in speeds:
        return speeds[wagon_id]
        
    return speeds

def load_wagon_config(config_path=wagon_config_csv):
    """
    Loads wagon configuration from the new CSV format.
    Fields: Wagon Number, Superfast Speed, Fast Speed, Slow Speed, Lift Time, 
            Lift Stroke Speed, Lower Time, Minimum Station No, Maximum Station No, 
            Basic Position, No Of Station To Stop
    """
    config = {}
    if not os.path.exists(config_path):
        return config
    try:
        with open(config_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                w_num = row['Wagon Number'].strip()
                config[w_num] = {
                    'sf': float(row['Superfast Speed']),
                    'f': float(row['Fast Speed']),
                    's': float(row['Slow Speed']),
                    'lift_time': float(row['Lift Time']),
                    'lift_stroke': float(row['Lift Stroke Speed']),
                    'lower_time': float(row['Lower Time']),
                    'min_stn': int(row['Minimum Station No']),
                    'max_stn': int(row['Maximum Station No']),
                    'basic_pos': int(row['Basic Position']),
                    'stop_count': int(row['No Of Station To Stop'])
                }
    except Exception as e:
        print(f"Error loading wagon config: {e}")
    return config

def root_cause_analysis(error_msg, context):
    """
    Performs Root Cause Analysis when an error occurs.
    """
    print("\n[ROOT CAUSE ANALYSIS]")
    print(f"Error Detected: {error_msg}")
    print("Diagnostics:")
    
    tanks = context.get('tanks', [])
    config = context.get('config', {})
    wagon_id = context.get('wagon_id')
    
    if wagon_id and wagon_id not in config:
        print(f"- CAUSE: Wagon ID '{wagon_id}' not found in configuration.")
        return

    w_conf = config.get(wagon_id)
    for tank in tanks:
        stn = int(tank.get('station_no', 0))
        if w_conf and (stn < w_conf['min_stn'] or stn > w_conf['max_stn']):
            print(f"- CAUSE: Station {stn} is outside operational range ({w_conf['min_stn']}-{w_conf['max_stn']}) for {wagon_id}.")
    
    if len(tanks) < 2:
        print("- CAUSE: Insufficient station data provided (need at least 2).")
    
    print("-----------------------\n")

def load_station_censor_data():
    """
    Loads CensorDistance from StationMaster.csv.
    Returns dict: (project_id_str, station_no_int) -> censor_distance_float
    Also tries (project_id_int, station_no_int)
    """
    censor_map = {}
    try:
        with open(station_master_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pid_str = row['ProjectID']
                    stn = int(row['StationNumber'])
                    cdist = row.get('CensorDistance', '')
                    if cdist and cdist.strip():
                        val = float(cdist)
                        censor_map[(pid_str, stn)] = val
                        # Try to handle numeric project ids
                        if pid_str.isdigit():
                            censor_map[(int(pid_str), stn)] = val
                except:
                    continue
    except:
        pass
    return censor_map

def calculate_time_value(distance1, distance2, distance3, sfspeed, fspeed, sspeed):
    """
    Calculates time value based on the formula:
    CalculatedTimevalue1 = ((distance1) / (sfspeed * 16.66)) + ((distance2) / (fspeed * 16.66)) + ((distance3) / (sspeed * 16.66)))
    """
    try:
        # Avoid division by zero
        sfs = sfspeed * 16.66 if sfspeed > 0 else 1.0
        fs = fspeed * 16.66 if fspeed > 0 else 1.0
        ss = sspeed * 16.66 if sspeed > 0 else 1.0
        
        t1 = (distance1) / sfs
        t2 = (distance2) / fs
        t3 = (distance3) / ss
        return t1 + t2 + t3
    except:
        return 0

def load_data():
    print("Loading data...")
    sequences = defaultdict(list)
    vocab_cmd = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
    
    with open(csv_programs, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seq_id = row['ProgramNo'] + "_" + row['ProjectID']
                cmd = row['Instruction']
                stn = int(row['InstructionValue']) if row['InstructionValue'] else 0
                if cmd not in vocab_cmd:
                    vocab_cmd[cmd] = len(vocab_cmd)
                sequences[seq_id].append((vocab_cmd[cmd], stn))
            except Exception:
                continue

    train_data = []
    for seq in sequences.values():
        train_data.append(seq)
        
    print(f"Loaded {len(train_data)} sequences.")
    
    adj_list = defaultdict(list)
    with open(csv_zones, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                s1 = int(row['Row1StationNo'])
                s2 = int(row['Row2StationNo'])
                adj_list[s1].append(s2)
                adj_list[s2].append(s1)
            except Exception:
                continue
                
    return train_data, adj_list, vocab_cmd

# ==========================================
# MODELS (TRANSFORMER + GNN + PPO)
# ==========================================
# (Condensed version of v4.py models for brevity in v5.py)
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
        t_logits_cmd, t_logits_stn = self.transformer(cmd_seq, stn_seq)
        seq_emb = torch.cat([t_logits_cmd[:, -1, :], t_logits_stn[:, -1, :]], dim=-1)
        
        # Proper context extraction (simplified for this script)
        seq_len = cmd_seq.size(1)
        pos = torch.arange(seq_len, device=cmd_seq.device).unsqueeze(0).expand(cmd_seq.size(0), -1)
        x_seq = self.transformer.cmd_emb(cmd_seq) + self.transformer.stn_emb(stn_seq) + self.transformer.pos_emb(pos)
        x_seq = self.transformer.transformer(x_seq)
        ctx_seq = x_seq[:, -1, :] 

        x_gnn = self.gnn(station_states, adj_matrix)
        ctx_safe, _ = torch.max(x_gnn, dim=1)
        
        state = torch.cat([ctx_seq, ctx_safe], dim=-1)
        
        return torch.softmax(self.actor_cmd(state), dim=-1), torch.softmax(self.actor_stn(state), dim=-1), self.critic(state)

class PPOAgent:
    def __init__(self, vocab_cmd, n_stations, adj_list):
        self.vocab_cmd = vocab_cmd
        self.n_stations = n_stations
        self.adj_matrix = self._build_adj_matrix(adj_list, n_stations)
        self.policy = ActorCritic(len(vocab_cmd), n_stations).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)
        
    def _build_adj_matrix(self, adj_list, n):
        mat = torch.zeros((n, n), device=device)
        for u, neighbors in adj_list.items():
            if u < n:
                for v in neighbors:
                    if v < n:
                        mat[u, v] = 1.0
        return mat

    def select_action(self, cmd_seq, stn_seq, station_states):
        cmd_seq = torch.tensor([cmd_seq], dtype=torch.long).to(device)
        stn_seq = torch.tensor([stn_seq], dtype=torch.long).to(device)
        station_states = torch.tensor([station_states], dtype=torch.long).to(device)
        with torch.no_grad():
            probs_cmd, probs_stn, val = self.policy(cmd_seq, stn_seq, station_states, self.adj_matrix)
        dist_cmd = Categorical(probs_cmd)
        dist_stn = Categorical(probs_stn)
        action_cmd = dist_cmd.sample()
        action_stn = dist_stn.sample()
        return action_cmd.item(), action_stn.item(), dist_cmd.log_prob(action_cmd) + dist_stn.log_prob(action_stn), val

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
# SEQUENCE GENERATION (MODIFIED)
# ==========================================

def gap_analysis_sequence(tanks, wagon_config, wagon_id, censor_map=None, default_censor=5.0):
    """
    Generates sequence using Gap Analysis approach.
    Accounts for Lift/Lower times and operational gaps.
    """
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
        
        # Validate range
        if s_curr < w['min_stn'] or s_curr > w['max_stn'] or s_next < w['min_stn'] or s_next > w['max_stn']:
            raise ValueError(f"Station range violation: {s_curr}->{s_next} for {wagon_id}")

        dist_curr = float(curr_tank.get('distance_mm', 0))
        dist_next = float(next_tank.get('distance_mm', 0))
        dist_total = abs(dist_next - dist_curr)
        
        # Gap Analysis: Calculate leg components
        distance3 = censor_map.get((curr_tank.get('project_id', '1'), s_next), default_censor)
        distance2 = max(0, dist_total - distance3)
        distance1 = dist_total # Total distance for SF speed component in original formula
        
        travel_time = calculate_time_value(distance1, distance2, distance3, w['sf'], w['f'], w['s'])
        
        # Add Lift/Lower overheads
        total_leg_time = travel_time + w['lift_time'] + w['lower_time']
        
        # Get From (Lift)
        sequence_data.append(["GET FROM", s_curr, f"{w['lift_time']:.2f}", f"{accumulated_time:.2f}"])
        accumulated_time += w['lift_time']
        
        # Put On (Travel + Lower)
        sequence_data.append(["PUT ON", s_next, f"{(travel_time + w['lower_time']):.2f}", f"{accumulated_time:.2f}"])
        accumulated_time += travel_time + w['lower_time']
        
        # Wait (Dip Time)
        dip_time = float(next_tank.get('dip_time_sec', 0))
        if dip_time > 0:
            sequence_data.append(["WAIT", int(dip_time), "0.00", f"{accumulated_time:.2f}"])
            accumulated_time += dip_time
            
    return sequence_data

def generate_sequence_from_tanks(csv_path, speeds_input=None):
    print(f"Generating sequence from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: File {csv_path} not found.")
        return

    # Load speeds from CSV first as reference
    wagon_speeds_map = load_wagon_speeds() or {}
    wagon_config = load_wagon_config()
    censor_map = load_station_censor_data()
    
    wagon_id = speeds_input if speeds_input else "Wagon 1"
    
    print(f"Using Wagon Config for: {wagon_id}")

    try:
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            tanks = list(reader)
            
        print(f"Found {len(tanks)} stations/processes.")
        
        # Use Gap Analysis Method
        seq_data = gap_analysis_sequence(tanks, wagon_config, wagon_id, censor_map=censor_map)
        
        # Table Display
        try:
            
            print("\n" + tabulate(seq_data[1:], headers=seq_data[0], tablefmt="grid"))
        except ImportError:
            # Fallback manual table
            print_simple_table(seq_data[0], seq_data[1:])
             
        print("\n==========================\n")
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        # Use Root Cause Analysis Method on Error
        wagon_config = load_wagon_config() 
        tanks = []
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                tanks = list(reader)
        except:
            pass
            
        root_cause_analysis(str(e), {
            'tanks': tanks,
            'config': wagon_config,
            'wagon_id': speeds_input if speeds_input else "Wagon 1"
        })

def print_simple_table(headers, rows):
    # Determine column widths
    widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(val)))
    
    # Create separator and format
    separator = "+-" + "-+-".join(["-" * w for w in widths]) + "-+"
    row_fmt = "| " + " | ".join([f"{{:<{w}}}" for w in widths]) + " |"
    
    print(separator)
    print(row_fmt.format(*headers))
    print(separator.replace("-", "="))
    for row in rows:
        print(row_fmt.format(*[str(r) for r in row]))
    print(separator)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    global wagon_config_csv
    parser = argparse.ArgumentParser(description="Train RL Agent or Generate Sequence")
    parser.add_argument("--input", type=str, help="Path to tanks/input CSV for sequence generation")
    parser.add_argument("--speeds", type=str, help="Wagon Number from config (e.g., 'Wagon 1')", default="Wagon 1")
    parser.add_argument("--config", type=str, help="Path to wagon config CSV", default=wagon_config_csv)
    args = parser.parse_args()
    
    if args.input:
        # Update global config path if provided
        wagon_config_csv = args.config
        generate_sequence_from_tanks(args.input, args.speeds)
        return

    # Default Training Mode
    train_data, adj_list, vocab_cmd = load_data()
    agent = PPOAgent(vocab_cmd, MAX_STATIONS, adj_list)
    env = WagonEnv(adj_list, MAX_STATIONS)
    print("\nStarting Training Loop...")
    for episode in range(1, 11): 
        state = env.reset()
        curr_cmd_seq = [vocab_cmd["<SOS>"]]
        curr_stn_seq = [0]
        ep_reward = 0
        for t in range(MAX_STEPS):
            a_cmd, a_stn, _, _ = agent.select_action(curr_cmd_seq, curr_stn_seq, state)
            next_state, reward, done = env.step(a_cmd, a_stn)
            curr_cmd_seq.append(a_cmd)
            curr_stn_seq.append(a_stn)
            state = next_state
            ep_reward += reward
            if done: break
        print(f"Episode {episode}: Total Reward = {ep_reward}")
    print("Training Finished.")
    torch.save(agent.policy.state_dict(), "model_v4.pth")

if __name__ == "__main__":
    main()
