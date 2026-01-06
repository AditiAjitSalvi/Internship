"""
auto_seq_cli.py

Usage:
  python auto_seq_cli.py \
      --seq_csv training_sequences.csv \
      --tanks_csv tanks.csv \
      --project_id 1 \
      --start_station 6
"""
'''
import csv
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

# ========= CONFIG =========
N_EPOCHS = 50
LR = 1e-3
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
MAX_STATION = 128  # max station_no value you expect

# ========= VOCABS =========
CMD2ID = {"GetFrom": 0, "PutOn": 1, "WaitForSec": 2, "END": 3}
ID2CMD = {v: k for k, v in CMD2ID.items()}

CRIT2ID = {"Low": 0, "High": 1}
ID2CRIT = {v: k for k, v in CRIT2ID.items()}

# ========= GLOBAL TANK TABLE =========
# key: (project_id, station_no) -> {"critical":0/1, "dip":seconds}
TANKS = {}


# ========= MODEL =========
class AutoSequenceModel(nn.Module):
    def __init__(self, n_commands=4, n_stations=MAX_STATION,
                 d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS):
        super().__init__()
        self.cmd_emb = nn.Embedding(n_commands, d_model)
        self.stn_emb = nn.Embedding(n_stations, d_model)
        self.crit_emb = nn.Embedding(2, d_model)  # High/Low
        self.wait_lin = nn.Linear(1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out_cmd = nn.Linear(d_model, n_commands)
        self.out_stn = nn.Linear(d_model, n_stations)
        self.out_wait = nn.Linear(d_model, 1)

    def forward(self, cmd_ids, stn_ids, crit_ids, wait_vals):
        # cmd_ids, stn_ids, crit_ids, wait_vals: (seq_len, batch)
        e = (
            self.cmd_emb(cmd_ids)
            + self.stn_emb(stn_ids)
            + self.crit_emb(crit_ids)
            + self.wait_lin(wait_vals.unsqueeze(-1))
        )
        z = self.encoder(e)
        logits_cmd = self.out_cmd(z)
        logits_stn = self.out_stn(z)
        wait_pred = self.out_wait(z).squeeze(-1)
        return logits_cmd, logits_stn, wait_pred


# ========= LOAD TANK DETAILS =========
def load_tanks(tanks_csv_path: str):
    """
    tanks.csv columns example:
    project_id,station_no,process_name,critical_status,distance_mm,dip_time_sec
    """
    global TANKS
    TANKS = {}
    with open(tanks_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["project_id"])
            stn = int(row["station_no"])
            crit_str = row["critical_status"]
            crit = 1 if crit_str == "High" else 0
            dip = float(row.get("dip_time_sec", 0) or 0)
            TANKS[(pid, stn)] = {"critical": crit, "dip": dip}


# ========= LOAD TRAINING SEQUENCES =========
def load_sequences(seq_csv_path: str):
    """
    training_sequences.csv columns example:
    seq_id,project_id,step_no,command,station_no,wait_sec,critical_status
    (critical_status in this file is optional; we overwrite from tank table)
    """
    grouped = defaultdict(list)
    with open(seq_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = int(row["seq_id"])
            project_id = int(row["project_id"])
            step_no = int(row["step_no"])
            cmd = row["command"]
            stn = int(row["station_no"])
            wait = float(row["wait_sec"])

            key = (project_id, stn)
            if key in TANKS:
                crit_id = TANKS[key]["critical"]
            else:
                # fallback if tank not in table
                crit_id = 1 if row.get("critical_status", "High") == "High" else 0

            grouped[(seq_id, project_id)].append(
                (step_no, cmd, stn, wait, crit_id, project_id)
            )

    seqs = []
    for (seq_id, pid), steps in grouped.items():
        steps_sorted = sorted(steps, key=lambda x: x[0])
        # list of tuples: (cmd, stn, wait, crit_id, project_id)
        seqs.append(
            [(cmd, stn, wait, crit_id, pid) for (_, cmd, stn, wait, crit_id, pid) in steps_sorted]
        )
    return seqs


def build_tensors_from_seq(seq):
    """
    seq: list[(cmd, stn, wait, crit_id, project_id)]
    teacher forcing: input = seq[:-1], target = seq[1:]
    """
    cmd_ids = [CMD2ID[c] for c, _, _, _, _ in seq]
    stn_ids = [s for _, s, _, _, _ in seq]
    waits = [w for _, _, w, _, _ in seq]
    crit_ids = [crit for _, _, _, crit, _ in seq]

    inp_cmd = torch.tensor(cmd_ids[:-1]).long().unsqueeze(1)
    inp_stn = torch.tensor(stn_ids[:-1]).long().unsqueeze(1)
    inp_crit = torch.tensor(crit_ids[:-1]).long().unsqueeze(1)
    inp_wait = torch.tensor(waits[:-1]).float().unsqueeze(1)

    tgt_cmd = torch.tensor(cmd_ids[1:]).long().unsqueeze(1)
    tgt_stn = torch.tensor(stn_ids[1:]).long().unsqueeze(1)
    tgt_wait = torch.tensor(waits[1:]).float().unsqueeze(1)

    return inp_cmd, inp_stn, inp_crit, inp_wait, tgt_cmd, tgt_stn, tgt_wait


# ========= TRAIN MODEL =========
def train_model(seq_csv_path: str):
    seqs = load_sequences(seq_csv_path)
    model = AutoSequenceModel()
    opt = optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        model.train()
        for seq in seqs:
            (
                inp_cmd,
                inp_stn,
                inp_crit,
                inp_wait,
                tgt_cmd,
                tgt_stn,
                tgt_wait,
            ) = build_tensors_from_seq(seq)

            opt.zero_grad()
            lc, ls, lw = model(inp_cmd, inp_stn, inp_crit, inp_wait)

            loss_cmd = ce(lc.view(-1, lc.size(-1)), tgt_cmd.view(-1))
            loss_stn = ce(ls.view(-1, ls.size(-1)), tgt_stn.view(-1))
            loss_wait = mse(lw.view(-1), tgt_wait.view(-1))

            loss = loss_cmd + loss_stn + 0.01 * loss_wait
            loss.backward()
            opt.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{N_EPOCHS} loss={total_loss:.4f}")

    model.eval()
    return model


# ========= GENERATE SEQUENCE =========
def get_crit_for_station(project_id: int, station_no: int) -> int:
    key = (project_id, station_no)
    if key in TANKS:
        return TANKS[key]["critical"]
    return 0  # default Low


def get_dip_for_station(project_id: int, station_no: int) -> float:
    key = (project_id, station_no)
    if key in TANKS:
        return TANKS[key]["dip"]
    return 0.0


def generate_sequence(
    model,
    project_id: int,
    start_station: int,
    max_steps: int = 40,
):
    cmd_ids = [CMD2ID["GetFrom"]]
    stn_ids = [start_station]
    waits = [0.0]
    crit_ids = [get_crit_for_station(project_id, start_station)]

    for _ in range(max_steps):
        inp_cmd = torch.tensor(cmd_ids).long().unsqueeze(1)
        inp_stn = torch.tensor(stn_ids).long().unsqueeze(1)
        inp_crit = torch.tensor(crit_ids).long().unsqueeze(1)
        inp_wait = torch.tensor(waits).float().unsqueeze(1)

        with torch.no_grad():
            lc, ls, lw = model(inp_cmd, inp_stn, inp_crit, inp_wait)
            cmd_next = lc[-1, 0].argmax().item()
            stn_next = ls[-1, 0].argmax().item()
            wait_next = float(lw[-1, 0].item())

        if cmd_next == CMD2ID["END"]:
            break

        # if WaitForSec, use dip time from tank table as default
        if ID2CMD[cmd_next] == "WaitForSec":
            wait_next = get_dip_for_station(project_id, stn_ids[-1])

        cmd_ids.append(cmd_next)
        stn_ids.append(stn_next)
        waits.append(max(0.0, wait_next))
        crit_ids.append(get_crit_for_station(project_id, stn_next))

    # to human‑readable
    seq_text = []
    for c, s, w in zip(cmd_ids, stn_ids, waits):
        name = ID2CMD[c]
        if name == "WaitForSec":
            seq_text.append(f"Wait For Sec {int(round(w))}")
        else:
            seq_text.append(f"{name} {s}")
    return seq_text


# ========= CLI =========
def main():
    seq_csv = "E:\\Internship\\code\\seq_csv.csv"
    tanks_csv = "E:\\Internship\\code\\tanks_csv.csv"
    project_id = 1
    start_station = 6

    print(f"Loading tank details from: {tanks_csv}")
    load_tanks(tanks_csv)

    print(f"Training from sequences: {seq_csv}")
    model = train_model(seq_csv)

    print("\n=== GENERATED SEQUENCE ===")
    seq = generate_sequence(
        model,
        project_id=project_id,
        start_station=start_station,
    )
    for i, step in enumerate(seq, start=1):
        print(f"{i:02d}: {step}")
'''

"""
auto_seq_cli.py
"""

import csv
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim

print("SCRIPT STARTED")  # <<< CHANGE (debug)

# ========= CONFIG =========
N_EPOCHS = 50
LR = 1e-3
D_MODEL = 64
N_LAYERS = 2
N_HEADS = 4
MAX_STATION = 128

# ========= VOCABS =========
CMD2ID = {"GetFrom": 0, "PutOn": 1, "WaitForSec": 2, "END": 3}
ID2CMD = {v: k for k, v in CMD2ID.items()}

CRIT2ID = {"Low": 0, "High": 1}
ID2CRIT = {v: k for k, v in CRIT2ID.items()}

# ========= GLOBAL TANK TABLE =========
TANKS = {}


# ========= MODEL =========
class AutoSequenceModel(nn.Module):
    def __init__(self, n_commands=4, n_stations=MAX_STATION,
                 d_model=D_MODEL, n_layers=N_LAYERS, n_heads=N_HEADS):
        super().__init__()
        self.cmd_emb = nn.Embedding(n_commands, d_model)
        self.stn_emb = nn.Embedding(n_stations, d_model)
        self.crit_emb = nn.Embedding(2, d_model)
        self.wait_lin = nn.Linear(1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=128
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.out_cmd = nn.Linear(d_model, n_commands)
        self.out_stn = nn.Linear(d_model, n_stations)
        self.out_wait = nn.Linear(d_model, 1)

    def forward(self, cmd_ids, stn_ids, crit_ids, wait_vals):
        e = (
            self.cmd_emb(cmd_ids)
            + self.stn_emb(stn_ids)
            + self.crit_emb(crit_ids)
            + self.wait_lin(wait_vals.unsqueeze(-1))
        )
        z = self.encoder(e)
        return self.out_cmd(z), self.out_stn(z), self.out_wait(z).squeeze(-1)


# ========= LOAD TANK DETAILS =========
def load_tanks(tanks_csv_path: str):
    global TANKS
    TANKS = {}

    with open(tanks_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["project_id"])
            stn = int(row["station_no"])
            crit = 1 if row["critical_status"] == "High" else 0
            dip = float(row.get("dip_time_sec", 0) or 0)
            TANKS[(pid, stn)] = {"critical": crit, "dip": dip}

    print("DEBUG: tanks loaded =", len(TANKS))  # <<< CHANGE


# ========= LOAD TRAINING SEQUENCES =========
def load_sequences(seq_csv_path: str):
    grouped = defaultdict(list)
    skipped = 0

    with open(seq_csv_path, newline="") as f:
        reader = csv.DictReader(f)

        for row_no, row in enumerate(reader, start=2):
            try:
                if not row["seq_id"].strip():
                    raise ValueError("empty seq_id")

                seq_id = int(row["seq_id"])
                project_id = int(row["project_id"])
                step_no = int(row["step_no"])
                cmd = row["command"].strip()
                stn = int(row["station_no"])
                wait = float(row["wait_sec"] or 0)

            except Exception as e:
                skipped += 1
                print(f"WARNING: skipping row {row_no}: {e}")
                continue

            key = (project_id, stn)
            crit_id = TANKS[key]["critical"] if key in TANKS else 0

            grouped[(seq_id, project_id)].append(
                (step_no, cmd, stn, wait, crit_id, project_id)
            )

    print(f"DEBUG: skipped rows = {skipped}")

    seqs = []
    for (_, _), steps in grouped.items():
        steps = sorted(steps, key=lambda x: x[0])
        if len(steps) >= 2:  # ensure usable sequence
            seqs.append([(c, s, w, cr, pid) for (_, c, s, w, cr, pid) in steps])

    return seqs

"""def load_sequences(seq_csv_path: str):
    grouped = defaultdict(list)

    with open(seq_csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            seq_id = int(row["seq_id"])
            project_id = int(row["project_id"])
            step_no = int(row["step_no"])
            cmd = row["command"]
            stn = int(row["station_no"])
            wait = float(row["wait_sec"])

            key = (project_id, stn)
            crit_id = TANKS[key]["critical"] if key in TANKS else 0

            grouped[(seq_id, project_id)].append(
                (step_no, cmd, stn, wait, crit_id, project_id)
            )

    seqs = []
    for (_, _), steps in grouped.items():
        steps = sorted(steps, key=lambda x: x[0])
        seqs.append([(c, s, w, cr, pid) for (_, c, s, w, cr, pid) in steps])

    return seqs


def build_tensors_from_seq(seq):
    cmd_ids = [CMD2ID[c] for c, _, _, _, _ in seq]
    stn_ids = [s for _, s, _, _, _ in seq]
    waits = [w for _, _, w, _, _ in seq]
    crit_ids = [cr for _, _, _, cr, _ in seq]

    return (
        torch.tensor(cmd_ids[:-1]).long().unsqueeze(1),
        torch.tensor(stn_ids[:-1]).long().unsqueeze(1),
        torch.tensor(crit_ids[:-1]).long().unsqueeze(1),
        torch.tensor(waits[:-1]).float().unsqueeze(1),
        torch.tensor(cmd_ids[1:]).long().unsqueeze(1),
        torch.tensor(stn_ids[1:]).long().unsqueeze(1),
        torch.tensor(waits[1:]).float().unsqueeze(1),
    )
"""

# ========= TRAIN MODEL =========
def train_model(seq_csv_path: str):
    seqs = load_sequences(seq_csv_path)

    print("DEBUG: sequences loaded =", len(seqs))  # <<< CHANGE
    if len(seqs) == 0:
        raise RuntimeError("No training data found")

    model = AutoSequenceModel()
    opt = optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    for epoch in range(N_EPOCHS):
        total_loss = 0.0
        used = 0
        model.train()

        for seq in seqs:
            if len(seq) < 2:
                continue  # <<< CHANGE (critical)

            used += 1

            inp_cmd, inp_stn, inp_crit, inp_wait, tgt_cmd, tgt_stn, tgt_wait = (
                build_tensors_from_seq(seq)
            )

            opt.zero_grad()
            lc, ls, lw = model(inp_cmd, inp_stn, inp_crit, inp_wait)

            loss = (
                ce(lc.view(-1, lc.size(-1)), tgt_cmd.view(-1))
                + ce(ls.view(-1, ls.size(-1)), tgt_stn.view(-1))
                + 0.01 * mse(lw.view(-1), tgt_wait.view(-1))
            )

            loss.backward()
            opt.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{N_EPOCHS} | used={used} | loss={total_loss:.4f}")  # <<< CHANGE

    model.eval()
    return model


# ========= GENERATE SEQUENCE =========
def get_crit_for_station(project_id, station_no):
    return TANKS.get((project_id, station_no), {}).get("critical", 0)


def get_dip_for_station(project_id, station_no):
    return TANKS.get((project_id, station_no), {}).get("dip", 0.0)


def generate_sequence(model, project_id, start_station, max_steps=40):
    cmd_ids = [CMD2ID["GetFrom"]]
    stn_ids = [start_station]
    waits = [0.0]
    crit_ids = [get_crit_for_station(project_id, start_station)]

    for _ in range(max_steps):
        inp_cmd = torch.tensor(cmd_ids).long().unsqueeze(1)
        inp_stn = torch.tensor(stn_ids).long().unsqueeze(1)
        inp_crit = torch.tensor(crit_ids).long().unsqueeze(1)
        inp_wait = torch.tensor(waits).float().unsqueeze(1)

        with torch.no_grad():
            lc, ls, lw = model(inp_cmd, inp_stn, inp_crit, inp_wait)
            cmd_next = lc[-1, 0].argmax().item()
            stn_next = ls[-1, 0].argmax().item()
            wait_next = lw[-1, 0].item()

        if cmd_next == CMD2ID["END"] and len(cmd_ids) > 3:  # <<< CHANGE
            break

        if ID2CMD[cmd_next] == "WaitForSec":
            wait_next = get_dip_for_station(project_id, stn_ids[-1])

        cmd_ids.append(cmd_next)
        stn_ids.append(stn_next)
        waits.append(max(0.0, wait_next))
        crit_ids.append(get_crit_for_station(project_id, stn_next))

    seq_text = []
    for c, s, w in zip(cmd_ids, stn_ids, waits):
        if ID2CMD[c] == "WaitForSec":
            seq_text.append(f"Wait For Sec {int(round(w))}")
        else:
            seq_text.append(f"{ID2CMD[c]} {s}")
    return seq_text


# ========= MAIN =========
def main():
    print("INSIDE MAIN")  # <<< CHANGE

    seq_csv = r"E:\Internship\code\seq_csv.csv"
    tanks_csv = r"E:\Internship\code\tanks_csv.csv"
    project_id = 1
    start_station = 6

    load_tanks(tanks_csv)
    model = train_model(seq_csv)

    print("\n=== GENERATED SEQUENCE ===")
    seq = generate_sequence(model, project_id, start_station)
    print("Generated steps:", len(seq))  # <<< CHANGE

    for i, step in enumerate(seq, 1):
        print(f"{i:02d}: {step}")


if __name__ == "__main__":
    main()
