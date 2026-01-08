# Sequence Generation Tool: v7 Update Documentation (Hybrid AI + Verification)

This document explains the updates introduced in `v7.py`, which is now a hybrid tool combining advanced Machine Learning with robust station verification.

---

## 1. How v7.py Works (Hybrid Approach)

`v7.py` integrates the best features from previous versions to provide a comprehensive solution for sequence generation:

### 🧠 Machine Learning Engine
- **Architectures:** Re-integrated **Transformer, Graph Neural Networks (GNN), and PPO (Proximal Policy Optimization)**.
- **Training Mode:** Supports a dedicated `--mode train` to learn optimal sequences from historical data (`AutoSequencePrograms.csv` and `CrossTrolleyMaster.csv`).
- **Safety Conscious:** The GNN specifically models station occupancy and adjacency to ensure the agent learns safe movement patterns.

### 🚀 Station Verification Module
- **Integrity Check:** Automatically verifies if the input data (`tanks_csv.csv`) contains the correct number of unique stations based on the wagon's configuration.
- **Fail-Safe:** Alerts the user with explicit `[WARNING]` or `[SUCCESS]` messages before generation, ensuring that no sequence is generated from incomplete data without notice.

### 📊 Physics-Based Generation (Gap Analysis)
- **Deterministic Accuracy:** Even with ML capabilities, the primary generation engine uses the reliable **Gap Analysis** method.
- **Precise Timing:** Calculates travel times using Superfast, Fast, and Slow speeds, including lift/lower overheads and dip times.

---

## 2. Key Features & Usage

### Usage Modes
1. **Sequence Generation (Default):**
   ```powershell
   python e:\Internship\code\v7.py --mode gen --input e:\Internship\code\tanks_csv.csv
   ```
2. **AI Training:**
   ```powershell
   python e:\Internship\code\v7.py --mode train
   ```

### Comparison Table

| Feature | v6.py | v7.py (Current) |
| :--- | :--- | :--- |
| **Model Type** | RL Research | **Hybrid (ML Training + Physics Gen)** |
| **Verification** | Basic | **Full Station Count Verification** |
| **Architectures** | Transformer, GNN, PPO | **Transformer, GNN, PPO** |
| **Production Ready** | Selective | **Yes (Validation + AI + Physics)** |

---

## 3. How v7.py Better Supports Sequence Generation

1. **Verification Gate:** It ensures your input data is complete before processing, preventing "garbage in, garbage out" scenarios.
2. **AI-Ready:** While it uses physics for generation, the built-in ML models allow the system to be trained on complex historical patterns, making it ready for future autonomous optimizations.
3. **Safety First:** The integration of the GNN into the training loop means that future AI-generated sequences will be inherently aware of station adjacency and potential collisions.

---
*Updated documentation for the integrated Hybrid v7 version.*
