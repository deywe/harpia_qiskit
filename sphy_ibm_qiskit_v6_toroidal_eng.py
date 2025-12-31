# -*- coding: utf-8 -*-
# ───────────────────────────────────────────────────────────────
# File: sphy_ibm_qiskit_v6_toroidal_eng.py (VISUALIZATION ENABLED)
# Purpose: Quantum Tunneling Simulation (2x2 Toroidal Grid) + SPHY Meissner
#          (REALIST MODE: INCLUDES BARRIER, NOISE, SPHY METRICS, AND 2D/3D PLOTS)
# ───────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ⭐ VISUALIZATION FIX: Change the backend to interactive
import matplotlib
# Use an interactive backend like 'TkAgg' (remove if using your system's default)
# If this fails, try removing the whole line or using 'Qt5Agg'
matplotlib.use('TkAgg') 
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# Qiskit Imports
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator 

from qiskit.result import marginal_counts

import numpy as np
import matplotlib.pyplot as plt # Import here, after backend configuration
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import csv
from datetime import datetime
from tqdm import tqdm
import hashlib
import os, random, sys
from multiprocessing import Pool, Manager
from scipy.interpolate import interp1d

# ASSUMPTION: 'meissner_core.py' is available The Gravito-Quantum Meissinser Correction AI.
try:
    from meissner_core import meissner_correction_step
except ImportError:
    print("❌ Critical Error: The file 'meissner_core.py' was not found.")
    print("Ensure it is in the same directory and is accessible.")
    sys.exit(1)

# 🔧 Configurations
GRID_SIZE = 2
NUM_QUBITS = GRID_SIZE * GRID_SIZE
TARGET_QUBIT_INDEX = 0 
LOG_DIR = "logs_sphy_toroidal_3d_realist"
os.makedirs(LOG_DIR, exist_ok=True)
IDEAL_ACCEPT_STATES = ['0000', '1111']

# 🚀 User Input
def get_user_parameters():
    try:
        total_frames = int(input("🔁 Enter Total Attempts (Frames) to simulate: "))
        barrier_input = float(input("🚧 Quantum Barrier Strength (0.0 - 1.0): "))
        if not 0.0 <= barrier_input <= 1.0:
            raise ValueError("Barrier Strength must be between 0.0 and 1.0.")
            
        rz_angle = barrier_input * np.pi
        
        return NUM_QUBITS, total_frames, rz_angle
    except ValueError as ve:
        print(f"❌ Error: Invalid input. {ve}")
        sys.exit(1)
    except Exception:
        print("❌ Error: Please enter valid numbers.")
        sys.exit(1)

# ⚙️ Quantum Circuit Construction (Corrected GHZ + BARRIER)
def create_toroidal_circuit(num_qubits, rz_angle, noise_phase_shift=0.0):
    """
    Constructs the GHZ circuit with RZ barrier and noise.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # 1. State Preparation: Hadamard on Qubit 0
    qc.h(0)
    
    # 2. GHZ Couplings: Standard CNOT sequence
    for i in range(1, num_qubits):
        qc.cx(0, i) 
    
    # 3. RZ Gate for barrier strength (rz_angle) + noise
    qc.rz(rz_angle + noise_phase_shift, TARGET_QUBIT_INDEX)
    
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


# 🎲 Single Frame Simulation
def simulate_frame(frame_data):
    frame, num_qubits, total_frames, noise_prob, sphy_coherence, rz_angle = frame_data
    current_timestamp = datetime.utcnow().isoformat()
    random.seed(os.getpid() * frame)

    try:
        # NOISE: Phase noise to simulate decoherence
        noise_phase_shift = random.uniform(-0.5, 0.5) if random.random() < noise_prob else 0.0
        
        qc = create_toroidal_circuit(num_qubits, rz_angle, noise_phase_shift)
        
        simulator = AerSimulator()
        job = simulator.run(qc, shots=1)
        result_counts = job.result().get_counts()
        result_bin = list(result_counts.keys())[0]

    except Exception as e:
        return None, None, f"[Frame Error {frame}]: {e}"

    accepted_classical = result_bin in IDEAL_ACCEPT_STATES
    result_raw = 1 if accepted_classical else 0
    ideal_state = 1

    # Meissner Core Inputs
    H = random.uniform(0.95, 1.0)
    S = random.uniform(0.95, 1.0)
    C = sphy_coherence / 100.0
    I = abs(H - S)
    psi_state = [3.0, 3.0, 1.2, 1.2, 0.6, 0.5]

    try:
        boost, _, psi_state = meissner_correction_step(H, S, C, I, frame, psi_state)
    except Exception as e:
        return None, None, f"[Meissner Error Frame {frame}]: {e}"

    delta = boost * 0.7
    new_coherence = min(100, sphy_coherence + delta)
    
    # Qubit phase: 1 if TARGET_QUBIT_INDEX (Q0) was measured as '1', -1 if '0'
    target_qubit_state = int(result_bin[- (TARGET_QUBIT_INDEX + 1)]) 
    target_qubit_phase = 1 if target_qubit_state == 1 else -1

    # Acceptance: Tunneling success AND active SPHY correction (Heuristic)
    accepted = result_raw == ideal_state and delta > 0

    sha_str = f"{frame}:{result_raw}:{H:.4f}:{S:.4f}:{C:.4f}:{I:.4f}:{boost:.4f}:{new_coherence:.4f}:{current_timestamp}"
    sha256 = hashlib.sha256(sha_str.encode('utf-8')).hexdigest()

    # Log with SPHY metrics (ORDER MUST MATCH CSV HEADER)
    log_entry = [
        frame, result_raw,
        target_qubit_phase, 
        round(H, 4), round(S, 4), round(C, 4), round(I, 4),
        round(boost, 4), round(new_coherence, 4), 
        "✅" if accepted else "❌",              
        sha256,                                  
        current_timestamp
    ]
    return log_entry, new_coherence, None


# 📊 SPHY Coherence Plot (with noise)
def plot_2d_stability(sphy_evolution_list, fig_filename):
    """
    Generates the 2D stability graphs showing SPHY coherence and variance.
    """
    if not sphy_evolution_list:
        print("❌ Empty SPHY data.")
        return
    sphy_arr = np.array(sphy_evolution_list)
    time_points = np.linspace(0, 1, len(sphy_arr))
    n_signals = 2
    signals = [interp1d(time_points, np.roll(sphy_arr, i), kind='cubic') for i in range(n_signals)]
    new_time = np.linspace(0, 1, 2000)
    data = [s(new_time) + np.random.normal(0, 0.15, len(new_time)) for s in signals]
    weights = np.linspace(1, 1.5, n_signals)
    stability_avg = np.average(data, axis=0, weights=weights)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    ax1.set_title("SPHY Coherence Evolution (with Redundancy)")
    for d in data:
        ax1.plot(new_time, d, alpha=0.3, color='blue')
    ax1.plot(new_time, stability_avg, 'k--', label="Weighted Average")
    ax1.set_xlabel("Normalized Time")
    ax1.set_ylabel("Coherence/Amplitude")
    ax1.legend(); ax1.grid()

    mean_s2 = np.mean(data[1])
    var_s2 = np.var(data[1])
    ax2.set_title("SPHY Stability")
    ax2.plot(new_time, data[1], 'r-', label="Signal 2")
    ax2.axhline(mean_s2, color='green', linestyle='--', label=f"Mean: {mean_s2:.2f}")
    ax2.axhline(mean_s2 + np.sqrt(var_s2), color='orange', linestyle='--')
    ax2.axhline(mean_s2 - np.sqrt(var_s2), color='orange', linestyle='--', label=f"± Variance")
    ax2.set_xlabel("Normalized Time")
    ax2.set_ylabel("Coherence/Amplitude")
    ax2.legend(); ax2.grid()

    fig.suptitle("SPHY - Quantum Stability", fontsize=16)
    fig.tight_layout()
    plt.savefig(fig_filename, dpi=300)
    # plt.close(fig) # Commented out to allow plt.show() to display
    print(f"🖼️ 2D Graph saved: {fig_filename}")


# 🌌 3D SPHY Field Trajectory Plot
def plot_3d_sphy_trajectory(csv_filename, fig_filename_3d):
    """
    Generates a 3D plot showing the trajectory of the SPHY coherence and qubit phase over time.
    """
    try:
        df = pd.read_csv(csv_filename)
        X = df["Frame"].values
        Y = df["SPHY (%)"].values
        phase_col = "Qubit_1_Phase" 
        
        if phase_col not in df.columns:
            print(f"❌ Error: Phase column '{phase_col}' not found in CSV. Cannot generate 3D plot.")
            return
            
        Z = df[phase_col].values
    except Exception as e:
        print(f"❌ Error loading CSV for 3D plot: {e}")
        return

    accepted_mask = df["Accepted"] == '✅'
    rejected_mask = df["Accepted"] == '❌'

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d') 
    ax.plot(X, Y, Z, color='gray', linewidth=0.5, alpha=0.5)
    ax.scatter(X[accepted_mask], Y[accepted_mask], Z[accepted_mask], c='green', marker="o", s=20, label="✅ Accepted")
    ax.scatter(X[rejected_mask], Y[rejected_mask], Z[rejected_mask], c='red', marker="x", s=20, label="❌ Rejected")
    ax.scatter(X[0], Y[0], Z[0], c="blue", s=50, marker="s", label="Start")

    ax.set_xlabel("Frame")
    ax.set_ylabel("SPHY Coherence (%)")
    ax.set_zlabel(f"Qubit 1 Phase (Q{TARGET_QUBIT_INDEX})")
    ax.set_title("🌌 3D SPHY Trajectory (Quantum Tunneling)")
    ax.legend()
    ax.grid(True)
    plt.savefig(fig_filename_3d, dpi=300)
    # plt.close(fig) # Commented out to allow plt.show() to display
    print(f"🖼️ 3D Graph saved: {fig_filename_3d}")

# 🧠 Multiprocessing Execution
def execute_simulation_multiprocessing(num_qubits, total_frames, rz_angle, noise_prob=1.0, num_processes=4):
    print("=" * 60)
    print(f"🔢 Number of Qubits (Lattice 2x2): {num_qubits}")
    print(f"🔁 Total Attempts (Frames): {total_frames:,}")
    print(f"🚧 Barrier Angle (RZ): {rz_angle * 180 / np.pi:.2f}°")
    print("=" * 60)
    print(f" ⚛️ SPHY WAVES: Toroidal Tunneling (Realist)")
    print("=" * 60)

    timecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_log_{timecode}.csv")
    fig_filename_2d = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_2D_{timecode}.png")
    fig_filename_3d = os.path.join(LOG_DIR, f"qghz_{num_qubits}q_graph_3D_{timecode}.png")


    manager = Manager()
    sphy_coherence = manager.Value('f', 90.0)
    log_data, sphy_evolution = manager.list(), manager.list()
    accepted_total = manager.Value('i', 0)
    
    frame_inputs = [
        (f, num_qubits, total_frames, noise_prob, sphy_coherence.value, rz_angle) 
        for f in range(1, total_frames + 1)
    ]

    print(f"🔧 Using {num_processes} process(es) for simulation...")
    with Pool(processes=num_processes) as pool:
        for log_entry, new_coherence, error in tqdm(pool.imap_unordered(simulate_frame, frame_inputs),
                                                    total=total_frames, desc="⏳ Simulating GHZ"):
            if error:
                print(f"\n{error}", file=sys.stderr)
                pool.terminate()
                break
            if log_entry:
                log_data.append(log_entry)
                sphy_evolution.append(new_coherence)
                sphy_coherence.value = new_coherence
                
                # The "Accepted" item is at index [-3]
                if log_entry[-3] == "✅": 
                    accepted_total.value += 1

    acceptance_rate = 100 * (accepted_total.value / total_frames) if total_frames > 0 else 0

    # 1. Save CSV
    header = ["Frame", "Result", "Qubit_1_Phase", "H", "S", "C", "I", "Boost", "SPHY (%)", "Accepted", "SHA256_Signature", "Timestamp"]
    with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(list(log_data))
    print(f"📥 CSV saved: {csv_filename}")

    # 2. Generate Graphs
    sphy_evolution_list = list(sphy_evolution)
    if sphy_evolution_list:
        plot_2d_stability(sphy_evolution_list, fig_filename_2d)
        plot_3d_sphy_trajectory(csv_filename, fig_filename_3d)

    # 3. Report
    sphy_evolution_np = np.array(sphy_evolution_list)
    mean_stability = np.mean(sphy_evolution_np)
    variance_stability = np.var(sphy_evolution_np)
    purity = min(((mean_stability / 100.0) ** 2) + 0.05, 1.0)
    squeezing_min = np.min(np.diff(np.clip(sphy_evolution_np / 100.0, 0, 1))) if len(sphy_evolution_np) > 2 else 0.0
    wgrid = np.linspace(0, np.pi, len(sphy_evolution_np))
    wigner_simulated = np.abs(np.sin(wgrid) * (sphy_evolution_np / 100.0))
    wigner_max = np.max(wigner_simulated)

    print("\n" + "=" * 60)
    print("           📊 SPHY PERFORMANCE REPORT")
    print("-" * 60)
    print(f"| ✅ Tunneling Success Rate (SPHY Toroidal): {accepted_total.value}/{total_frames:,}".replace(',', '.') + f"  ➤  {acceptance_rate:.2f}%")
    print("-" * 60)
    print(f"| ⭐ Mean SPHY Stability: {mean_stability:.4f}")
    print(f"| 🌊 Stability Variance: {variance_stability:.6f}")
    print("-" * 60)
    print(f"| ⚛️ Final State Purity (μ): {purity:.4f}     (SPHY Ideal < 1.0 | QEC Ideal = 1.0)")
    print(f"| 🔬 Minimum Squeezing (λ_min): {squeezing_min:.4f}   (Squeezed < 0.5)")
    print(f"| 📈 Maximum Wigner Value (W_max): {wigner_max:.4f}         (W_max ≤ 1/π ≈ 0.3183)")
    print("=" * 60)

    # ⭐ VISUALIZATION FIX: Call plt.show() to open the graphs
    if sphy_evolution_list:
        plt.show()


# 🎬 Entry Point
if __name__ == "__main__":
    qubits, pairs, rz_angle = get_user_parameters() 
    
    execute_simulation_multiprocessing(
        num_qubits=qubits, 
        total_frames=pairs, 
        rz_angle=rz_angle, 
        noise_prob=1.0, 
        num_processes=4
    )