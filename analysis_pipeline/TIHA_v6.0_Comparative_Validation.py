# =============================================================================
# TIHA SCRIPT v6.0 (Comparative Validation: EEG vs. PNS)
#
# File: TIHA_v6.0_Comparative_Validation.py
# Author: Leandro Antonio da Silva Pacheco (Original)
# Date: 2025-11-09
#
# OVERVIEW:
# This is the canonical analysis script for the TIHA v6.0 paper.
# Its purpose is to conduct the "Model Competition" on the multimodal
# DEAP dataset to validate the "TIHA-Lite" (PNS/Wearable) pivot.
#
# This script compares two Idiosyncratic Linear Models (MLI):
#
# 1. Model A (v5.1 EEG): [Φ, E, S_eeg]
#    - Validated in the v5.1 analysis (R2=0.181, VIF<2.74, p_paradox>0.1)
#
# 2. Model B (v6.0 PNS): [HRV_rmssd, EDA_peaks, S_gsr]
#    - The new "TIHA-Lite" model being validated here.
#
# This script generates the 'TIHA_v6.0_results.json' file, which
# provides the final comparative verdict (EEG R2=0.181 vs. PNS R2=0.091).
#
# All proxy calculations are imported from 'tiha_core.py'.
# =============================================================================

# --- 0. Environment Setup ---
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, zscore, ttest_1samp
import pickle
import glob
import os
import json 

# --- Import Core TIHA Library ---
try:
    import tiha_core
except ImportError:
    print("FATAL ERROR: 'tiha_core.py' not found.")
    print("Please ensure 'tiha_core.py' is in the same directory or Python path.")
    sys.exit(1)

# Import ML libraries
from sklearn.linear_model import Ridge 
from sklearn.metrics import r2_score, mean_absolute_error 
from statsmodels.stats.outliers_influence import variance_inflation_factor 
import statsmodels.api as sm

def setup_environment():
    """Sets global configurations for the analysis."""
    np.random.seed(42) # Ensures reproducibility
    plt.style.use('seaborn-v0_8-whitegrid')
    warnings.filterwarnings('ignore') # Suppress warnings for clean output
    print("TIHA (v6.0) Comparative Analysis Environment Loaded.")
    print("-" * 50)

# --- 1. Data Loading Functions ---

# Path to the DEAP dataset. 
# See README.md for instructions on downloading and placing the data.
PATH_TO_DEAP_DATA = "./deap_data/data_preprocessed_python/s*.dat"

def load_deap_data_multimodal(path_pattern=PATH_TO_DEAP_DATA):
    """
    Generator function to load multimodal data (EEG + PNS) from DEAP files.
    """
    participant_files = sorted(glob.glob(path_pattern))
    if not participant_files:
        print(f"WARNING: No DEAP files ('{path_pattern}') found.")
        print("Please check the PATH_TO_DEAP_DATA variable and README.md.")
        return
    
    print(f"Found {len(participant_files)} participant files (s01.dat ... s32.dat).")

    for p_idx, p_file in enumerate(participant_files):
        participant_id = p_idx + 1
        try:
            with open(p_file, 'rb') as f:
                participant_data = pickle.load(f, encoding='latin1')
            
            data = participant_data['data']   # (40, 40, 8064)
            labels = participant_data['labels'] # (40, 4)
            
            for i in range(data.shape[0]): # Iterate over 40 trials
                # CNS (EEG) Signals
                eeg_data = data[i, :32, :]      # Channels 1-32
                # PNS (Peripheral) Signals
                gsr_data = data[i, 36, :]       # Channel 37: GSR
                pleth_data = data[i, 38, :]     # Channel 39: Plethysmograph (for HRV)
                
                valence_label_deap = labels[i, 0] # V_H (Hedonic Valence)
                liking_label_deap = labels[i, 3]  # V_E (Eudaimonic Valence)
                
                yield (participant_id, eeg_data, gsr_data, pleth_data, 
                       valence_label_deap, liking_label_deap)
        except Exception as e:
            print(f"Error processing multimodal file {p_file}: {e}")

def normalize_valence(v_deap):
    """Normalizes DEAP valence/liking scores (1-9) to (-1, 1)."""
    return (v_deap - 5.0) / 4.0

# --- Proxy functions are all in 'tiha_core.py' ---

def run_analysis():
    """
    Main function to execute the full v6.0 comparative analysis pipeline.
    """
    
    # --- 2. PHASE 1: Raw Data Collection (All 6 Proxies) ---
    print("\n--- Starting TIHA v6.0 Comparative Validation Test ---")
    print("Phase 1: Collecting data (3 EEG + 3 PNS) from 1280 trials...")
    all_data_list = []
    n_trials_processed = 0

    for p_id, eeg, gsr, pleth, vh_deap, ve_deap in load_deap_data_multimodal():
        vh_norm = normalize_valence(vh_deap)
        ve_norm = normalize_valence(ve_deap)
        
        # --- Model A: EEG (v5.1) Proxies ---
        phi_raw = tiha_core.calcular_phi_alpha_real(eeg, fs=128)
        e_raw = tiha_core.calcular_entropia_espectral_real(eeg, fs=128)
        s_eeg_raw = tiha_core.calcular_surpresa_eeg_fz(eeg, channel_index=4)
        
        # --- Model B: PNS (v6.0) Proxies ---
        hrv_raw = tiha_core.calcular_hrv_rmssd(pleth, fs=128)
        eda_raw = tiha_core.calcular_eda_phasic_peaks(gsr, fs=128)
        s_gsr_raw = tiha_core.calcular_surpresa_gsr(gsr)
        
        # Validation: Ignore trials where HRV peak-finding failed (hrv_raw=0)
        if all(np.isfinite([vh_norm, ve_norm, phi_raw, e_raw, s_eeg_raw, hrv_raw, eda_raw, s_gsr_raw])) and hrv_raw > 0:
            all_data_list.append({
                'p_id': p_id, 'val_h': vh_norm, 'val_e': ve_norm,
                'phi': phi_raw, 'e': e_raw, 's_eeg': s_eeg_raw,
                'hrv': hrv_raw, 'eda': eda_raw, 's_gsr': s_gsr_raw
            })
            n_trials_processed += 1
        
        if n_trials_processed % 100 == 0 and n_trials_processed > 99:
            print(f"   ... {n_trials_processed} multimodal trials collected...")

    if n_trials_processed < 100: # Fail-safe
        raise Exception(f"ERROR: Phase 1 failed. Only {n_trials_processed} valid trials collected (HRV peak detection may have failed?).")
    
    df = pd.DataFrame(all_data_list)
    print(f"Phase 1 complete. {n_trials_processed} valid multimodal trials collected.")

    # --- 3. PHASE 2: Intra-Subject Test Loop (Model Competition) ---
    print("Phase 2: Starting Intra-Subject test loop (Model Competition)...")

    # Model A (v5.1 EEG): [Phi, E, S_eeg]
    model_A_vif = {'phi': [], 'e': [], 's_eeg': []}
    model_A_vh = {'r2': [], 'mae': [], 'r': []}
    model_A_ve = {'r2': [], 'mae': [], 'r': []}

    # Model B (v6.0 PNS): [HRV, EDA, S_gsr]
    model_B_vif = {'hrv': [], 'eda': [], 's_gsr': []}
    model_B_vh = {'r2': [], 'mae': [], 'r': []}
    model_B_ve = {'r2': [], 'mae': [], 'r': []}

    n_participants_valid = 0
    model = Ridge(alpha=1.0) 

    for p_id_test in df['p_id'].unique():
        df_subject = df[df['p_id'] == p_id_test]
        if len(df_subject) < 10: continue
        
        y_vh = df_subject['val_h'].values
        y_ve = df_subject['val_e'].values

        # --- Prepare Model A (EEG) ---
        phi_norm = np.nan_to_num(zscore(df_subject['phi'].values), nan=0.0)
        e_norm = np.nan_to_num(zscore(df_subject['e'].values), nan=0.0)
        s_eeg_norm = np.nan_to_num(zscore(df_subject['s_eeg'].values), nan=0.0)
        X_eeg = np.stack([phi_norm, e_norm, s_eeg_norm], axis=1)

        # --- Prepare Model B (PNS) ---
        hrv_norm = np.nan_to_num(zscore(df_subject['hrv'].values), nan=0.0)
        eda_norm = np.nan_to_num(zscore(df_subject['eda'].values), nan=0.0)
        s_gsr_norm = np.nan_to_num(zscore(df_subject['s_gsr'].values), nan=0.0)
        X_snp = np.stack([hrv_norm, eda_norm, s_gsr_norm], axis=1)

        try:
            # --- Test A (EEG v5.1) ---
            X_vif_a = sm.add_constant(X_eeg); vif_data_a = [variance_inflation_factor(X_vif_a, i) for i in range(1, X_vif_a.shape[1])]
            model_A_vif['phi'].append(vif_data_a[0]); model_A_vif['e'].append(vif_data_a[1]); model_A_vif['s_eeg'].append(vif_data_a[2])
            
            model.fit(X_eeg, y_vh); y_pred_vh_a = model.predict(X_eeg)
            model_A_vh['r2'].append(r2_score(y_vh, y_pred_vh_a)); model_A_vh['mae'].append(mean_absolute_error(y_vh, y_pred_vh_a)); model_A_vh['r'].append(pearsonr(y_vh, y_pred_vh_a)[0])
            
            model.fit(X_eeg, y_ve); y_pred_ve_a = model.predict(X_eeg)
            model_A_ve['r2'].append(r2_score(y_ve, y_pred_ve_a)); model_A_ve['mae'].append(mean_absolute_error(y_ve, y_pred_ve_a)); model_A_ve['r'].append(pearsonr(y_ve, y_pred_ve_a)[0])

            # --- Test B (PNS v6.0) ---
            X_vif_b = sm.add_constant(X_snp); vif_data_b = [variance_inflation_factor(X_vif_b, i) for i in range(1, X_vif_b.shape[1])]
            model_B_vif['hrv'].append(vif_data_b[0]); model_B_vif['eda'].append(vif_data_b[1]); model_B_vif['s_gsr'].append(vif_data_b[2])

            model.fit(X_snp, y_vh); y_pred_vh_b = model.predict(X_snp)
            model_B_vh['r2'].append(r2_score(y_vh, y_pred_vh_b)); model_B_vh['mae'].append(mean_absolute_error(y_vh, y_pred_vh_b)); model_B_vh['r'].append(pearsonr(y_vh, y_pred_vh_b)[0])

            model.fit(X_snp, y_ve); y_pred_ve_b = model.predict(X_snp)
            model_B_ve['r2'].append(r2_score(y_ve, y_pred_ve_b)); model_B_ve['mae'].append(mean_absolute_error(y_ve, y_pred_ve_b)); model_B_ve['r'].append(pearsonr(y_ve, y_pred_ve_b)[0])
            
            n_participants_valid += 1
        except Exception:
            continue

    print(f"Phase 2 complete. {n_participants_valid} participants analyzed in both models.")

    # --- 4. PHASE 3: Final Analysis (v6.0 - Model Competition) ---
    if n_participants_valid > 1:

        # --- Collect Results (A: EEG) ---
        r2_vh_a = np.mean(model_A_vh['r2']); t_vh_a, p_vh_a = ttest_1samp(model_A_vh['r2'], 0, alternative='greater')
        mae_vh_a = np.mean(model_A_vh['mae']); r_vh_a = np.mean(model_A_vh['r'])
        vif_a_max = max(np.mean(model_A_vif['phi']), np.mean(model_A_vif['e']), np.mean(model_A_vif['s_eeg']))
        
        # --- Collect Results (B: PNS) ---
        r2_vh_b = np.mean(model_B_vh['r2']); t_vh_b, p_vh_b = ttest_1samp(model_B_vh['r2'], 0, alternative='greater')
        mae_vh_b = np.mean(model_B_vh['mae']); r_vh_b = np.mean(model_B_vh['r'])
        vif_b_max = max(np.mean(model_B_vif['hrv']), np.mean(model_B_vif['eda']), np.mean(model_B_vif['s_gsr']))
        
        # --- Console Display (V_H Comparison) ---
        print("\n" + "=" * 70)
        print("--- RESULTS (v6.0) - Model Competition (Hedonic Valence V_H) ---")
        print("=" * 70)
        print(f"  METRIC        | Model v5.1 (EEG) [Phi, E, S_eeg] | Model v6.0 (PNS) [HRV, EDA, S_gsr]")
        print("-" * 70)
        print(f"  R2 (Mean)     | {r2_vh_a:<25.4f} | {r2_vh_b:<25.4f}")
        print(f"  p-value (R2)  | {p_vh_a:<25e} | {p_vh_b:<25e}")
        print(f"  MAE (Mean)    | {mae_vh_a:<25.4f} | {mae_vh_b:<25.4f}")
        print(f"  r (Mean)      | {r_vh_a:<25.4f} | {r_vh_b:<25.4f}")
        print(f"  VIF (Max)     | {vif_a_max:<25.3f} | {vif_b_max:<25.3f}")
        print("=" * 70)

        # --- Final Verdict ---
        print("\n--- FINAL VERDICT (v6.0) - 'TIHA-Lite' Pivot ---")
        if p_vh_b < 0.05:
            print(f"  VERDICT: SUCCESS! The PNS (Wearable) Model is a statistically significant predictor.")
            print(f"  Performance Loss (R2): {((r2_vh_a - r2_vh_b) / r2_vh_a) * 100:.1f}%")
            print(f"  (EEG R2={r2_vh_a:.3f} vs PNS R2={r2_vh_b:.3f}). The pivot is viable.")
        else:
            print(f"  VERDICT: FAILURE. The PNS (Wearable) Model is NOT a significant predictor (p={p_vh_b:.3f}).")
            print("  Conclusion: The peripheral proxies (HRV/EDA) from DEAP do not capture valence.")
        print("=" * 70)

        # --- Results Export ---
        # NOTE: JSON keys are kept in Portuguese to match the original v6.0
        # results artifact, ensuring 1:1 replication.
        output_data = {
            "model_version": "TIHA_v6.0_Comparative_Validation",
            "N_participantes": n_participants_valid,
            "Model_A_v5_1_EEG": {
                "components": ["Phi (Integracao Alfa)", "E (Entropia Espectral)", "S (Surpresa EEG-Fz)"],
                "validation_hedonic_VH": {"R2_mean": r2_vh_a, "R2_p_value": p_vh_a, "MAE_mean": mae_vh_a, "r_mean": r_vh_a},
                "robustness_VIF": {"VIF_max": vif_a_max, "verdict": "SUCESSO (VIF < 5)" if vif_a_max < 5 else "FALHA (VIF >= 5)"}
            },
            "Model_B_v6_0_SNP_TIHA_Lite": {
                "components": ["HRV (RMSSD)", "EDA (Phasic Peaks)", "S (Surpresa GSR-Var)"],
                "validation_hedonic_VH": {"R2_mean": r2_vh_b, "R2_p_value": p_vh_b, "MAE_mean": mae_vh_b, "r_mean": r_vh_b},
                "robustness_VIF": {"VIF_max": vif_b_max, "verdict": "SUCESSO (VIF < 5)" if vif_b_max < 5 else "FALHA (VIF >= 5)"}
            }
        }
        output_filename = "TIHA_v6.0_results.json"
        try:
            with open(output_filename, 'w') as f: json.dump(output_data, f, indent=4)
            print(f"\n--- SUCCESS: Final results (v6.0) exported to '{output_filename}' ---")
        except Exception as e:
            print(f"\n--- EXPORT FAILED: {e} ---")

    else:
        print("\nERROR: Phase 3 failed. No regressions were completed.")

# Standard Python practice: call main() when the script is executed
if __name__ == "__main__":
    setup_environment()
    try:
        run_analysis()
    except Exception as e:
        print(f"\n--- ANALYSIS FAILED ---")
        print(f"Error: {e}")
        print("Please check data paths and 'requirements.txt'.")
