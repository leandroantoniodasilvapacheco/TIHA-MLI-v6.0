# =============================================================================
# TIHA SCRIPT v5.1 (Publication & Paradox Test)
#
# File: TIHA_v5.1_Publication_Paradox.py
# Author: Leandro Antonio da Silva Pacheco (Original)
# Date: 2025-11-09
#
# OVERVIEW:
# This is the canonical analysis script used to generate the results for
# the TIHA v5.1 model (EEG-only), as presented in the v6.0 paper.
# It validates the 3-component Idiosyncratic Linear Model (MLI) 
# [Φ, E, S_eeg] using Ridge Regression.
#
# This script performs 5 key analyses, matching the v5.1 results file:
# 1. (Part A) Validation for Hedonic Valence (V_H): R2=0.181
# 2. (Part B) Validation for Eudaimonic Valence (V_E): R2=0.151
# 3. (Part C) Robustness Check (VIF): VIF_max=2.739
# 4. (Part D) Driver Analysis (Mean |Beta|): Φ (Integration) is primary.
# 5. (Part E) Idiosyncratic Paradox Test (t-test): p_min=0.104
#
# This script has been refactored to import all proxy calculations
# from the 'tiha_core.py' library, ensuring clean, reproducible code.
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
# All proxy functions (e.g., calcular_phi_alpha_real) are now imported
# from the central library, not re-defined here.
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
    np.random.seed(42)  # Ensures reproducibility
    plt.style.use('seaborn-v0_8-whitegrid')
    warnings.filterwarnings('ignore') # Suppress warnings for clean output
    print("TIHA (v5.1) Analysis Environment Loaded.")
    print("-" * 50)

# --- 1. Data Loading Functions ---

# Path to the DEAP dataset. 
# See README.md for instructions on downloading and placing the data.
PATH_TO_DEAP_DATA = "./deap_data/data_preprocessed_python/s*.dat"

def load_deap_data(path_pattern=PATH_TO_DEAP_DATA):
    """
    Generator function to load and yield data from the preprocessed DEAP files.
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
            
            data = participant_data['data']   # (40 trials, 40 channels, 8064 samples)
            labels = participant_data['labels'] # (40 trials, 4 labels)
            
            for i in range(data.shape[0]): # Iterate over 40 trials
                eeg_data = data[i, :32, :]      # Channels 1-32 are EEG
                valence_label_deap = labels[i, 0] # V_H (Hedonic Valence)
                liking_label_deap = labels[i, 3]  # V_E (Eudaimonic Valence)
                yield participant_id, eeg_data, valence_label_deap, liking_label_deap
        except Exception as e:
            print(f"Error processing file {p_file}: {e}")

def normalize_valence(v_deap):
    """Normalizes DEAP valence/liking scores (1-9) to (-1, 1)."""
    return (v_deap - 5.0) / 4.0

# --- Proxy functions (e.g., calcular_phi_alpha_real) are now in 'tiha_core.py' ---

def run_analysis():
    """
    Main function to execute the full v5.1 analysis pipeline.
    """
    
    # --- 2. PHASE 1: Raw Data Collection (3 Proxies) ---
    print("\n--- Starting TIHA v5.1 Final Test (Ridge + Paradox) ---")
    print("Phase 1: Collecting data (3 Proxies: Phi, E, S_eeg) from 1280 trials...")
    all_data_list = []
    n_trials_processed = 0

    for p_id, eeg_data, vh_deap, ve_deap in load_deap_data():
        vh_norm = normalize_valence(vh_deap)
        ve_norm = normalize_valence(ve_deap)
        
        # --- Call the 'tiha_core' library for feature extraction ---
        # 1. Φ (Phi) - Integration (Alpha-band Correlation)
        phi_raw = tiha_core.calcular_phi_alpha_real(eeg_data, fs=128)
        
        # 2. E (E) - Entropy (Spectral Entropy)
        e_raw = tiha_core.calcular_entropia_espectral_real(eeg_data, fs=128)
        
        # 3. S_eeg (S) - Surprise (Fz Variance)
        # The 'channel_index=4' matches the Fz channel in the DEAP dataset.
        s_raw = tiha_core.calcular_surpresa_eeg_fz(eeg_data, channel_index=4)
        # ---
        
        # Check for valid numerical data before appending
        if all(np.isfinite([vh_norm, ve_norm, phi_raw, e_raw, s_raw])):
            all_data_list.append({
                'p_id': p_id, 
                'val_h': vh_norm, 'val_e': ve_norm, 
                'phi': phi_raw, 'e': e_raw, 's': s_raw
            })
            n_trials_processed += 1
        
        if n_trials_processed % 100 == 0 and n_trials_processed > 99:
            print(f"   ... {n_trials_processed} trials collected...")

    if n_trials_processed == 0:
        raise Exception("ERROR: Phase 1 failed. No data was collected.")
    
    df = pd.DataFrame(all_data_list)
    print(f"Phase 1 complete. {n_trials_processed} valid trials collected.")

    # --- 3. PHASE 2: Intra-Subject Test Loop (Ridge Model v5.1) ---
    print("Phase 2: Starting Intra-Subject test loop (Model v5.1 [Phi, E, S] - Ridge)...")

    results_vif = {'phi': [], 'e': [], 's': []}
    results_vh = {'r2': [], 'mae': [], 'r': []}
    results_ve = {'r2': [], 'mae': [], 'r': []}
    # Store |Beta| for Driver Analysis (Part D)
    results_weights_abs_vh = {'phi': [], 'e': [], 's': []} 
    # Store raw Beta for Paradox Test (Part E)
    results_weights_raw_vh = {'phi': [], 'e': [], 's': []} 

    n_participants = df['p_id'].nunique()
    model = Ridge(alpha=1.0) # The v5.1 canonical model

    for p_id_test in range(1, n_participants + 1):
        df_subject = df[df['p_id'] == p_id_test]
        if len(df_subject) < 10: continue # Skip if insufficient data
        
        # Z-score (normalize) features *within* each subject
        phi_norm = np.nan_to_num(zscore(df_subject['phi'].values), nan=0.0)
        e_norm = np.nan_to_num(zscore(df_subject['e'].values), nan=0.0)
        s_norm = np.nan_to_num(zscore(df_subject['s'].values), nan=0.0) 
        
        y_vh = df_subject['val_h'].values 
        y_ve = df_subject['val_e'].values
        X = np.stack([phi_norm, e_norm, s_norm], axis=1)
        
        try:
            # --- Part C: VIF Check ---
            X_vif = sm.add_constant(X) # Add constant for VIF calculation
            vif_data = [variance_inflation_factor(X_vif, i) for i in range(1, X_vif.shape[1])]
            results_vif['phi'].append(vif_data[0])
            results_vif['e'].append(vif_data[1])
            results_vif['s'].append(vif_data[2])
            
            # --- Part A: Hedonic (V_H) Model ---
            model.fit(X, y_vh)
            y_pred_vh = model.predict(X)
            weights_vh = model.coef_
            results_vh['r2'].append(r2_score(y_vh, y_pred_vh))
            results_vh['mae'].append(mean_absolute_error(y_vh, y_pred_vh))
            results_vh['r'].append(pearsonr(y_vh, y_pred_vh)[0])
            
            # --- Part D & E: Store Weights ---
            results_weights_abs_vh['phi'].append(np.abs(weights_vh[0]))
            results_weights_abs_vh['e'].append(np.abs(weights_vh[1]))
            results_weights_abs_vh['s'].append(np.abs(weights_vh[2]))
            
            results_weights_raw_vh['phi'].append(weights_vh[0])
            results_weights_raw_vh['e'].append(weights_vh[1])
            results_weights_raw_vh['s'].append(weights_vh[2])
            
            # --- Part B: Eudaimonic (V_E) Model ---
            model.fit(X, y_ve)
            y_pred_ve = model.predict(X)
            results_ve['r2'].append(r2_score(y_ve, y_pred_ve))
            results_ve['mae'].append(mean_absolute_error(y_ve, y_pred_ve))
            results_ve['r'].append(pearsonr(y_ve, y_pred_ve)[0])
            
        except Exception:
            continue # Skip participant on error

    print(f"Phase 2 complete. {len(results_vh['r2'])} participants analyzed.")

    # --- 4. PHASE 3: Final Analysis (v5.1) ---
    if len(results_vh['r2']) > 1:

        # --- Part A: Hedonic (V_H) Results ---
        r2_vh = np.mean(results_vh['r2'])
        mae_vh = np.mean(results_vh['mae'])
        r_vh = np.mean(results_vh['r'])
        # One-sample t-test (R2 > 0)
        t_vh, p_vh = ttest_1samp(results_vh['r2'], 0, alternative='greater')
        
        # --- Part B: Eudaimonic (V_E) Results ---
        r2_ve = np.mean(results_ve['r2'])
        mae_ve = np.mean(results_ve['mae'])
        r_ve = np.mean(results_ve['r'])
        t_ve, p_ve = ttest_1samp(results_ve['r2'], 0, alternative='greater')
        
        # --- Part C: Robustness (VIF) Results ---
        vif_phi = np.mean(results_vif['phi'])
        vif_e = np.mean(results_vif['e'])
        vif_s = np.mean(results_vif['s'])
        vif_max = max(vif_phi, vif_e, vif_s)
        
        # --- Part D: Driver Analysis (V_H) Results ---
        mean_abs_phi = np.mean(results_weights_abs_vh['phi'])
        mean_abs_e = np.mean(results_weights_abs_vh['e'])
        mean_abs_s = np.mean(results_weights_abs_vh['s'])
        forces = {"Phi (Integration)": mean_abs_phi, "E (Entropy)": mean_abs_e, "S (Surprise)": mean_abs_s}
        primary_driver = max(forces, key=forces.get)
        
        # --- Part E: Idiosyncratic Paradox Test (V_H) Results ---
        # H0: The mean weight (Beta) across all subjects is 0.
        # If we FAIL to reject H0 (p > 0.05), it means there is no 
        # "universal" component, confirming the Idiosyncratic Paradox.
        t_paradox_phi, p_paradox_phi = ttest_1samp(results_weights_raw_vh['phi'], 0)
        t_paradox_e, p_paradox_e = ttest_1samp(results_weights_raw_vh['e'], 0)
        t_paradox_s, p_paradox_s = ttest_1samp(results_weights_raw_vh['s'], 0)
        p_min_paradox = min(p_paradox_phi, p_paradox_e, p_paradox_s)
        
        # --- Console Display (Translated to English) ---
        print("\n" + "=" * 60)
        print("--- RESULTS (v5.1) - Part A: Hedonic Valence (V_H) ---")
        print(f"  Mean R2: {r2_vh:.4f} (p={p_vh:e}) | MAE: {mae_vh:.4f} | r: {r_vh:.4f}")
        print("=" * 60)
        
        print("\n" + "=" * 60)
        print("--- RESULTS (v5.1) - Part B: Eudaimonic Valence (V_E) ---")
        print(f"  Mean R2: {r2_ve:.4f} (p={p_ve:e}) | MAE: {mae_ve:.4f} | r: {r_ve:.4f}")
        print("=" * 60)
        
        print("\n" + "=" * 60)
        print("--- RESULTS (v5.1) - Part C: Robustness Analysis (VIF) ---")
        print(f"  Mean VIF (Φ): {vif_phi:.3f} | (E): {vif_e:.3f} | (S): {vif_s:.3f}")
        print(f"  VIF VERDICT (v5.1): {'SUCCESS' if vif_max < 5 else 'FAILURE'} (Max: {vif_max:.3f})")
        print("=" * 60)
        
        print("\n" + "=" * 60)
        print("--- RESULTS (v5.1) - Part D: Driver Analysis (V_H) ---")
        print(f"  Mean Force |Φ| (Alpha Integration): {mean_abs_phi:.4f}")
        print(f"  Mean Force |E| (Spectral Entropy): {mean_abs_e:.4f}")
        print(f"  Mean Force |S| (Surprise/Variance): {mean_abs_s:.4f}")
        print(f"\n  DRIVER VERDICT (v5.1): **{primary_driver}**")
        print("=" * 60)
        
        print("\n" + "=" * 60)
        print("--- RESULTS (v5.1) - Part E: Idiosyncratic Paradox ---")
        print("  (t-test on raw mean weights. H0: Mean(Weight) = 0)")
        print(f"  P-Value (Mean Weight Φ): {p_paradox_phi:.4f}")
        print(f"  P-Value (Mean Weight E): {p_paradox_e:.4f}")
        print(f"  P-Value (Mean Weight S): {p_paradox_s:.4f}")
        if p_min_paradox > 0.05:
            print(f"\n  PARADOX VERDICT (v5.1): SUCCESS (P-min = {p_min_paradox:.3f} > 0.05)")
            print("  No mean weight is universal. Idiosyncratic hypothesis is confirmed.")
        else:
            print(f"\n  PARADOX VERDICT (v5.1): FAILURE (P-min = {p_min_paradox:.3f} < 0.05)")
            print("  A universal component was detected. The paradox is refuted.")
        print("=" * 60)
        
        # --- Part F: Results Export ---
        # This section is preserved exactly to ensure the output JSON
        # file is identical for replication, including Portuguese keys.
        output_data = {
            "model_version": "TIHA_v5.1_Ridge_Paradox",
            "model_components": ["Phi (Integracao Alfa)", "E (Entropia Espectral)", "S (Surpresa/Variancia)"],
            "model_type": "Ridge Regression (alpha=1.0)", "N_participantes": len(results_vh['r2']),
            "validation_hedonic_VH": {"R2_mean": r2_vh, "R2_p_value": p_vh, "MAE_mean": mae_vh, "r_mean": r_vh},
            "validation_eudaimonic_VE": {"R2_mean": r2_ve, "R2_p_value": p_ve, "MAE_mean": mae_ve, "r_mean": r_ve},
            "robustness_VIF": {"VIF_Phi_mean": vif_phi, "VIF_E_mean": vif_e, "VIF_S_mean": vif_s, "VIF_max": vif_max, "verdict": "SUCESSO (VIF < 5)" if vif_max < 5 else "FALHA (VIF >= 5)"},
            "driver_analysis_VH": {"driver_Phi_mean_abs_beta": mean_abs_phi, "driver_E_mean_abs_beta": mean_abs_e, "driver_S_mean_abs_beta": mean_abs_s, "primary_driver": primary_driver},
            "idiosyncratic_paradox_VH": {
                "p_value_mean_beta_Phi": p_paradox_phi,
                "p_value_mean_beta_E": p_paradox_e,
                "p_value_mean_beta_S": p_paradox_s,
                "verdict": "SUCESSO (Paradoxo Confirmado, p > 0.05)" if p_min_paradox > 0.05 else "FALHA (Componente Universal Encontrado, p < 0.05)"
            }
        }
        output_filename = "TIHA_v5.1_results.json"
        try:
            with open(output_filename, 'w') as f:
                json.dump(output_data, f, indent=4)
            print(f"\n--- SUCCESS: Final results (v5.1) exported to '{output_filename}' ---")
        except Exception as e:
            print(f"\n--- EXPORT FAILED: {e} ---")

        # --- 6. PHASE 4: Visualization (Final Model) ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: R-squared Distribution
        axes[0].hist(results_vh['r2'], bins=10, edgecolor='black', alpha=0.7)
        axes[0].axvline(r2_vh, color='red', linestyle='dashed', linewidth=2, label=f'Mean (R2 = {r2_vh:.3f})')
        axes[0].axvline(0, color='black', linestyle='solid', linewidth=1)
        axes[0].set_title('R-squared Distribution (v5.1 - V_H)', fontsize=16)
        axes[0].set_xlabel('R-squared (Predictive Power)')
        axes[0].set_ylabel('Number of Participants')
        axes[0].legend()
        
        # Plot 2: Proxy Force Distribution (Box Plot)
        weights_abs_df = pd.DataFrame({
            '|Φ| (Integration)': results_weights_abs_vh['phi'], 
            '|E| (Entropy)': results_weights_abs_vh['e'], 
            '|S| (Surprise)': results_weights_abs_vh['s']
        })
        weights_abs_df.plot(kind='box', ax=axes[1], sym='k+')
        axes[1].axhline(0, color='black', linestyle='dashed', linewidth=1)
        axes[1].set_title('Proxy Force Distribution (v5.1 - V_H)', fontsize=16)
        axes[1].set_xlabel('TIHA Components (Final Model)')
        axes[1].set_ylabel('Absolute Regression Weight |Beta|')
        
        plt.tight_layout()
        plt.savefig("TIHA_v5.1_plots.png")
        print(f"--- SUCCESS: Plots saved to 'TIHA_v5.1_plots.png' ---")
        # plt.show() # Disabled for non-interactive environments
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
