# =============================================================================
# TIHA Framework - Example (Phase III): 'TIHA-Lite' Wearable App Demo
#
# File: demo_snp.py
#
# Author: Leandro Antonio da Silva Pacheco (Original)
# Date: 2025-11-09
#
# OVERVIEW:
# This script serves as a high-level demonstration for a "Phase III" 
# application ("TIHA-Lite") as described in the TIHA Framework 'Manual.md'.
#
# This "TIHA-Lite" model was validated in the v6.0 paper (R2=0.091, p<4.9e-11),
# proving that a viable MLI model can be built using *only* peripheral 
# nervous system (PNS) signals.
#
# This script simulates capturing a window of PNS data from a wearable 
# (e.g., Apple Watch, Whoop, Oura Ring) and calculates the three validated 
# proxies from the canonical v6.0 PNS Model:
#
# 1. HRV   - Variability (PPG/ECG RMSSD)
# 2. EDA   - Activation (GSR Phasic Peaks)
# 3. S_gsr - Surprise (GSR Variance)
#
# This script is for demonstration only and uses simulated (random) data.
# =============================================================================

import tiha_core  # Imports the core TIHA proxy calculation functions
import numpy as np

def run_snp_demo():
    """
    Main function to run the "TIHA-Lite" (Phase III) demo simulation.
    """
    
    # --- 1. Hardware Simulation (Mock Data) ---
    # In a real-world application (Phase III), this section would be replaced 
    # by the SDK from your specific wearable hardware (e.g., Apple HealthKit).
    
    FS = 128                # Sampling Frequency (Hz). This matches the DEAP dataset.
    DURATION_SECONDS = 10   # PNS signals (especially for HRV) require longer 
                            # windows for stable, reliable feature extraction.
    
    print(f"Simulating {DURATION_SECONDS}s of mock PNS data (GSR & PPG @ {FS}Hz)...")
    
    # --- Mock PPG (Photoplethysmograph) Signal ---
    # Simulates a signal for detecting heartbeats (peaks) to calculate HRV.
    # This is a simple sine wave (~72 bpm) plus noise.
    n_samples = DURATION_SECONDS * FS
    time = np.arange(n_samples) / FS
    heartbeats = np.sin(2 * np.pi * 1.2 * time) # 1.2 Hz = 72 bpm
    noise = np.random.rand(n_samples) * 0.3
    mock_ppg_data = heartbeats + noise

    # --- Mock GSR (Galvanic Skin Response) Signal ---
    # Simulates a signal for measuring electrodermal activity (skin conductance).
    mock_gsr_data = np.random.rand(n_samples) * 0.1

    # --- 2. Proxy Calculation (The TIHA 'Engine') ---
    # This section calls the validated functions from 'tiha_core' to 
    # calculate the three components (features) of the v6.0 PNS model.
    print("Calculating TIHA v6.0 'TIHA-Lite' (PNS) proxies from mock data...")

    # Proxy 1: HRV (Variability)
    # Calculates RMSSD (Root Mean Square of Successive Differences) from the
    # simulated PPG signal. This is a standard measure of parasympathetic 
    # nervous system activity, or "resilience".
    hrv = tiha_core.calcular_hrv_rmssd(mock_ppg_data, fs=FS)
    
    # Proxy 2: EDA (Activation)
    # Calculates the count of phasic (rapid) peaks in the GSR signal.
    # This measures sympathetic nervous system arousal (e.g., "fight or flight").
    eda = tiha_core.calcular_eda_phasic_peaks(mock_gsr_data, fs=FS)
    
    # Proxy 3: S_gsr (Surprise)
    # Calculates the simple variance of the raw GSR signal.
    # This is the PNS analogue to the S_eeg proxy, measuring the 
    # "unpredictability" or "surprise" of the peripheral signal.
    s_gsr = tiha_core.calcular_surpresa_gsr(mock_gsr_data)

    # --- 3. Display Results ---
    # In a real application, these three values would be passed as input 
    # to the user's pre-calibrated MLI (e.g., a Ridge Regression model) 
    # to predict their current affective valence.
    
    print("\n" + "="*45)
    print("--- TIHA 'TIHA-Lite' (Phase III) Demo Results ---")
    print(f"   1. HRV (Variability) Proxy:  {hrv:.4f} (ms)")
    print(f"   2. EDA (Activation) Proxy:   {eda:.1f} (peaks)")
    print(f"   3. S_gsr (Surprise) Proxy:   {s_gsr:.4f} (variance)")
    print("="*45 + "\n")
    print("In a real application, these three features would be fed into")
    print("the user's individually-calibrated MLI model to predict valence.")
    print("See 'TIHA_v6.0_Paper.pdf' for scientific details (R2=0.091).")

# Standard Python practice: only run the demo if the script is executed directly
if __name__ == "__main__":
    run_snp_demo()
