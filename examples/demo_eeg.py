# =============================================================================
# TIHA Framework - Example (Phase II): EEG Application Demo
#
# File: demo_eeg.py
#
# Author: Leandro Antonio da Silva Pacheco (Original)
# Date: 2025-11-09
#
# OVERVIEW:
# This script serves as a high-level demonstration for a "Phase II" 
# application as described in the TIHA Framework 'Manual.md'. 
#
# It simulates capturing a window of EEG data (e.g., from a Muse, Emotiv, 
# or OpenBCI headset) and uses the 'tiha_core.py' library to calculate 
# the three validated proxies (features) from the canonical v5.1 EEG Model:
#
# 1. Φ (Phi)   - Integration (Alpha-band Correlation)
# 2. E (E)     - Entropy (Spectral Entropy)
# 3. S_eeg (S) - Surprise (Fz Channel Variance)
#
# This script is for demonstration only and uses simulated (random) data.
# It does *not* perform the MLI calibration, but rather shows how to 
# extract the features that would be *fed into* a calibrated model.
# =============================================================================

import tiha_core  # Imports the core TIHA proxy calculation functions
import numpy as np

def run_eeg_demo():
    """
    Main function to run the EEG (Phase II) demo simulation.
    """
    
    # --- 1. Hardware Simulation (Mock Data) ---
    # In a real-world application (Phase II), this section would be replaced 
    # by the SDK (Software Development Kit) of your specific EEG hardware.
    
    FS = 128                # Sampling Frequency (Hz). This matches the DEAP dataset.
    N_CHANNELS_EEG = 14     # Simulating a 14-channel headset (e.g., Emotiv Insight)
    DURATION_SECONDS = 5    # Simulating a 5-second data window for analysis
    
    print(f"Simulating {DURATION_SECONDS}s of mock EEG data ({N_CHANNELS_EEG} channels @ {FS}Hz)...")
    
    # Generate random data as a placeholder for real EEG signals.
    # Shape: (n_channels, n_samples)
    mock_eeg_data = np.random.rand(N_CHANNELS_EEG, DURATION_SECONDS * FS) 

    # Define the *hypothetical* channel index for the Fz electrode.
    # This index is entirely device-specific and must be mapped by the developer.
    # In this 14-channel example, we assume Fz is channel index 3.
    # (Note: The original v5.1 model used DEAP's channel 4 for Fz).
    IDX_FZ_HYPOTHETICAL = 3 
    
    # --- 2. Proxy Calculation (The TIHA 'Engine') ---
    # This section calls the validated functions from the 'tiha_core' library
    # to calculate the three components (features) of the v5.1 EEG model.
    # These features are the inputs for the Idiosyncratic Linear Model (MLI).
    print("Calculating TIHA v5.1 (EEG) proxies from mock data...")

    # Proxy 1: Φ (Phi) - Integration
    # The primary driver identified in the v5.1 study (R2=0.181, p<1.7e-11).
    # Measures the mean absolute correlation across all EEG channels in the 
    # alpha band (8-12Hz). Represents functional connectivity.
    phi = tiha_core.calcular_phi_alpha_real(mock_eeg_data, fs=FS)
    
    # Proxy 2: E (Entropy) - Spectral Entropy
    # Measures the complexity, or "chaos", of the EEG signal's power spectrum.
    # A high-entropy signal is more complex (like white noise), while a 
    # low-entropy signal is more ordered (like a pure sine wave).
    e = tiha_core.calcular_entropia_espectral_real(mock_eeg_data, fs=FS)
    
    # Proxy 3: S_eeg (Surprise) - EEG Surprise (Variance)
    # Measures the simple variance of a single, specific channel (Fz).
    # This serves as a proxy for signal "surprise" or unpredictability 
    # originating from the frontal lobe.
    s_eeg = tiha_core.calcular_surpresa_eeg_fz(
        mock_eeg_data, 
        channel_index=IDX_FZ_HYPOTHETICAL
    )
    
    # --- 3. Display Results ---
    # In a real application, these three values would be passed as input 
    # to the user's pre-calibrated MLI (e.g., a Ridge Regression model) 
    # to predict their current affective valence.
    
    print("\n" + "="*40)
    print("--- TIHA EEG (Phase II) Demo Results ---")
    print(f"   1. Φ (Integration) Driver: {phi:.4f}")
    print(f"   2. E (Spectral Entropy):   {e:.4f}")
    print(f"   3. S_eeg (Surprise @ Fz):  {s_eeg:.4f}")
    print("="*40 + "\n")
    print("In a real application, these three features would be fed into")
    print("the user's individually-calibrated MLI model to predict valence.")
    print("See 'Manual.md' and 'TIHA_v6.0_Paper.pdf' for scientific details.")

# Standard Python practice: only run the demo if the script is executed directly
if __name__ == "__main__":
    run_eeg_demo()
