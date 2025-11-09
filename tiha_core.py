# =============================================================================
# TIHA Core Library (tiha_core.py)
#
# Author: Leandro Antonio da Silva Pacheco (Original)
# Date: 2025-11-09
#
# OVERVIEW:
# This is the central library for the TIHA v6.0 framework.
# It contains the 6 canonical, validated proxy calculation functions
# for both EEG (CNS) and Peripheral (PNS) signals.
#
# All analysis scripts (v5.1, v6.0) and demos (Phase II, III) 
# import their functions from this single, authoritative source.
#
# This file is intended for import, not direct execution.
# =============================================================================

import numpy as np
from scipy.stats import entropy
from scipy.signal import welch, butter, filtfilt, coherence, find_peaks

# --- EEG (CNS) Component Functions ---
# (Validated in v5.1 Paper: R2=0.181, VIF<2.74, p_paradox>0.1)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Standard Butterworth bandpass filter.
    
    Args:
        data (np.array): Input signal (Channels, Samples).
        lowcut (float): Low frequency cutoff.
        highcut (float): High frequency cutoff.
        fs (int): Sampling frequency.
        order (int): Filter order.
        
    Returns:
        np.array: Filtered data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    # Apply filter along the last axis (time)
    return filtfilt(b, a, data, axis=-1)

def calcular_phi_alpha_real(eeg_data, fs=128):
    """
    Proxy Φ (Phi - Integration): Mean absolute correlation in the Alpha band.
    
    This is the primary driver proxy identified in the v5.1 analysis.
    It measures the global integration of the 8-12 Hz rhythm.
    
    Args:
        eeg_data (np.array): EEG data (Channels, Samples).
        fs (int): Sampling frequency.
        
    Returns:
        float: Proxy Φ value.
    """
    try:
        # Ensure data is 2D (Channels, Samples)
        if eeg_data.ndim == 1: 
            eeg_data = eeg_data.reshape(1, -1)
        
        eeg_filtered = butter_bandpass_filter(eeg_data, 8.0, 12.0, fs)
        
        # Guard for single-channel data (correlation requires >= 2)
        if eeg_filtered.shape[0] < 2: 
            return 0.0 
        
        corr_matrix = np.corrcoef(eeg_filtered)
        # Get indices for the upper triangle of the matrix (k=1 to exclude diagonal)
        indices_triu = np.triu_indices(corr_matrix.shape[0], k=1)
        return np.mean(np.abs(corr_matrix[indices_triu]))
    except Exception: 
        return 0.0

def calcular_entropia_espectral_real(eeg_data, fs=128):
    """
    Proxy E (Entropy): Mean Spectral Entropy (Welch's method).
    
    This proxy measures the "unpredictability" or "complexity" of the
    power spectral density (PSD) across all channels.
    Validated in v5.1.
    
    Args:
        eeg_data (np.array): EEG data (Channels, Samples).
        fs (int): Sampling frequency.
        
    Returns:
        float: Proxy E value.
    """
    try:
        if eeg_data.ndim == 1: 
            eeg_data = eeg_data.reshape(1, -1)
        
        spectral_entropies = []
        for i in range(eeg_data.shape[0]):
            signal = eeg_data[i, :]
            # Use 2-second windows (fs*2) for Welch, matching DEAP analysis
            freqs, psd = welch(signal, fs, nperseg=fs*2)
            psd_norm = psd / np.sum(psd)
            se = entropy(psd_norm)
            if np.isfinite(se): 
                spectral_entropies.append(se)
        
        return np.mean(spectral_entropies) if spectral_entropies else 0.0
    except Exception:
        return 0.0

def calcular_surpresa_eeg_fz(eeg_data, channel_index=4):
    """
    Proxy S_eeg (Surprise): Variance of the Fz signal.
    
    This proxy measures the "surprise" or "energy" of the Fz channel,
    which is a key hub in affective processing.
    Validated in v5.1.
    
    Args:
        eeg_data (np.array): EEG data (Channels, Samples).
        channel_index (int): Index of the Fz channel (DEAP default is 4).
        
    Returns:
        float: Proxy S_eeg value.
    """
    try:
        if eeg_data.ndim == 1: 
            eeg_data = eeg_data.reshape(1, -1)
        
        # Guard against index out of bounds
        if channel_index >= eeg_data.shape[0]:
            channel_index = 0
            
        signal_fz = eeg_data[channel_index, :]
        return np.var(signal_fz)
    except Exception:
        return 0.0

# --- PNS (Peripheral) Component Functions ---
# (Validated in v6.0 Paper: R2=0.091, VIF<1.1)

def calcular_hrv_rmssd(pleth_or_ecg_data, fs=128):
    """
    Proxy HRV (Variability): RMSSD from Plethysmograph (PPG) or ECG.
    
    This proxy is the core of the "TIHA-Lite" v6.0 model.
    It calculates the Root Mean Square of Successive Differences (RMSSD)
    from detected signal peaks (heartbeats).
    
    Args:
        pleth_or_ecg_data (np.array): 1D PNS signal (PPG or ECG).
        fs (int): Sampling frequency.
        
    Returns:
        float: RMSSD value (in ms).
    """
    try:
        signal_pns = pleth_or_ecg_data
        if signal_pns.ndim > 1: 
            signal_pns = signal_pns.flatten()

        # Find peaks (heartbeats)
        # 'distance' = 0.5s * fs, ensures min 0.5s between beats (max 120 bpm)
        peaks, _ = find_peaks(signal_pns, distance=fs*0.5, height=np.mean(signal_pns))
        
        # Guard: Need at least 5 peaks for a stable HRV reading
        if len(peaks) < 5: 
            return 0.0 
        
        # Calculate RR intervals in seconds
        intervals_sec = np.diff(peaks) / fs
        # Calculate successive differences in milliseconds
        diffs_ms = np.diff(intervals_sec) * 1000 
        rmssd = np.sqrt(np.mean(diffs_ms ** 2))
        
        return rmssd if np.isfinite(rmssd) else 0.0
    except Exception:
        return 0.0

def calcular_eda_phasic_peaks(gsr_data, fs=128):
    """
    Proxy EDA (Activation): Count of phasic (SCR) peaks in GSR signal.
    
    This proxy measures sympathetic arousal by separating the rapid (phasic)
    skin conductance responses (SCRs) from the slow (tonic) component.
    Validated in v6.0.
    
    Args:
        gsr_data (np.array): 1D GSR signal.
        fs (int): Sampling frequency.
        
    Returns:
        float: Count of EDA peaks.
    """
    try:
        signal_pns = gsr_data
        if signal_pns.ndim > 1: 
            signal_pns = signal_pns.flatten()
        
        # 1. Get tonic (slow) component with a 1Hz low-pass filter
        nyq = 0.5 * fs; 
        b, a = butter(4, 1.0 / nyq, btype='low')
        tonic_component = filtfilt(b, a, signal_pns)
        
        # 2. Get phasic (fast) component by subtraction
        phasic_component = signal_pns - tonic_component
        
        # 3. Count peaks (SCRs) in the phasic signal
        # 'height=0.01' is a standard minimum threshold for a valid SCR
        peaks, _ = find_peaks(phasic_component, height=0.01, distance=fs)
        return float(len(peaks))
    except Exception:
        return 0.0

def calcular_surpresa_gsr(gsr_data):
    """
    Proxy S_gsr (Surprise): Variance of the raw GSR signal.
    
    This is the PNS analogue to S_eeg, measuring the total
    signal energy or "surprise" in the peripheral system.
    Validated in v6.0.
    
    Args:
        gsr_data (np.array): 1D GSR signal.
        
    Returns:
        float: Proxy S_gsr value.
    """
    try:
        signal_pns = gsr_data
        if signal_pns.ndim > 1: 
            signal_pns = signal_pns.flatten()
            
        return np.var(signal_pns)
    except Exception:
        return 0.0

# --- Deprecated Proxy (v4.6 / v4.7) ---

def calcular_acoplamento_fronto_parietal_real(eeg_data, fz_idx=4, pz_idx=16, fs=128):
    """ 
    **DEPRECATED in v4.8**
    Proxy C (Coupling): Fronto-parietal Coherence (Fz-Pz) in Alpha.
    
    This proxy was used in the v4.6 model but was **deprecated**.
    The v4.7 analysis proved it was highly collinear with Proxy Φ 
    (VIF > 5), making the model unstable. It was replaced by Φ (Integration)
    in the final v5.1 model.
    
    It is included here only for historical/replication purposes.
    """
    try:
        signal_fz = eeg_data[fz_idx, :]
        signal_pz = eeg_data[pz_idx, :]
        f, Cxy = coherence(signal_fz, signal_pz, fs=fs, nperseg=fs*2)
        alpha_band = (f >= 8) & (f <= 12)
        return np.mean(Cxy[alpha_band]) if np.any(alpha_band) else 0.0
    except Exception:
        return 0.0

print("TIHA Core Library (tiha_core.py) loaded successfully.")
