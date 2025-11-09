## The Hybrid Affective Integration (TIHA) v6.0 Manifest

**Date:** November 8, 2025 **Author:** Leandro Antonio da Silva Pacheco **OSF Project DOI:** [https://doi.org/10.17605/OSF.IO/BJ2TS](https://doi.org/10.17605/OSF.IO/BJ2TS "null") **GitHub Repository:** `github.com/leandroantoniodasilvapacheco/TIHA-MLI-v6.0`

### Preamble: From Robustness to the Pivot (The End of EEG-Dependency)

TIHA v3.0 was a hypothesis of a "universal formula" (`Operator ⊙`) that was rigorously falsified (v3.5, LOSO).

This forced the "Ontological Pivot" (v4.0): the search for an **Idiosyncratic Linear Model (MLI)**.

The v4.6 model (4 components) was a predictive success ($R^2 \approx 0.229$), but a subsequent "elucidative failure" (v4.7) revealed high multicollinearity (VIF > 5). The $\Phi$ (Integration) and $C$ (Coupling) proxies were redundant.

The v5.1 model (3 components, $\Phi, E, S$) resolved this flaw, proving to be methodologically robust (VIF < 5) and formally idiosyncratic ($p > 0.104$).

This version, **v6.0**, represents the final **Comparative Validation**. It proves that the MLI framework is not restricted to EEG, successfully validating the "TIHA-Lite" (PNS) pivot.

### Principle 1: The Comparative Validation (CNS vs. PNS)

We confirm that the MLI framework functions in both signal domains (CNS and PNS). The "model competition" (for $V_H$) on the DEAP dataset produced significant results for both:

- **Model A (CNS v5.1):** $R^2 = 0.1810$ ($p < 1.7\text{e-}11$)
    
- **Model B (PNS v6.0):** $R^2 = 0.0914$ ($p < 4.9\text{e-}11$)
    

The pivot to wearables is viable, quantified by a 49.5% loss in predictive power.

### Principle 2: The Idiosyncratic Linear Model (MLI)

The `Operator ⊙` is refuted. We declare (v6.0): TIHA is an **Idiosyncratic Linear Model (MLI)**. The affective experience (IHA) correlates with a weighted linear combination of neural _or_ peripheral proxies:

$$IHA_i \approx (w_1 \cdot \text{Proxy}_1) + (w_2 \cdot \text{Proxy}_2) + (w_3 \cdot \text{Proxy}_3)$$

Crucially, the **v5.1 analysis formally proved the "Idiosyncratic Paradox."** A T-Test on the mean weights (betas) confirms they are not universally different from zero (p-minimum = 0.104 > 0.05). The formula _must_ be calibrated for each individual.

### Principle 3: The Canonical Proxy Sets

Our v4.6 proxies ($\Phi, E, C, S$) were flawed (VIF > 5). We declare (v6.0) the two validated proxy sets:

- **Set A (CNS v5.1):**
    
    - $\Phi$ (Alpha Integration)
        
    - $E$ (Spectral Entropy)
        
    - $S_{eeg}$ (Fz Surprise)
        
- **Set B (PNS v6.0):**
    
    - $HRV$ (Plethysmograph RMSSD)
        
    - $EDA$ (GSR Phasic Peaks)
        
    - $S_{gsr}$ (GSR Surprise)
        

The `C` (Fz-Pz Coupling) proxy is **obsolete**, as v4.8 proved that $\Phi$ is a superior and less redundant predictor.

### Principle 4: The Primacy of the Individual (The End of Generalization)

Our most important failure was the LOSO test (v3.5).

We declare (v6.0): TIHA is a theory of **intra-subject** variance. It does not describe a universal "human nature," but rather an individual "neural character." The MLI model explains, on average, **18.1%** (via EEG) or **9.1%** (via PNS/Wearable) of the change in an individual's valence.

### The Call (v6.0)

TIHA v3.x was a refuted hypothesis. v4.6 was a promising but methodologically flawed discovery.

Today, **TIHA v6.0 is an empirical, linear, and idiosyncratic framework**, validated with Ridge Regression, VIF tests, and the **formal proof of the Idiosyncratic Paradox**. Most importantly, we have proven the framework is viable for both **EEG (CNS)** and **peripheral signals (PNS)**.

The "hollow center" is filled by each brain's unique 3-component signature. We invite the community to replicate our results (v6.0) and to build Phases II and III upon this validated foundation.
