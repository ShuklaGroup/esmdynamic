# ESMDynamic Output Interpretation

This document explains how to interpret the outputs produced by `run_esmdynamic`.

The model generates predictions from three complementary heads:
- Dynamic contact classification
- Contact frequency (occupancy) regression
- Contact kinetics classification

Each sequence is processed independently, and outputs are organized into a structured directory for easy inspection and downstream analysis.

---

## Directory Structure

For a sequence with identifier `MY_PROTEIN`, the output directory will look like:

```
outputs/
└── MY_PROTEIN/
    ├── MY_PROTEIN.pdb
    ├── MY_PROTEIN_all_outputs.pt
    ├── dynamic/
    ├── frequency/
    ├── kinetics/
    ├── native/
    ├── dynamic_nonnative/
    └── native_nondynamic/
```

---

## Temperature Axis

All predictions are produced for five model conditions corresponding to the mdCATH training temperatures:

- 320 K
- 348 K
- 379 K
- 413 K
- 450 K

These should be interpreted as MD-simulated temperatures, not experimental conditions. We recommend starting analysis with 320K outputs, and use higher temperatures to explore increased flexibility or disorder.

---

## 1. Classification Head (`dynamic/`)

This head predicts whether a residue pair is dynamic, meaning it transitions between contact and non-contact states across the ensemble.

### Files

```
dynamic/
├── *_dynamic_prob_<TEMP>K.*
├── *_dynamic_pred_<TEMP>K.*
├── *_dynamic_confidence_<TEMP>K.*
```

### Outputs

- **dynamic_prob**  
  Continuous values in `[0, 1]` representing probability of being a dynamic contact.

- **dynamic_pred**  
  Binary map (threshold = `0.5`).

- **dynamic_confidence**  
  Per-residue confidence score. Usually, >0.9 is a stringent threshold. 

---

## 2. Frequency Head (`frequency/`)

This head predicts **contact occupancy** (fraction of time a contact is formed).

### Files

```
frequency/
├── *_frequency_pred_<TEMP>K.*
├── *_frequency_error_<TEMP>K.*
```

### Outputs

- **frequency_pred**  
  Values in `[0, 1]` representing fraction of ensemble in contact

- **frequency_error**  
  Predicted error (uncertainty proxy)

Note that a residue pair may have high probability of being classified as dynamic (classification head) even if it is relatively unstable (low occupancy).

---

## 3. Kinetics Head (`kinetics/`)

This head predicts coarse-grained kinetic classes for:

- On-time (or contact lifetime)  
- Off-time (or time to formation)  

### Files

```
kinetics/
├── *_kinetics_on_class_<TEMP>K.*
├── *_kinetics_off_class_<TEMP>K.*
├── *_kinetics_on_probabilities_<TEMP>K.npz
├── *_kinetics_off_probabilities_<TEMP>K.npz
├── *_kinetics_on_classes_<TEMP>K.txt
├── *_kinetics_off_classes_<TEMP>K.txt
├── *_kinetics_confidence_<TEMP>K.*
```

### Classes

**On-time**
```
0: always_on
1: 1–10 ns
2: 10–100 ns
3: 100–300 ns
4: >300 ns
5: never_on
```

**Off-time**
```
0: always_off
1: 1–10 ns
2: 10–100 ns
3: 100–300 ns
4: >300 ns
5: never_off
```

Predictions are categorical and they represent coarse timescales in the nanosecond regime (mdCATH simulation length). We recommend to use these predictions comparatively (e.g., fast vs slow contacts).  

---

## 4. Native Contact Outputs

Derived from the ESMFold structure.

### Folders

```
native/
dynamic_nonnative/
native_nondynamic/
```

### Meaning

- **native_contacts**  
  Static structure contacts (8 Å Cα cutoff)

- **dynamic_nonnative**  
  Dynamic contacts not present in static structure

- **native_nondynamic**  
  Stable contacts not predicted to be dynamic. We note that this category can be very sparse and even produce empty maps at high T.

---

## 5. PDB File

```
MY_PROTEIN.pdb
```

- ESMFold-predicted structure  
- Useful for mapping predicted contacts onto 3D coordinates  

---

## 6. Raw PyTorch Bundle

```
MY_PROTEIN_all_outputs.pt
```

Contains all processed outputs in a single dictionary.

### Example

```python
import torch

out = torch.load("MY_PROTEIN_all_outputs.pt")
print(out.keys())
```

Recommended for downstream programmatic uses.

---

## Multimer Inputs

Use `:` to separate chains:

```
A_SEQUENCE:B_SEQUENCE
```

- Internal linkers are removed from outputs  
- Chain boundaries appear as white lines in visualizations  
- Residue labels follow:

```
A-K15
B-F42
```

---

## Recommended Workflow

1. Inspect `dynamic_prob (320K)`  
2. Compare with `frequency_pred (320K)`  
3. Examine `kinetics` classes for key residue pairs  
4. Use confidence/error maps to filter unreliable regions  

---

## Loading Outputs in Python

### Text files

```python
import numpy as np

dyn = np.loadtxt("dynamic_prob_320K.txt")
freq = np.loadtxt("frequency_pred_320K.txt")
```

### Kinetics probabilities

```python
probs = np.load("kinetics_on_probabilities_320K.npz")
print(probs.files)
```

---

## Some Caveats

- Predictions describe contact-level dynamics, global geometric shifts may be difficult to infer  
- Kinetics are coarse-grained and trained on the nanosecond regime  
- Temperature outputs are modeled from MD simulations, not experimental conditions  

---

## Summary

| Head      | What it predicts         | Use case                        |
|-----------|------------------------|--------------------------------|
| Dynamic   | Contact variability     | Identify switching contacts     |
| Frequency | Contact occupancy       | Estimate stability              |
| Kinetics  | Timescale classes       | Rank fast vs slow contacts      |