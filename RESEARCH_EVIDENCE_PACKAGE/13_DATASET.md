# Dataset Manifest, Distribution, & Imbalance Analysis

## 1. Dataset Identification & Source
- **Benchmark Dataset**: **FakeAVCeleb** (Multimodal Deepfake Detection Benchmark)
- **Primary Source Directory**: `dataset/FakeAVCeleb/` and `processed/`
- **Metadata Generator**: `preprocessing/05_create_metadata.py`
- **Container Structure**: Standard video recordings organized into 4 manipulation categories:
  - `RV_RA`: Real Video, Real Audio (Authentic baseline)
  - `FV_RA`: Fake Video, Real Audio (Facial manipulation: Faceswap, FSGAN, Wav2Lip visual)
  - `RV_FA`: Real Video, Fake Audio (Acoustic manipulation: SV2TTS cloned speech)
  - `FV_FA`: Fake Video, Fake Audio (Dual multimodal manipulation)

---

## 2. Dataset Distribution & Splitting Methodology

Splitting was executed via stratified sampling using `sklearn.model_selection.train_test_split` with `random_state=42` and `stratify=df["label"]`:
- **Training Set Split**: 70.0% of total manifest (`processed/train.csv`)
- **Validation Set Split**: 15.0% of total manifest (`processed/val.csv`)
- **Testing Set Split**: 15.0% of total manifest (`processed/test.csv` and `test_manifest.csv`)

```
========================================================================================================================
SUBSET                 TOTAL SAMPLES    AUTHENTIC (REAL)     MANIPULATED (FAKE)   CLASS SKEW (FAKE %)  SOURCE FILE
========================================================================================================================
Training Set           11,266           259 (2.30%)          11,007 (97.70%)      97.70% Fake          processed/train.csv
Validation Set          2,414            55 (2.28%)           2,359 (97.72%)      97.72% Fake          processed/val.csv
Testing Set               604            19 (3.15%)             585 (96.85%)      96.85% Fake          test_manifest.csv
------------------------------------------------------------------------------------------------------------------------
TOTAL BENCHMARK        14,284           333 (2.33%)          13,951 (97.67%)      97.67% Fake          processed/metadata.csv
========================================================================================================================
```

---

## 3. Severe Class Imbalance & Mitigation
- **Imbalance Ratio**: Approximately **42.5 to 1** (42.5 Fake samples for every 1 Real sample).
- **Failure Mode Without Mitigation**: Standard unweighted models collapsed (Phase 1 Baseline: 7.04% accuracy; Phase 2A Class Weight: 0.00% Real Recall).
- **Implemented Mitigation**:
  - `WeightedRandomSampler` dynamically balances batch distributions to ~50% Real / 50% Fake during training (`training/dataloader.py`).
  - `FocalLoss` with focusing parameter $\gamma = 2.0$ down-weights easy majority-class gradients (`training/losses.py`).
