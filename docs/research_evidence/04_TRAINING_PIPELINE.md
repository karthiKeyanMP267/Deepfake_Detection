# Training Pipeline: Optimization, Losses, & Imbalance Strategy

## 1. Experimental Training Setup
The training pipeline is engineered for rapid convergence, numerical stability, and robust handling of extreme dataset class skewness (97.7% Fake / 2.3% Real).

```
===================================================================================================
HYPERPARAMETER               CONFIGURATION / VALUE              SOURCE CODE REFERENCE
===================================================================================================
Optimizer                    AdamW (beta1=0.9, beta2=0.999)      training/train.py:207, 307-317
Initial Learning Rate        1.0e-4 (Fusion / Heads)             training/train.py:208, 308
Backbone Fine-Tune LR        1.0e-5 (Phase 4 Group 2)            training/train.py:309
Weight Decay                 1.0e-5                              training/train.py:209, 308
Learning Rate Scheduler      ReduceLROnPlateau (mode=max, f=0.5) training/train.py:210, 323-328
Scheduler Patience / Factor  Patience = 3 epochs, Factor = 0.5   training/train.py:326-327
Batch Size                   8 (4 in Phase 4 Partial FT)         training/train.py:211
Maximum Epochs               15 epochs                           training/train.py:212
Early Stopping               Patience = 5 epochs, delta = 1e-4   training/train.py:39-55
Loss Function                Multimodal Focal + InfoNCE Loss     training/train.py:22, 170-173
Focal Loss Parameters        gamma = 2.0, alpha = 0.25           training/losses.py:10, 54
Contrastive Loss Weight      lambda_contrastive = 0.20           training/losses.py:45, 85
Classification Loss Weight   lambda_classification = 1.00        training/losses.py:44, 81
Sampling Strategy            WeightedRandomSampler (Enabled)     training/train.py:71, 123
Mixed Precision (AMP)        torch.cuda.amp.GradScaler (FP16)    training/train.py:331-332, 365-380
Random Seed                  42 (Deterministic cuDNN)            training/train.py:26-33
Feature Caching Protocol     Offline Extraction (.pt Tensors)    verify_3v3_cached_features.py
===================================================================================================
```

---

## 2. Loss Function Formulations

### A. Multimodal Composite Loss
- **Source File**: `training/losses.py` (`MultimodalLoss`)
- The overall optimization objective combines classification loss and cross-modal contrastive alignment:
  $$\mathcal{L}_{	ext{total}} = \lambda_{	ext{cls}} \mathcal{L}_{	ext{classification}} + \lambda_{	ext{cont}} \mathcal{L}_{	ext{contrastive}}$$
  where $\lambda_{	ext{cls}} = 1.0$ and $\lambda_{	ext{cont}} = 0.2$.

### B. Focal Loss for Severe Class Imbalance
- **Source File**: `training/losses.py` (`FocalLoss`)
- Standard Cross-Entropy produces overwhelming gradient updates from easy majority (Fake) examples. Focal Loss dynamically down-weights well-classified examples with focusing parameter $\gamma = 2.0$:
  $$	ext{FL}(p_t) = -lpha_t (1 - p_t)^\gamma \log(p_t)$$
  where probability of the true class $p_t$ is defined as:
  $$p_t = egin{cases} p & 	ext{if } y = 1 	ext{ (Fake)} \ 1 - p & 	ext{if } y = 0 	ext{ (Real)} \end{cases}$$
  and $lpha_t$ is the weighting factor: $lpha_t = lpha = 0.25$ for Fake, and $lpha_t = 1 - lpha = 0.75$ for Real.

### C. Cross-Modal InfoNCE Contrastive Loss
- **Source File**: `models/contrastive_module.py` (`ContrastiveModule`)
- Aligns positive visual-acoustic embedding pairs $(\mathbf{e}_{v,i}, \mathbf{e}_{a,i})$ from the same video while repelling negative pairs across the minibatch of size $B$:
  $$\mathcal{L}_{	ext{contrastive}} = rac{1}{2} (\mathcal{L}_{i2a} + \mathcal{L}_{a2i})$$
  $$\mathcal{L}_{i2a} = -rac{1}{B} \sum_{i=1}^B \log rac{\exp(\mathbf{e}_{v,i}^T \mathbf{e}_{a,i} / 	au)}{\sum_{j=1}^B \exp(\mathbf{e}_{v,i}^T \mathbf{e}_{a,j} / 	au)}, \quad \mathcal{L}_{a2i} = -rac{1}{B} \sum_{i=1}^B \log rac{\exp(\mathbf{e}_{a,i}^T \mathbf{e}_{v,i} / 	au)}{\sum_{j=1}^B \exp(\mathbf{e}_{a,i}^T \mathbf{e}_{v,j} / 	au)}$$
  with temperature $	au = 0.07$.

---

## 3. Class Imbalance Handling: WeightedRandomSampler
- **Source File**: `training/dataloader.py` (`get_train_loader`)
- In the training set of 11,266 samples, class distribution is:
  - Real ($y=0$): 259 samples (2.30%)
  - Fake ($y=1$): 11,007 samples (97.70%)
- Sampling weights are computed per sample:
  $$w_i = rac{1}{N_{y_i}} \implies w_{	ext{real}} = rac{1}{259} pprox 0.003861, \quad w_{	ext{fake}} = rac{1}{11007} pprox 0.00009085$$
- `WeightedRandomSampler(weights, num_samples=len(data), replacement=True)` ensures each batch presents approximately 50% Real and 50% Fake samples to the network during training, preventing gradient saturation.

---

## 4. Parameter Freezing & Fine-Tuning Strategies

Across the experimental phases, three distinct freezing strategies were implemented and evaluated:

```
======================================================================================================
STRATEGY                COMPONENTS FROZEN                   COMPONENTS TRAINABLE         PARAMETERS
======================================================================================================
1. Full Freezing        All 6 Foundation Encoders           Aggregators, Projectors,      9,212,612
   (Phase 3 & Phase 5)  (EfficientNet, Video Swin, ResNet,  Adaptive Fusion, Classifier  (Trainable)
                        Whisper, Wav2Vec2, SpecCNN)         (Total: 166.44M params)
------------------------------------------------------------------------------------------------------
2. Partial Fine-Tuning  Lower layers of backbones           Top layers of backbones:      30,500,644
   (Phase 4)            (EffNet 0..6, Swin 0..5,            - EffNet features[7..8]      (Trainable)
                        Whisper conv+0..3, Wav2Vec2 0..7)   - Swin stage 6 + LayerNorm
                                                            - Whisper blocks 4..5
                                                            - Wav2Vec2 layers 8..11
                                                            + All Aggregators & Fusion
------------------------------------------------------------------------------------------------------
3. Unweighted Baseline  All Encoders Frozen                 Only Fusion + Classifier      4,510,722
   (Phase 1 Baseline)   (EfficientNet + Whisper)                                         (Trainable)
======================================================================================================
```

### Optimizer Parameter Groups (Partial Fine-Tuning):
In `training/train.py` (lines 307-310), parameter groups are configured with differentiated learning rates:
- **Group 1** (Aggregators, Projectors, Fusion, Classifier): `lr = 1e-4`, `weight_decay = 1e-5`
- **Group 2** (Unfrozen Backbone Layers): `lr = 1e-5`, `weight_decay = 1e-5` (10x smaller learning rate to prevent catastrophic forgetting).

---

## 5. Offline Feature Caching Protocol
To maximize training throughput on resource-constrained hardware:
1. All 6 foundation encoders extract 1D feature embeddings once across the entire dataset.
2. Extracted tensors are saved to disk:
   - `train_features.pt` (11,266 samples)
   - `val_features.pt` (2,414 samples)
   - `test_features.pt` (604 samples)
3. During Phase 5 training, dataloaders stream cached 1D tensors directly to the GPU, reducing epoch execution time from ~155 minutes to **8.2 seconds per epoch** without altering optimization mathematics.
