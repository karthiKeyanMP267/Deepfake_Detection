# Final Phase-5 Training & Optimization Configuration

## 1. Parameter Breakdown

| Category | Parameter Count | Percentage of Total |
| :--- | :---: | :---: |
| **Total Model Parameters** | 166,440,678 | 100.00% |
| **Frozen Pretrained Encoders** | 157,682,658 | 94.74% |
| **Trainable Fusion & Head Params** | 8,758,020 | 5.26% |

---

## 2. Exact Hyperparameters & Setup

- **Optimization Paradigm**: Two-stage fast feature precomputation & in-memory cached training
- **Optimizer**: AdamW (`lr = 1e-4`, `weight_decay = 1e-5`, `eps = 1e-8`)
- **Loss Function**: `MultimodalLoss` (Focal Loss gamma=2.0, alpha=0.25 + InfoNCE Contrastive weight lambda=0.10)
- **Gradient Clipping**: `torch.nn.utils.clip_grad_norm_(max_norm=1.0)`
- **Learning Rate Scheduler**: `ReduceLROnPlateau(mode='max', factor=0.5, patience=1)`
- **Batch Size**: 32 (Training), 32 (Validation / Test)
- **Epochs Executed**: 5 (Best checkpoint selected at Epoch 4, Val F1 = 0.9722)
- **Early Stopping Patience**: 2 epochs
- **Class Imbalance Handling**: `WeightedRandomSampler` inversely proportional to class frequencies
- **Random Seed**: 42 (deterministic CuDNN backend)
- **Device**: NVIDIA GeForce RTX 3050 Laptop GPU (CUDA / FP32 feature tensors)
