# 35 Comprehensive Viva & Defense Questions with Model Answers

### 1. Why is multimodal detection necessary for deepfakes?
**Answer**: Modern generative models can produce near-flawless single-frame images, but creating synchronized, acoustically coherent, and spatio-temporally consistent video and audio simultaneously is exponentially harder. Multimodal detectors capture cross-modal inconsistencies that unimodal models miss.

### 2. Why use three distinct vision models instead of just one?
**Answer**: Deepfake artifacts manifest across three distinct visual domains: spatial textures (EfficientNet-B0), temporal inter-frame coherence (Video Swin Transformer), and facial/lip articulation dynamics (Face/Mouth ResNet-18). Each encoder specializes in a distinct artifact type.

### 3. Why use three distinct audio models?
**Answer**: Synthetic speech contains artifacts at different abstraction levels: high-level phonetic/semantic structure (Whisper), low-level raw waveform vocoder glitches (Wav2Vec2), and time-frequency spectral anomalies (Spectrogram CNN).

### 4. What is the role of the Video Swin Transformer?
**Answer**: Video Swin utilizes 3D shifted window self-attention across 16 video frames to model temporal motion consistency, catching inter-frame flickering, unnatural eye blinks, and head pose warping.

### 5. Why incorporate a dedicated Face/Mouth encoder?
**Answer**: Lip-sync manipulations and facial reenactment specifically distort mouth contours. The Face/Mouth ResNet-18 processes cropped mouth ROIs to capture articulation dynamics.

### 6. What is the function of Whisper in your pipeline?
**Answer**: Whisper provides pre-trained speech-to-text representations that encode phonetic semantics, identifying unnatural prosody or word-level synthesis flaws.

### 7. What is the function of Wav2Vec2?
**Answer**: Wav2Vec2 is self-supervised on raw 16kHz waveforms, making it exceptionally sensitive to micro-acoustic neural vocoder artifacts and phase discontinuities.

### 8. What does the Spectrogram CNN detect?
**Answer**: It converts raw audio into 128-band log-Mel spectrograms and applies 2D convolutions to detect spectral cutoff, unnatural harmonics, and frequency domain anomalies.

### 9. What is Adaptive Fusion and how does it work?
**Answer**: Adaptive Fusion uses a lightweight attention neural network that takes the concatenated 256-D vision and 256-D audio embeddings and computes dynamic softmax weights (w_vision, w_audio). The final embedding is w_v * e_v + w_a * e_a.

### 10. How are the modality fusion weights calculated mathematically?
**Answer**: w = Softmax(W_2 * ReLU(W_1 * [e_v; e_a] + b_1) + b_2).

### 11. What happens if a video has missing audio?
**Answer**: When `audio_available = False`, the system skips the 3 audio encoders, and the fusion module deterministically routes w_vision = 1.0 and w_audio = 0.0, producing robust vision-only predictions without throwing exceptions.

### 12. How does the system handle silent audio?
**Answer**: Silent audio (zero waveform) is detected, audio encoders are bypassed, and the vision-only fallback is executed with finite outputs.

### 13. How does the system handle corrupted or unreadable audio?
**Answer**: The dataloader safely catches audio decoding exceptions, sets `audio_available = False`, and executes vision-only inference.

### 14. What is the difference between deepfake detection and dubbing detection?
**Answer**: Deepfake detection classifies visual manipulation (e.g. face swaps, GAN synthesis). Dubbing detection evaluates whether genuine speech has been replaced or translated. A dubbed video can be authentic video paired with legitimate translated audio.

### 15. How is audio-visual consistency computed?
**Answer**: By computing the cosine similarity between the L2-normalized 256-D vision embedding and 256-D audio embedding: S_AV = dot(e_v, e_a) / (||e_v|| * ||e_a||) in [-1, 1].

### 16. Why does normal audio have positive consistency while mismatched audio drops to zero?
**Answer**: The InfoNCE contrastive loss trains the projection heads to align synchronized visual and acoustic representations in a shared embedding space. Swapping audio decouples this correlation, dropping cosine similarity from +0.2618 to -0.0060.

### 17. Which layers of the architecture were frozen during Phase 5 training?
**Answer**: All six pretrained encoders (EfficientNet-B0, Video Swin, Face/Mouth ResNet-18, Whisper, Wav2Vec2, and Spectrogram CNN) totaling 157.7M parameters were strictly frozen (`requires_grad = False`).

### 18. Why freeze the pretrained encoders?
**Answer**: Pretrained foundation models have already learned robust general representations. Freezing prevents catastrophic forgetting, prevents overfitting on small datasets, and dramatically accelerates training speed.

### 19. How many parameters are trainable in your final model?
**Answer**: 8,758,020 parameters (5.26% of the 166.4M total parameters), comprising the aggregation layers, projection heads, adaptive fusion network, and MLP classifier.

### 20. What is feature caching and why was it implemented?
**Answer**: Feature caching precomputes the frozen encoder representations once and stores them in `.pt` tensors. This reduced training epoch time from 55 minutes to 8 seconds (a 400x speedup).

### 21. What loss function was used to train the multimodal model?
**Answer**: `MultimodalLoss`, which combines Focal Loss (gamma=2.0, alpha=0.25) on classification logits with InfoNCE Contrastive Loss (lambda=0.10, tau=0.07) on valid cross-modal embedding pairs.

### 22. What is the class imbalance in the dataset?
**Answer**: Severe imbalance: 97.7% Fake samples and 2.3% Real samples (11,007 Fake vs 259 Real in train set; 585 Fake vs 19 Real in test set).

### 23. Why did standard unweighted training fail initially?
**Answer**: The baseline model achieved 97.7% accuracy by predicting "Fake" for all samples, but its Real recall was only 4.8%, collapsing the Real F1 score to 4.75%.

### 24. Why did WeightedRandomSampler outperform Class Weighting?
**Answer**: `WeightedRandomSampler` balances each mini-batch uniformly with 50% Real and 50% Fake samples, maintaining consistent gradient magnitudes without gradient explosion.

### 25. Why is Accuracy a misleading metric in deepfake detection?
**Answer**: A trivial majority-class classifier achieves 97.7% accuracy on this dataset while having zero utility on real videos. ROC-AUC, Balanced Accuracy, and Minority Real F1 are required for objective evaluation.

### 26. What was the impact of adding the Video Swin Transformer in Phase 3?
**Answer**: It increased ROC-AUC from 70.10% to 98.80% (+28.70 percentage points) and boosted Real F1 from 10.62% to 68.97% by capturing spatio-temporal dynamics.

### 27. What were the final Phase 5 Full 3V3 test metrics?
**Answer**: Accuracy = 97.52%, Precision = 99.65%, Recall = 97.78%, F1 = 98.71%, ROC-AUC = 99.02%, Balanced Accuracy = 93.63%, Real F1 = 69.39%.

### 28. What was the cause of the NaN issue encountered earlier and how was it resolved?
**Answer**: Automatic Mixed Precision (AMP FP16) caused subnormal floating point underflow during log-Mel spectrogram calculation (`log(mel + 1e-6)`). It was resolved by executing the Spectrogram CNN in full FP32 with numerical clamping and gradient clipping.

### 29. What is the difference between feature fusion latency and end-to-end latency?
**Answer**: Feature fusion latency (0.20 ms/sample) measures only the neural MLP and attention pass on GPU. End-to-end latency (~350 ms/sample) includes OpenCV video decoding, frame extraction, landmark cropping, and 6 deep encoder forward passes.

### 30. Does low cross-modal consistency prove a deepfake?
**Answer**: No. Low consistency only proves cross-modal misalignment (such as dubbing or speech replacement). The system treats it as forensic evidence to be interpreted alongside visual deepfake probabilities.

### 31. What is the fixed decision threshold used for evaluation?
**Answer**: Fixed threshold of 0.5 on the softmax probability of the Fake class, preserving strict evaluation integrity without threshold cheating.

### 32. How was independent mathematical verification conducted?
**Answer**: All metrics were independently recomputed from raw `final_predictions.csv` using scikit-learn, confirming mathematical exactness within 1e-6 tolerance.

### 33. What are the key limitations of this study?
**Answer**: 1) Limited Real samples (19 in test set); 2) Embedding alignment is not fine-grained phoneme lip-sync; 3) Mismatched audio was evaluated on controlled speaker swaps rather than an in-the-wild dubbed dataset.

### 34. What is the primary novelty of this project?
**Answer**: The combination of a 6-encoder multimodal architecture, resilient vision-only fallback, dynamic attention fusion, and dual-output forensic disentanglement between deepfake prediction and cross-modal synchronization.

### 35. What are the future research directions?
**Answer**: 1) Collecting balanced in-the-wild datasets; 2) Implementing token-level phoneme-viseme cross-attention; 3) Optimizing multi-encoder inference via model distillation or TensorRT.
