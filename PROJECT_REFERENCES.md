# Academic References and Verified Bibliography

**Project Title:** Agent-Assisted Multimodal Deepfake Detection via Cross-Modal Consistency Learning and Adaptive Fusion  
**Scope:** Verified bibliographic details for the final-year project report  
**Status:** All entries verified against published academic literature and repository implementations.

---

## 1. Visual Deepfake Detection & Facial Feature Extraction

1. **Tan, M., & Le, Q. V.** (2019).  
   *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.*  
   **Venue:** International Conference on Machine Learning (ICML 2019), PMLR 97:6105–6114.  
   **URL/DOI:** [https://doi.org/10.48550/arXiv.1905.11946](https://doi.org/10.48550/arXiv.1905.11946)  
   **Project Relevance:** Used as Vision Stream Branch 1 (`EfficientNet-B0`, 1280-D output) to capture high-frequency spatial texture anomalies, blending boundaries, and facial warping artifacts (`models/efficientnet.py`).  
   **Verification Status:** VERIFIED

2. **Liu, Z., Ning, J., Cao, Y., Wei, Y., Zhang, Z., Lin, S., & Hu, H.** (2022).  
   *Video Swin Transformer.*  
   **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2022), pp. 3202–3211.  
   **URL/DOI:** [https://doi.org/10.1109/CVPR52688.2022.00322](https://doi.org/10.1109/CVPR52688.2022.00322)  
   **Project Relevance:** Used as Vision Stream Branch 2 (`Video Swin-Tiny`, 768-D output) with 3D shifted window spatio-temporal self-attention across 16 frames to capture inter-frame temporal discontinuities and motion jitter (`models/video_swin.py`).  
   **Verification Status:** VERIFIED

3. **He, K., Zhang, X., Ren, S., & Sun, J.** (2016).  
   *Deep Residual Learning for Image Recognition.*  
   **Venue:** IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2016), pp. 770–778.  
   **URL/DOI:** [https://doi.org/10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90)  
   **Project Relevance:** Serves as the backbone for Vision Stream Branch 3 (`Face/Mouth ResNet-18`, 512-D output) to extract specialized perioral and lip articulation dynamics (`models/face_mouth_encoder.py`).  
   **Verification Status:** VERIFIED

4. **Guo, J., Deng, J., Lattas, A., & Zafeiriou, S.** (2021).  
   *Sample and Computation Redistribution for Efficient Face Detection (SCRFD).*  
   **Venue:** arXiv preprint arXiv:2105.04714 / InsightFace Library.  
   **URL/DOI:** [https://doi.org/10.48550/arXiv.2105.04714](https://doi.org/10.48550/arXiv.2105.04714)  
   **Project Relevance:** Used in the preprocessing pipeline via InsightFace (`buffalo_l`) to robustly localize face bounding boxes with a 15% contextual boundary padding (`preprocessing/03_detect_faces.py`, `app/video_processor.py`).  
   **Verification Status:** VERIFIED

5. **Rossler, A., Cozzolino, D., Verdoliva, L., Riess, C., Thies, J., & Nießner, M.** (2019).  
   *FaceForensics++: Learning to Detect Manipulated Facial Images.*  
   **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV 2019), pp. 1–11.  
   **URL/DOI:** [https://doi.org/10.1109/ICCV.2019.00009](https://doi.org/10.1109/ICCV.2019.00009)  
   **Project Relevance:** Foundational visual deepfake benchmark establishing manipulation taxonomies (Deepfakes, Face2Face, FaceSwap, NeuralTextures) and compression baselines.  
   **Verification Status:** VERIFIED

---

## 2. Audio Deepfake Detection & Acoustic Modeling

6. **Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I.** (2023).  
   *Robust Speech Recognition via Large-Scale Weak Supervision.*  
   **Venue:** International Conference on Machine Learning (ICML 2023), PMLR 202:28492–28518.  
   **URL/DOI:** [https://doi.org/10.48550/arXiv.2212.04356](https://doi.org/10.48550/arXiv.2212.04356)  
   **Project Relevance:** Audio Stream Branch 4 uses the pretrained Whisper Base encoder (`whisper.load_model('base').encoder`, 512-D output) to capture high-level semantic speech structure and phonetic transitions (`models/whisper_encoder.py`).  
   **Verification Status:** VERIFIED

7. **Baevski, A., Zhou, Y., Mohamed, A., & Auli, M.** (2020).  
   *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations.*  
   **Venue:** Advances in Neural Information Processing Systems (NeurIPS 2020), Vol. 33, pp. 12449–12460.  
   **URL/DOI:** [https://doi.org/10.48550/arXiv.2006.11477](https://doi.org/10.48550/arXiv.2006.11477)  
   **Project Relevance:** Audio Stream Branch 5 uses `torchaudio.pipelines.WAV2VEC2_BASE` (768-D output) to extract acoustic waveform tokens and vocoder synthesis artifacts (`models/wav2vec2_encoder.py`).  
   **Verification Status:** VERIFIED

8. **Tak, H., Patino, J., Todisco, M., Nautsch, A., Evans, N., & Larcher, A.** (2021).  
   *End-to-End Anti-Spoofing with RawNet2.*  
   **Venue:** IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP 2021), pp. 6369–6373.  
   **URL/DOI:** [https://doi.org/10.1109/ICASSP39728.2021.9414234](https://doi.org/10.1109/ICASSP39728.2021.9414234)  
   **Project Relevance:** Relevant baseline for raw waveform audio deepfake detection and synthetic speech detection.  
   **Verification Status:** VERIFIED

---

## 3. Audio-Visual Consistency & Multimodal Deepfake Benchmarks

9. **Khalid, H., Tariq, S., Kim, M., & Woo, S. S.** (2021).  
   *FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset.*  
   **Venue:** Thirty-fifth Conference on Neural Information Processing Systems (NeurIPS 2021) Track on Datasets and Benchmarks.  
   **URL/DOI:** [https://doi.org/10.48550/arXiv.2108.05080](https://doi.org/10.48550/arXiv.2108.05080)  
   **Project Relevance:** Primary benchmark dataset for the entire project (`dataset/FakeAVCeleb/`), encompassing real videos from VoxCeleb2 and manipulations created with Faceswap, FSGAN, Wav2Lip, and SV2TTS voice cloning.  
   **Verification Status:** VERIFIED

10. **Chung, J. S., & Zisserman, A.** (2016).  
    *Out of Time: Automated Lip Sync in the Wild.*  
    **Venue:** Asian Conference on Computer Vision (ACCV 2016), Springer, Cham, pp. 251–263.  
    **URL/DOI:** [https://doi.org/10.1007/978-3-319-54427-4_16](https://doi.org/10.1007/978-3-319-54427-4_16)  
    **Project Relevance:** Establishes the SyncNet two-stream architecture for cross-modal lip synchronization that inspired the audio-visual consistency formulation $S_{AV} = \mathbf{e}_v^T \mathbf{e}_a$.  
    **Verification Status:** VERIFIED

11. **Haliassos, A., Vougioukas, K., Petridis, S., & Pantic, M.** (2021).  
    *Lips Don't Lie: A Generalisable Approach to Face Forgery Detection.*  
    **Venue:** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2021), pp. 5039–5049.  
    **URL/DOI:** [https://doi.org/10.1109/CVPR46437.2021.00499](https://doi.org/10.1109/CVPR46437.2021.00499)  
    **Project Relevance:** Demonstrates the high discriminative power of mouth ROI spatio-temporal features (LipForensics) for generalizable face forgery detection, motivating Branch 3 (`models/face_mouth_encoder.py`).  
    **Verification Status:** VERIFIED

12. **Zhou, Y., & Lim, S. N.** (2021).  
    *Joint Audio-Visual Deepfake Detection.*  
    **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV 2021), pp. 14800–14809.  
    **URL/DOI:** [https://doi.org/10.1109/ICCV48922.2021.01454](https://doi.org/10.1109/ICCV48922.2021.01454)  
    **Project Relevance:** Analyzes modality interactions and cross-modal attention mechanisms in joint audio-visual deepfake detection.  
    **Verification Status:** VERIFIED

---

## 4. Multimodal Fusion, Contrastive Learning & Optimization

13. **Oord, A. v. d., Li, Y., & Vinyals, O.** (2018).  
    *Representation Learning with Contrastive Predictive Coding.*  
    **Venue:** arXiv preprint arXiv:1807.03748.  
    **URL/DOI:** [https://doi.org/10.48550/arXiv.1807.03748](https://doi.org/10.48550/arXiv.1807.03748)  
    **Project Relevance:** Formulates the InfoNCE mutual information lower bound utilized in `models/contrastive_module.py` for cross-modal contrastive alignment with temperature $\tau = 0.07$.  
    **Verification Status:** VERIFIED

14. **Radford, A., et al.** (2021).  
    *Learning Transferable Visual Models From Natural Language Supervision (CLIP).*  
    **Venue:** International Conference on Machine Learning (ICML 2021), PMLR 139:8748–8763.  
    **URL/DOI:** [https://doi.org/10.48550/arXiv.2103.00020](https://doi.org/10.48550/arXiv.2103.00020)  
    **Project Relevance:** Inspires symmetric bidirectional cross-entropy over normalized embedding dot products ($\mathcal{L}_{i2a} + \mathcal{L}_{a2i}$)/2 in `models/contrastive_module.py`.  
    **Verification Status:** VERIFIED

15. **Lin, T. Y., Goyal, P., Girshick, R., He, K., & Dollár, P.** (2017).  
    *Focal Loss for Dense Object Detection.*  
    **Venue:** IEEE International Conference on Computer Vision (ICCV 2017), pp. 2980–2988.  
    **URL/DOI:** [https://doi.org/10.1109/ICCV.2017.324](https://doi.org/10.1109/ICCV.2017.324)  
    **Project Relevance:** Implemented in `training/losses.py` ($\gamma = 2.0, \alpha = 0.25$) to prevent majority-class gradient saturation under 97.7% class imbalance.  
    **Verification Status:** VERIFIED

16. **Loshchilov, I., & Hutter, F.** (2019).  
    *Decoupled Weight Decay Regularization (AdamW).*  
    **Venue:** International Conference on Learning Representations (ICLR 2019).  
    **URL/DOI:** [https://doi.org/10.48550/arXiv.1711.05101](https://doi.org/10.48550/arXiv.1711.05101)  
    **Project Relevance:** Primary optimizer across all phases (`lr = 1e-4, weight_decay = 1e-5`) in `training/train.py` and `run_phase5_full_3v3_experiment.py`.  
    **Verification Status:** VERIFIED

---

## 5. Model Calibration, Out-of-Distribution & Reliability

17. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q.** (2017).  
    *On Calibration of Modern Neural Networks.*  
    **Venue:** International Conference on Machine Learning (ICML 2017), PMLR 70:1321–1330.  
    **URL/DOI:** [https://doi.org/10.48550/arXiv.1706.04599](https://doi.org/10.48550/arXiv.1706.04599)  
    **Project Relevance:** Foundation for post-hoc Temperature Scaling implemented in Phase 8 (`experiments/phase8/calibration/temperature_scaling.py`), reducing ECE from 5.25% to 3.21% via learned $T = 0.7924$.  
    **Verification Status:** VERIFIED

18. **Lee, K., Lee, K., Lee, H., & Shin, J.** (2018).  
    *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.*  
    **Venue:** Advances in Neural Information Processing Systems (NeurIPS 2018), Vol. 31, pp. 7167–7177.  
    **URL/DOI:** [https://doi.org/10.48550/arXiv.1807.03888](https://doi.org/10.48550/arXiv.1807.03888)  
    **Project Relevance:** Inspires the centroid distance and variance-normalized Mahalanobis representation distance in `experiments/phase8/ood/reliability_detector.py` to identify domain-shifted screen recordings.  
    **Verification Status:** VERIFIED

19. **Hendrycks, D., & Gimpel, K.** (2017).  
    *A Baseline for Detecting Misclassified and Out-of-Distribution Examples in Neural Networks.*  
    **Venue:** International Conference on Learning Representations (ICLR 2017).  
    **URL/DOI:** [https://doi.org/10.48550/arXiv.1610.02136](https://doi.org/10.48550/arXiv.1610.02136)  
    **Project Relevance:** Provides theoretical background for softmax overconfidence on anomalous and out-of-distribution media.  
    **Verification Status:** VERIFIED

---

## 6. Generative Deepfake & Voice Synthesis Engines (Attack Methods)

20. **Prajwal, K. R., Mukhopadhyay, R., Namboodiri, V. P., & Jawahar, C. V.** (2020).  
    *A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild (Wav2Lip).*  
    **Venue:** ACM International Conference on Multimedia (ACM MM 2020), pp. 484–492.  
    **URL/DOI:** [https://doi.org/10.1145/3394171.3413532](https://doi.org/10.1145/3394171.3413532)  
    **Project Relevance:** Primary generative method used in FakeAVCeleb for audio-driven lip synchronization manipulation.  
    **Verification Status:** VERIFIED

21. **Nirkin, Y., Keller, Y., & Hassner, T.** (2019).  
    *FSGAN: Subject Agnostic Face Swapping and Reenactment.*  
    **Venue:** IEEE/CVF International Conference on Computer Vision (ICCV 2019), pp. 7184–7193.  
    **URL/DOI:** [https://doi.org/10.1109/ICCV.2019.00728](https://doi.org/10.1109/ICCV.2019.00728)  
    **Project Relevance:** Face swapping and reenactment method utilized in FakeAVCeleb video generation.  
    **Verification Status:** VERIFIED

22. **Jia, Y., Zhang, Y., Weiss, R., Wang, Q., Shen, J., Ren, F., Nguyen, P., Pang, R., Lopez Moreno, I., & Wu, Y.** (2018).  
    *Transfer Learning from Speaker Verification to Multispeaker Text-to-Speech Synthesis (SV2TTS).*  
    **Venue:** Advances in Neural Information Processing Systems (NeurIPS 2018), Vol. 31, pp. 4480–4490.  
    **URL/DOI:** [https://doi.org/10.48550/arXiv.1806.04558](https://doi.org/10.48550/arXiv.1806.04558)  
    **Project Relevance:** Real-Time Voice Cloning (RTVC / SV2TTS) voice synthesis engine used in FakeAVCeleb for cloned synthetic speech generation.  
    **Verification Status:** VERIFIED

---

## 7. Explainable AI & Agentic Forensic Architectures

23. **Ribeiro, M. T., Singh, S., & Guestrin, C.** (2016).  
    *"Why Should I Trust You?": Explaining the Predictions of Any Classifier.*  
    **Venue:** ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD 2016), pp. 1135–1144.  
    **URL/DOI:** [https://doi.org/10.1145/2939672.2939778](https://doi.org/10.1145/2939672.2939778)  
    **Project Relevance:** Foundational motivation for explainability and grounded evidence synthesis in forensic classification.  
    **Verification Status:** VERIFIED

24. **Touvron, H., et al.** (2023).  
    *Llama 2: Open Foundation and Fine-Tuned Chat Models.*  
    **Venue:** arXiv preprint arXiv:2307.09288.  
    **URL/DOI:** [https://doi.org/10.48550/arXiv.2307.09288](https://doi.org/10.48550/arXiv.2307.09288)  
    **Project Relevance:** Contextualizes LLM reasoning agents (LLaMA-3 family used via Groq in `agent/llm_forensic_agent.py`) for translating quantitative telemetry into natural language forensic findings.  
    **Verification Status:** VERIFIED
