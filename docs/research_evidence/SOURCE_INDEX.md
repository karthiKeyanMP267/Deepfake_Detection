# Master Source Code & Evidence Index

This index provides 100% verifiable mapping between every research claim in the paper and its corresponding source code file, class, function, and line number.

```
========================================================================================================================
RESEARCH CLAIM / METRIC               SOURCE CODE FILE                                  FUNCTION / CLASS / SECTION
========================================================================================================================
EfficientNet-B0 Spatial Branch        models/efficientnet.py                            EfficientNetEncoder.forward:65-111
Video Swin 3D Temporal Branch         models/video_swin.py                              VideoSwinEncoder.forward:76-98
Face/Mouth ResNet-18 Perioral Branch  models/face_mouth_encoder.py                      FaceMouthEncoder.forward:70-103
Whisper Base.en Semantic Audio Branch models/whisper_encoder.py                         WhisperEncoder.forward:86-126
Wav2Vec2 Base Acoustic Audio Branch   models/wav2vec2_encoder.py                        Wav2Vec2Encoder.forward:67-97
Spectrogram 2D CNN Harmonic Branch    models/spectrogram_cnn.py                         SpectrogramCNNEncoder.forward:94-130
Vision Feature Aggregator (2560->1280)models/vision_feature_aggregation.py              VisionFeatureAggregation:42-45, 84-88
Audio Feature Aggregator (1536->512)  models/audio_feature_aggregation.py               AudioFeatureAggregation:42-45, 84-88
Vision Projection Head (1280->256)    models/projection_heads.py                        VisionProjectionHead:64-83
Audio Projection Head (512->256)      models/projection_heads.py                        AudioProjectionHead:84-103
Adaptive Multimodal Attention Fusion  models/adaptive_fusion.py                         AdaptiveFusion.forward:71-134
Missing Audio Dynamic Fallback        models/adaptive_fusion.py                         AdaptiveFusion.forward:77-89
Cross-Modal InfoNCE Contrastive Loss  models/contrastive_module.py                      ContrastiveModule.forward:44-98
Deepfake Classifier MLP Head          models/classifier.py                              DeepfakeClassifier.forward:67-71
Audio Availability Detection          preprocessing/audio_detector.py                   AudioDetector.check_waveform:62-90
InsightFace SCRFD Face Extraction     preprocessing/03_detect_faces.py                  initialize_detector:36-70
Mouth ROI Bilinear Extraction         models/face_mouth_encoder.py                      FaceMouthEncoder.forward:87-95
FFmpeg Monophonic Audio Extraction    preprocessing/02_extract_audio.py                 extract_audio:23-48
WeightedRandomSampler Implementation  training/dataloader.py                            get_train_loader:203-248
Focal Loss Formulation                training/losses.py                                FocalLoss.forward:17-34
Multimodal Loss (1.0 Cls + 0.2 Cont)  training/losses.py                                MultimodalLoss.forward:64-98
Partial Fine-Tuning Parameter Groups  training/train.py                                 Trainer.__init__:285-317
Metrics Tracking & Confusion Matrix   training/metrics.py                               MetricTracker.compute:72-170
Deterministic Forensic Reasoning      agent/reasoning.py                                ForensicReasoningEngine:14-470
12-Criterion Domain Reliability Layer agent/reasoning.py                                evaluate_domain_reliability:150-282
Audio-Visual Consistency Formulation  agent/reasoning.py                                evaluate_cross_modal...:109-147
Automated Markdown Report Generator   agent/report_generator.py                         ForensicReportGenerator:18-180
Autonomous Forensic Agent Orchestratoragent/forensic_agent.py                           ForensicAgent.process_evidence:25-116
Final 3V3 Test Metrics (97.52%, 99.02)experiments/05_full_3v3/metrics/metrics.json      Exact Serialized Metrics
Same-Test-Set Benchmark Verification  experiments/03_vision_swin/same_test_comparison/ verification_report.md:1-85
Missing Audio Benchmark Metrics       experiments/06_forensic_analysis/logs/            inference_log.json / missing_audio_tests.json
Dubbed Audio Benchmark Metrics        experiments/06_forensic_analysis/dubbed_audio/    predictions.csv / inference_log.json
Latency Breakdown Audit               experiments/06_forensic_analysis/reports/         latency_audit.md:1-75
External OOD Generalization Audit     experiments/06_forensic_analysis/reports/         external_generalization_diagnosis.md:1-120
Demo Troubleshooting & Summary        experiments/demo_debug/                           demo_diagnosis.md / DEMO_SUMMARY.md
========================================================================================================================
```
