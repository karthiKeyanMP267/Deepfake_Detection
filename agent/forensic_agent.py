"""
agent/forensic_agent.py

Forensic Agent Class for Agentic Multimodal Deepfake Detection.

Coordinates evidence intake, cross-modal reasoning, structured report generation,
and provides a clean interface for UI/API integration.
"""

from typing import Dict, Any, Optional
from agent.reasoning import ForensicReasoningEngine
from agent.report_generator import ForensicReportGenerator
from agent.llm_forensic_agent import LLMForensicAgent


class ForensicAgent:
    """
    Autonomous Forensic Reasoning Agent that operates on deep learning detector
    outputs to generate grounded, explainable forensic intelligence.
    """

    def __init__(self):
        self.reasoning_engine = ForensicReasoningEngine()
        self.report_generator = ForensicReportGenerator()
        self.llm_agent = LLMForensicAgent()

    def process_evidence(
        self,
        raw_model_output: Dict[str, Any],
        video_name: str = "uploaded_video.mp4",
    ) -> Dict[str, Any]:
        """
        Process the model inference outputs and produce a complete, evidence-grounded
        forensic analysis object.

        Parameters
        ----------
        raw_model_output : Dict[str, Any]
            Dictionary returned by the multimodal model / inference service:
            - prediction: "REAL" or "FAKE"
            - confidence: float (0.0 - 1.0)
            - probability_real: float
            - probability_fake: float
            - audio_available: bool
            - fusion_weights: {"vision": float, "audio": float}
            - av_consistency: float or None
            - intermediate_features: optional dict
        video_name : str
            Filename of analyzed asset
        """
        prediction = raw_model_output.get("prediction", "UNKNOWN")
        confidence = float(raw_model_output.get("confidence", 0.5))
        prob_real = float(raw_model_output.get("probability_real", 0.5))
        prob_fake = float(raw_model_output.get("probability_fake", 0.5))
        audio_available = bool(raw_model_output.get("audio_available", False))
        fusion_weights = raw_model_output.get("fusion_weights", {"vision": 1.0, "audio": 0.0})
        av_consistency = raw_model_output.get("av_consistency")

        # 1. Construct internal evidence structure
        internal_evidence = {
            "filename": video_name,
            "final_prediction": prediction,
            "confidence": confidence,
            "probability_real": prob_real,
            "probability_fake": prob_fake,
            "audio_available": audio_available,
            "fusion_weights": fusion_weights,
            "av_consistency": av_consistency,
            "missing_audio": not audio_available,
            "video_metadata": raw_model_output.get("video_metadata", {}),
        }

        # 2. Run agent reasoning engine
        agent_reasoning = self.reasoning_engine.synthesize_reasoning(internal_evidence)

        # 3. Generate structured LLM explanation grounded in detector evidence
        domain_reliability = agent_reasoning.get("domain_reliability", {})
        structured_payload = {
            "verdict": prediction,
            "confidence": confidence,
            "probability_real": prob_real,
            "probability_fake": prob_fake,
            "visual_evidence": agent_reasoning["visual_evidence"],
            "audio_evidence": agent_reasoning["audio_evidence"],
            "av_consistency": agent_reasoning["cross_modal_analysis"],
            "missing_audio": not audio_available,
            "forensic_status": agent_reasoning.get("forensic_status", "HIGH RELIABILITY — IN-DISTRIBUTION VERIFIED"),
            "reliability_tier": agent_reasoning.get("forensic_reliability", "HIGH"),
            "is_ood": domain_reliability.get("is_ood", False),
            "ood_flags": domain_reliability.get("ood_flags", []),
            "modality_contributions": {
                "vision_weight": fusion_weights.get("vision", 1.0),
                "audio_weight": fusion_weights.get("audio", 0.0),
            },
            "video_metadata": raw_model_output.get("video_metadata", {}),
        }

        llm_explanation = self.llm_agent.generate_explanation(
            structured_evidence=structured_payload,
            fallback_reasoning=agent_reasoning,
        )

        narrative_text = llm_explanation.get("final_explanation", agent_reasoning["narrative_reasoning"])

        # 4. Assemble composite analysis object
        composite_result = {
            "prediction": prediction,
            "confidence": confidence,
            "confidence_percentage": agent_reasoning["confidence_percentage"],
            "probability_real": prob_real,
            "probability_fake": prob_fake,
            "audio_available": audio_available,
            "fusion_weights": fusion_weights,
            "av_consistency": av_consistency,
            "vision_evidence": agent_reasoning["visual_evidence"],
            "audio_evidence": agent_reasoning["audio_evidence"],
            "forensic_evidence": [
                *agent_reasoning["visual_evidence"],
                *agent_reasoning["audio_evidence"],
            ],
            "llm_explanation": llm_explanation,
            "agent_explanation": llm_explanation,
            "agent_analysis": {
                "verdict": agent_reasoning["verdict"],
                "confidence_percentage": agent_reasoning["confidence_percentage"],
                "probability_real": prob_real,
                "probability_fake": prob_fake,
                "forensic_status": agent_reasoning.get("forensic_status", "HIGH RELIABILITY — IN-DISTRIBUTION VERIFIED"),
                "forensic_reliability": agent_reasoning.get("forensic_reliability", "HIGH"),
                "domain_reliability": domain_reliability,
                "reasoning": narrative_text,
                "narrative_reasoning": narrative_text,
                "risk_level": agent_reasoning["risk_level"],
                "recommendation": agent_reasoning["recommendation"],
                "visual_evidence": agent_reasoning["visual_evidence"],
                "audio_evidence": agent_reasoning["audio_evidence"],
                "cross_modal_analysis": agent_reasoning["cross_modal_analysis"],
                "fusion_breakdown": agent_reasoning["fusion_breakdown"],
                "llm_explanation": llm_explanation,
            },
        }

        # 5. Generate markdown report
        markdown_report = self.report_generator.generate_markdown_report(
            analysis_result=composite_result,
            video_filename=video_name,
        )
        composite_result["markdown_report"] = markdown_report

        return composite_result

