"""
tests/test_llm_agent.py

Unit tests for LLM Forensic Explanation Agent and integration with ForensicAgent.
"""

import unittest
from unittest.mock import patch, MagicMock
from agent.llm_forensic_agent import LLMForensicAgent
from agent.forensic_agent import ForensicAgent


class TestLLMForensicAgent(unittest.TestCase):

    def setUp(self):
        self.llm_agent = LLMForensicAgent()
        self.forensic_agent = ForensicAgent()

    # 1. Deterministic Fallback when unconfigured
    def test_deterministic_fallback_when_no_api_key(self):
        agent = LLMForensicAgent(api_key="")
        evidence = {
            "verdict": "FAKE",
            "confidence": 0.984,
            "probability_real": 0.016,
            "probability_fake": 0.984,
            "missing_audio": False,
            "av_consistency": {"score": 0.0412, "is_mismatched": True},
            "is_ood": False,
            "reliability_tier": "HIGH",
            "ood_flags": [],
        }
        res = agent.generate_explanation(evidence)
        self.assertEqual(res["explanation_source"], "deterministic_engine")
        self.assertIn("summary", res)
        self.assertIn("key_evidence", res)
        self.assertIn("cross_modal_assessment", res)
        self.assertIn("reliability", res)
        self.assertIn("final_explanation", res)
        self.assertTrue(any("Visual" in k for k in res["key_evidence"]))
        self.assertTrue(any("Audio" in k for k in res["key_evidence"]))
        self.assertTrue(any("Cross-modal" in k for k in res["key_evidence"]))

    # 2. Authentic Scenario
    def test_authentic_scenario_reasoning(self):
        raw_output = {
            "prediction": "REAL",
            "confidence": 0.978,
            "probability_real": 0.978,
            "probability_fake": 0.022,
            "audio_available": True,
            "fusion_weights": {"vision": 0.52, "audio": 0.48},
            "av_consistency": 0.384,
            "video_metadata": {"width": 224, "height": 224, "face_area_ratio": 0.55},
        }
        result = self.forensic_agent.process_evidence(raw_output, "test_real.mp4")
        self.assertEqual(result["prediction"], "REAL")
        self.assertIn("llm_explanation", result)
        expl = result["llm_explanation"]
        self.assertIn("REAL", expl["summary"].upper())
        self.assertTrue("synchroniz" in expl["cross_modal_assessment"].lower() or "aligned" in expl["cross_modal_assessment"].lower())


    # 3. Deepfake Scenario
    def test_deepfake_scenario_reasoning(self):
        raw_output = {
            "prediction": "FAKE",
            "confidence": 0.994,
            "probability_real": 0.006,
            "probability_fake": 0.994,
            "audio_available": True,
            "fusion_weights": {"vision": 0.61, "audio": 0.39},
            "av_consistency": 0.041,
            "video_metadata": {"width": 224, "height": 224, "face_area_ratio": 0.60},
        }
        result = self.forensic_agent.process_evidence(raw_output, "test_fake.mp4")
        self.assertEqual(result["prediction"], "FAKE")
        expl = result["llm_explanation"]
        self.assertIn("FAKE", expl["summary"].upper())
        self.assertIn("synthetic", expl["summary"].lower())

    # 4. Missing Audio Scenario (Vision-Only Fallback)
    def test_missing_audio_scenario_reasoning(self):
        raw_output = {
            "prediction": "FAKE",
            "confidence": 0.981,
            "probability_real": 0.019,
            "probability_fake": 0.981,
            "audio_available": False,
            "fusion_weights": {"vision": 1.0, "audio": 0.0},
            "av_consistency": None,
            "video_metadata": {"width": 224, "height": 224, "face_area_ratio": 0.58},
        }
        result = self.forensic_agent.process_evidence(raw_output, "test_silent.mp4")
        self.assertFalse(result["audio_available"])
        expl = result["llm_explanation"]
        self.assertTrue("vision" in expl["summary"].lower() or "visual" in expl["summary"].lower())
        self.assertTrue(any("unavailable" in k.lower() or "bypassed" in k.lower() for k in expl["key_evidence"]))

    # 5. Dubbed / Audio-Visual Mismatch Scenario
    def test_dubbed_mismatch_scenario_reasoning(self):
        raw_output = {
            "prediction": "REAL",
            "confidence": 0.912,
            "probability_real": 0.912,
            "probability_fake": 0.088,
            "audio_available": True,
            "fusion_weights": {"vision": 0.74, "audio": 0.26},
            "av_consistency": -0.0385,
            "video_metadata": {"width": 224, "height": 224, "face_area_ratio": 0.52},
        }
        result = self.forensic_agent.process_evidence(raw_output, "test_dubbed.mp4")
        expl = result["llm_explanation"]
        self.assertTrue(
            "inconsistency" in expl["cross_modal_assessment"].lower()
            or "desynchronization" in expl["cross_modal_assessment"].lower()
            or "dubbing" in expl["cross_modal_assessment"].lower()
        )

    # 6. Out-of-Distribution / Low Reliability Scenario
    def test_ood_scenario_reasoning(self):
        raw_output = {
            "prediction": "REAL",
            "confidence": 0.965,
            "probability_real": 0.965,
            "probability_fake": 0.035,
            "audio_available": True,
            "fusion_weights": {"vision": 0.61, "audio": 0.39},
            "av_consistency": 0.187,
            "video_metadata": {
                "width": 1336,
                "height": 688,
                "aspect_ratio": 1.94,
                "face_area_ratio": 0.037,
                "audio_sample_rate": 48000,
                "is_screen_recording": True,
                "codec": "h264",
            },
        }
        result = self.forensic_agent.process_evidence(raw_output, "test_screen_rec.mp4")
        self.assertEqual(result["agent_analysis"]["forensic_reliability"], "LOW")
        expl = result["llm_explanation"]
        self.assertIn("out-of-distribution", expl["reliability"].lower())

    # 7. Mocked Successful LLM Call
    @patch("agent.llm_forensic_agent.requests.post")
    def test_mocked_successful_llm_call(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": """{
                        "summary": "The media was classified as FAKE with 99.4% confidence due to observed facial warping.",
                        "key_evidence": [
                            "Visual evidence: Spatial boundary artifacts identified in facial region.",
                            "Audio evidence: Spectral patterns indicate neural vocoder synthesis.",
                            "Cross-modal consistency: S_AV indicates cross-modal desynchronization.",
                            "Reliability: Verified in-distribution input."
                        ],
                        "cross_modal_assessment": "Audio-visual desynchronization detected.",
                        "reliability": "High reliability on verified benchmark data.",
                        "final_explanation": "Multimodal analysis confirms artificial synthesis across visual and audio streams."
                    }"""
                }
            }]
        }
        mock_post.return_value = mock_response

        agent = LLMForensicAgent(api_key="sk-test-key-mock")
        res = agent.generate_explanation({"verdict": "FAKE", "confidence": 0.994})
        self.assertEqual(res["explanation_source"], "llm")
        self.assertEqual(len(res["key_evidence"]), 4)
        self.assertIn("facial warping", res["summary"])

    # 8. Graceful Fallback on LLM Network Timeout / Exception
    @patch("agent.llm_forensic_agent.requests.post", side_effect=Exception("Connection timed out"))
    def test_graceful_fallback_on_llm_error(self, mock_post):
        agent = LLMForensicAgent(api_key="sk-test-key-mock")
        evidence = {
            "verdict": "REAL",
            "confidence": 0.95,
            "missing_audio": False,
            "av_consistency": {"score": 0.35, "is_mismatched": False},
            "reliability_tier": "HIGH",
        }
        res = agent.generate_explanation(evidence)
        self.assertEqual(res["explanation_source"], "deterministic_engine")
        self.assertIn("REAL", res["summary"])


if __name__ == "__main__":
    unittest.main()
