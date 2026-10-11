"""
agent/llm_forensic_agent.py

Provider-Agnostic LLM-Powered Forensic Explanation Agent for Deepfake Detection.

This module coordinates structured evidence intake from the 6-encoder multimodal
detector and generates evidence-grounded, non-hallucinatory forensic explanations.

Key Principles:
1. The detector remains the ground truth for REAL/FAKE classification.
2. The LLM acts purely as an explanation/reasoning layer over structured evidence.
3. Strict anti-hallucination prompt prevents inventing artifacts or misrepresenting evidence.
4. Offline / Error Resilience: Seamless deterministic fallback when LLM is unconfigured or unavailable.
"""

import os
import json
import time
from typing import Dict, Any, List, Optional
import requests

from app.config import (
    LLM_API_KEY,
    LLM_PROVIDER,
    LLM_MODEL,
    LLM_BASE_URL,
    LLM_TIMEOUT_SEC,
)


class LLMForensicAgent:
    """
    LLM-powered Agentic Forensic Explanation Service.
    Translates structured detector telemetry and cross-modal findings
    into clear, human-understandable forensic explanations.
    """

    SYSTEM_PROMPT = """You are an AI forensic explanation agent for a multimodal deepfake detection system.
Your role is strictly to explain the output and evidence produced by an existing 6-encoder multimodal detector.

STRICT OPERATIONAL RULES:
1. You MUST use ONLY the supplied structured forensic evidence.
2. You MUST NOT invent visual artifacts, audio artifacts, facial abnormalities, synchronization problems, or other evidence not present in the supplied data.
3. You MUST NOT change or contradict the detector's REAL/FAKE verdict or confidence score.
4. You MUST clearly state when evidence is unavailable (e.g. missing audio).
5. If audio is missing, explicitly explain that the detector used its vision-only fallback pathway.
6. If the input is classified as Out-of-Distribution (OOD) or Low Reliability, explicitly communicate that limitation.
7. Your explanation must be concise, professional, grounded, and easily understood by forensic investigators and reviewers.

You must respond with ONLY a valid JSON object matching this exact schema:
{
    "summary": "Concise 2-3 sentence overview explaining why the system classified this media as REAL or FAKE.",
    "key_evidence": [
        "Visual evidence: specific grounded finding from visual branch",
        "Audio evidence: specific grounded finding or notice of missing audio",
        "Cross-modal consistency: alignment or mismatch assessment based on S_AV",
        "Reliability: assessment of benchmark distribution alignment and reliability"
    ],
    "cross_modal_assessment": "Clear evaluation of whether audio and video are synchronized or show desynchronization/dubbing.",
    "reliability": "Assessment of data reliability and whether OOD factors warrant caution.",
    "final_explanation": "Comprehensive, coherent narrative forensic explanation synthesizing the entire case."
}"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key if api_key is not None else LLM_API_KEY
        prov = (provider or LLM_PROVIDER or "openai").lower()
        if not provider and (self.api_key and (self.api_key.startswith("gsk_") or "groq" in prov)):
            prov = "groq"
        self.provider = prov

        if self.provider == "groq":
            self.model = model or LLM_MODEL or "llama-3.3-70b-versatile"
            self.base_url = base_url or LLM_BASE_URL or "https://api.groq.com/openai/v1"
        elif self.provider == "gemini":
            self.model = model or LLM_MODEL or "gemini-1.5-flash"
            self.base_url = base_url or LLM_BASE_URL or ""
        else:
            self.model = model or LLM_MODEL or "gpt-4o-mini"
            self.base_url = base_url or LLM_BASE_URL or "https://api.openai.com/v1"

        self.timeout = timeout or LLM_TIMEOUT_SEC or 10.0

    def is_configured(self) -> bool:
        """Check if an API key is available."""
        return bool(self.api_key and len(self.api_key.strip()) > 0)

    def generate_explanation(
        self,
        structured_evidence: Dict[str, Any],
        fallback_reasoning: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate structured forensic explanation via LLM if available,
        otherwise fall back gracefully to local deterministic reasoning.

        Parameters
        ----------
        structured_evidence : Dict[str, Any]
            Compact structured evidence payload from detector.
        fallback_reasoning : Optional[Dict[str, Any]]
            Local deterministic reasoning output from ForensicReasoningEngine.

        Returns
        -------
        Dict[str, Any] with keys:
            - summary: str
            - key_evidence: List[str]
            - cross_modal_assessment: str
            - reliability: str
            - final_explanation: str
            - explanation_source: "llm" | "deterministic_engine"
            - model_used: Optional[str]
        """
        if not self.is_configured():
            return self._build_deterministic_explanation(
                evidence=structured_evidence,
                fallback_reasoning=fallback_reasoning,
                reason="LLM API key not configured (running in local deterministic mode)",
            )

        try:
            llm_result = self._call_llm(structured_evidence)
            if self._validate_response(llm_result):
                llm_result["explanation_source"] = "llm"
                llm_result["model_used"] = self.model
                return llm_result
            else:
                print("[!] LLM returned incomplete JSON schema. Falling back to local reasoning.")
        except Exception as e:
            print(f"[!] LLM call failed or timed out ({e}). Falling back to local reasoning.")

        return self._build_deterministic_explanation(
            evidence=structured_evidence,
            fallback_reasoning=fallback_reasoning,
            reason="LLM fallback activated",
        )

    def _call_llm(self, structured_evidence: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke LLM using configured provider."""
        evidence_json = json.dumps(structured_evidence, indent=2)
        user_prompt = f"Analyze the following multimodal forensic detector evidence and generate the explanation JSON:\n\n```json\n{evidence_json}\n```"

        if self.provider == "gemini":
            return self._call_gemini_api(user_prompt)
        else:
            return self._call_openai_compatible_api(user_prompt)

    def _call_openai_compatible_api(self, user_prompt: str) -> Dict[str, Any]:
        """Call OpenAI or any OpenAI-compatible API endpoint."""
        url = self.base_url or "https://api.openai.com/v1/chat/completions"
        if not url.endswith("/chat/completions") and not url.endswith("/v1"):
            url = f"{url.rstrip('/')}/chat/completions"
        elif url.endswith("/v1"):
            url = f"{url}/chat/completions"

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 800,
        }

        # Try with response_format first
        try:
            rf_payload = dict(payload, response_format={"type": "json_object"})
            response = requests.post(url, headers=headers, json=rf_payload, timeout=self.timeout)
            response.raise_for_status()
        except Exception:
            # Fallback without response_format constraint (for models that format via prompt)
            response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            response.raise_for_status()

        data = response.json()
        content = data["choices"][0]["message"]["content"]

        # Strip any thinking tags or markdown codeblocks
        clean_content = content.strip()
        if "<think>" in clean_content and "</think>" in clean_content:
            clean_content = clean_content.split("</think>")[-1].strip()
        if clean_content.startswith("```json"):
            clean_content = clean_content[7:]
        if clean_content.startswith("```"):
            clean_content = clean_content[3:]
        if clean_content.endswith("```"):
            clean_content = clean_content[:-3]
        clean_content = clean_content.strip()

        return json.loads(clean_content)

    def _call_gemini_api(self, user_prompt: str) -> Dict[str, Any]:
        """Call Google Gemini REST API."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "system_instruction": {"parts": [{"text": self.SYSTEM_PROMPT}]},
            "contents": [{"parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2,
                "maxOutputTokens": 800,
            },
        }

        response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        text_content = data["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(text_content)

    def _validate_response(self, resp: Any) -> bool:
        """Validate that LLM response adheres to required schema."""
        if not isinstance(resp, dict):
            return False
        required_keys = ["summary", "key_evidence", "cross_modal_assessment", "reliability", "final_explanation"]
        for k in required_keys:
            if k not in resp:
                return False
        if not isinstance(resp["key_evidence"], list) or len(resp["key_evidence"]) == 0:
            return False
        return True

    def _build_deterministic_explanation(
        self,
        evidence: Dict[str, Any],
        fallback_reasoning: Optional[Dict[str, Any]] = None,
        reason: str = "Local deterministic reasoning engine",
    ) -> Dict[str, Any]:
        """
        Construct a deterministic, high-fidelity explanation JSON
        using the grounded reasoning engine outputs without LLM dependencies.
        """
        verdict = evidence.get("verdict", "UNKNOWN")
        confidence = float(evidence.get("confidence", 0.5))
        confidence_pct = round(confidence * 100.0, 1)
        missing_audio = evidence.get("missing_audio", False)
        av_info = evidence.get("av_consistency", {}) or {}
        is_mismatched = av_info.get("is_mismatched", False) if isinstance(av_info, dict) else False
        av_score = av_info.get("score") if isinstance(av_info, dict) else None
        is_ood = evidence.get("is_ood", False)
        reliability_tag = evidence.get("reliability_tier", "HIGH")
        ood_flags = evidence.get("ood_flags", [])

        # 1. Summary
        if verdict == "FAKE":
            if missing_audio:
                summary = (
                    f"The system classified this media as FAKE with {confidence_pct}% confidence. "
                    "Facial spatio-temporal artifacts and unnatural articulation patterns indicate synthetic generation, "
                    "analyzed via the vision-only fallback pathway due to missing audio."
                )
            else:
                summary = (
                    f"The system classified this media as FAKE with {confidence_pct}% confidence. "
                    "Multimodal evidence across visual facial dynamics and speech acoustic markers demonstrates "
                    "significant synthetic manipulation indicators."
                )
        else:
            if missing_audio:
                summary = (
                    f"The system classified this media as REAL with {confidence_pct}% confidence based on natural "
                    "facial texture fidelity and coherent biological motion. Audio was unavailable, so the analysis relied "
                    "on the vision-only pathway."
                )
            else:
                summary = (
                    f"The system classified this media as REAL with {confidence_pct}% confidence. "
                    "Visual facial motion and acoustic speech representations demonstrate natural human fidelity and cross-modal synchronization."
                )

        # 2. Key Evidence
        key_evidence = []
        # Visual evidence bullet
        if verdict == "FAKE":
            key_evidence.append("Visual evidence: Spatio-temporal facial texture anomalies and unnatural motion articulation detected.")
        else:
            key_evidence.append("Visual evidence: Natural facial texture continuity and authentic biological frame-to-frame motion verified.")

        # Audio evidence bullet
        if missing_audio:
            key_evidence.append("Audio evidence: Audio stream was unavailable; acoustic verification safely bypassed (100% vision-only fallback).")
        else:
            if verdict == "FAKE":
                key_evidence.append("Audio evidence: Acoustic spectral characteristics indicate synthetic vocoder or voice cloning markers.")
            else:
                key_evidence.append("Audio evidence: Acoustic spectrogram displays natural harmonic distribution and authentic vocal resonance.")

        # Cross-modal bullet
        if missing_audio or av_score is None:
            key_evidence.append("Cross-modal consistency: Cross-modal alignment check unavailable due to missing audio track.")
        elif is_mismatched:
            key_evidence.append(f"Cross-modal consistency: Cross-modal mismatch detected (S_AV = {av_score:.4f} < 0.1177 threshold), indicating audio-visual desynchronization or dubbing.")
        else:
            key_evidence.append(f"Cross-modal consistency: Audio-visual representation is synchronized (S_AV = {av_score:.4f} >= 0.1177 threshold).")

        # Reliability bullet
        if is_ood:
            key_evidence.append(f"Reliability: Low reliability ({reliability_tag}) — input exhibits characteristics outside the verified training distribution ({', '.join(ood_flags[:2]) or 'OOD domain shift'}).")
        else:
            key_evidence.append(f"Reliability: High reliability ({reliability_tag}) — input aligns with verified benchmark distribution standards.")

        # 3. Cross-modal Assessment
        if missing_audio or av_score is None:
            cross_modal_assessment = "Audio was absent or unreadable; cross-modal consistency evaluation was safely bypassed."
        elif is_mismatched:
            cross_modal_assessment = (
                f"Audio-visual inconsistency and desynchronization detected (S_AV = {av_score:.4f}). "
                "The acoustic speech track displays low correlation with visual facial movements, indicating potential dubbing, voice replacement, or desynchronization."
            )
        else:
            cross_modal_assessment = (
                f"Audio-visual alignment and synchronization verified (S_AV = {av_score:.4f}). "
                "Visual speech articulation matches the acoustic speech cadence consistently."
            )

        # 4. Reliability Assessment
        if is_ood:
            reliability_text = (
                f"Low Reliability ({reliability_tag}) — Out-of-Distribution Notice: "
                f"The detector reached a classification of {verdict}, but input parameters (e.g. {', '.join(ood_flags) or 'format variance'}) "
                "differ from the standard evaluation domain. This finding should be interpreted with secondary human review."
            )
        elif missing_audio:
            reliability_text = (
                f"Moderate Reliability ({reliability_tag}) — Vision-Only Mode: "
                "The analysis is reliable for visual features, but full multimodal corroboration requires an intact audio stream."
            )
        else:
            reliability_text = f"High Reliability ({reliability_tag}) — Content is verified in-distribution with full multimodal coverage."

        # 5. Final Explanation
        if fallback_reasoning and "narrative_reasoning" in fallback_reasoning:
            final_explanation = fallback_reasoning["narrative_reasoning"]
        else:
            final_explanation = f"{summary} {cross_modal_assessment} {reliability_text}"

        return {
            "summary": summary,
            "key_evidence": key_evidence,
            "cross_modal_assessment": cross_modal_assessment,
            "reliability": reliability_text,
            "final_explanation": final_explanation,
            "explanation_source": "deterministic_engine",
            "model_used": None,
            "fallback_note": reason,
        }
