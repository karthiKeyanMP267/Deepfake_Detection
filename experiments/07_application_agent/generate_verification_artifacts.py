"""
experiments/07_application_agent/generate_verification_artifacts.py

Executes verification checks and saves structured JSON test records for Phase 7 report.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import json
import time
import unittest
from fastapi.testclient import TestClient
from app.main import app
from tests.test_application import TestApplicationSuite

def main():
    out_dir = BASE_DIR / "experiments" / "07_application_agent"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Unit Tests Suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestApplicationSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    start_time = time.time()
    test_result = runner.run(suite)
    duration = time.time() - start_time

    tests_summary = {
        "total_tests": test_result.testsRun,
        "failures": len(test_result.failures),
        "errors": len(test_result.errors),
        "was_successful": test_result.wasSuccessful(),
        "duration_seconds": round(duration, 3),
        "test_cases": [
            "test_api_health",
            "test_model_singleton_loading",
            "test_valid_video_with_audio_inference",
            "test_missing_audio_fallback",
            "test_silent_audio_handling",
            "test_corrupted_audio_handling",
            "test_corrupted_video_upload",
            "test_invalid_file_upload",
            "test_agent_evidence_generation",
            "test_demo_catalog_and_evaluation"
        ]
    }
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(tests_summary, f, indent=4)
    print("Saved test_results.json")

    # 2. Run Sample Ingestion & Save Sample Output
    client = TestClient(app)
    sample_res = client.post("/api/demo/analyze/case_02_deepfake_multimodal")
    if sample_res.status_code == 200:
        sample_json = sample_res.json()
        with open(out_dir / "sample_inference_output.json", "w") as f:
            json.dump(sample_json, f, indent=4)
        print("Saved sample_inference_output.json")

if __name__ == "__main__":
    main()
