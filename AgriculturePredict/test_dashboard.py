#!/usr/bin/env python
"""
AgriYield AI - Automated Verification & Smoke Test Suite
Validates the canonical 6-stage architecture, datasets, models, plots,
and Flask REST API endpoints without requiring a live browser.
"""

import os
import sys
import unittest
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "05_Web_Application"))

import importlib.util
app_file = BASE_DIR / "05_Web_Application" / "app.py"
spec = importlib.util.spec_from_file_location("agri_test_app", str(app_file))
web_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(web_app)
flask_app = web_app.app


class TestAgriYieldApp(unittest.TestCase):
    def setUp(self):
        flask_app.config['TESTING'] = True
        self.client = flask_app.test_client()

    def test_01_stages_exist(self):
        """Verify all 6 canonical stages exist."""
        required_stages = [
            "01_Datasets",
            "02_Data_Preprocessing",
            "03_Exploratory_Data_Analysis",
            "04_Models_and_Pipelines",
            "05_Web_Application",
            "06_Outputs_and_Utils"
        ]
        for stage in required_stages:
            stage_dir = BASE_DIR / stage
            self.assertTrue(stage_dir.is_dir(), f"Stage directory missing: {stage}")

    def test_02_datasets_exist(self):
        """Verify primary datasets are present in 01_Datasets."""
        raw_data = BASE_DIR / "01_Datasets" / "crop_yield.csv"
        processed_data = BASE_DIR / "01_Datasets" / "processed_crop_yield.csv"
        self.assertTrue(raw_data.is_file(), "Raw dataset missing from 01_Datasets")
        self.assertTrue(processed_data.is_file(), "Processed dataset missing from 01_Datasets")

    def test_03_model_artifacts_exist(self):
        """Verify serialized models exist in 06_Outputs_and_Utils."""
        model_file = BASE_DIR / "06_Outputs_and_Utils" / "Model_Outputs" / "Regression" / "random_forest.joblib"
        prep_file = BASE_DIR / "06_Outputs_and_Utils" / "processed_data" / "preprocessor.joblib"
        self.assertTrue(model_file.is_file(), "Regression model missing from 06_Outputs_and_Utils")
        self.assertTrue(prep_file.is_file(), "Preprocessor missing from 06_Outputs_and_Utils")

    def test_04_home_route(self):
        """Verify dashboard index page serves HTTP 200."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"AgriYield AI", res.data)

    def test_05_api_data(self):
        """Verify /api/data returns valid JSON stats."""
        res = self.client.get("/api/data")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("kpis", data)
        self.assertIn("total_records", data["kpis"])
        self.assertGreater(data["kpis"]["total_records"], 0)

    def test_06_api_metadata(self):
        """Verify /api/metadata returns unique crops and states."""
        res = self.client.get("/api/metadata")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("crops", data)
        self.assertIn("states", data)

    def test_07_api_predict(self):
        """Verify /api/predict computes yield from machine learning model."""
        payload = {
            "crop": "Wheat",
            "state": "Punjab",
            "season": "Rabi",
            "crop_year": 2024,
            "area": 1200.0,
            "annual_rainfall": 1100.0,
            "fertilizer": 90000.0,
            "pesticide": 700.0
        }
        res = self.client.post("/api/predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("predicted_yield_t_ha", data)
        self.assertGreater(data["predicted_yield_t_ha"], 0)

        # Warm prediction should be virtually instantaneous thanks to in-memory model cache
        import time
        t0 = time.time()
        res2 = self.client.post("/api/predict", json=payload)
        latency = time.time() - t0
        self.assertEqual(res2.status_code, 200)
        self.assertLess(latency, 1.0, f"Cached prediction latency ({latency:.3f}s) exceeded 1.0s threshold")


    def test_08_api_plots_list(self):
        """Verify /api/plots-list discovers EDA and model visual plots."""
        res = self.client.get("/api/plots-list")
        self.assertEqual(res.status_code, 200)
        plots = res.get_json()
        self.assertIsInstance(plots, list)
        self.assertGreater(len(plots), 0)

    def test_09_api_models_comparison(self):
        """Verify /api/models/comparison returns benchmark metrics."""
        res = self.client.get("/api/models/comparison")
        self.assertEqual(res.status_code, 200)
        models = res.get_json()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        self.assertIn("Model", models[0])
        self.assertIn("R2_Score", models[0])

    def test_10_api_feature_importances(self):
        """Verify /api/models/feature-importances returns ranking features."""
        res = self.client.get("/api/models/feature-importances")
        self.assertEqual(res.status_code, 200)
        features = res.get_json()
        self.assertIsInstance(features, list)
        self.assertGreater(len(features), 0)
        self.assertIn("Feature", features[0])

    def test_11_api_clustering(self):
        """Verify /api/clustering returns persona archetypes."""
        res = self.client.get("/api/clustering")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("personas", data)

    def test_12_api_recommend_crops(self):
        """Verify /api/recommend-crops returns ranked suitable crops."""
        payload = {
            "state": "Punjab",
            "season": "Kharif",
            "area": 1000.0,
            "annual_rainfall": 1200.0,
            "fertilizer": 80000.0,
            "pesticide": 600.0
        }
        res = self.client.post("/api/recommend-crops", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("top_recommendations", data)
        self.assertGreater(len(data["top_recommendations"]), 0)
        self.assertIn("badge", data["top_recommendations"][0])

    def test_13_api_optimize_inputs(self):
        """Verify /api/optimize-inputs computes fertilizer sweet spot and curve."""
        payload = {
            "crop": "Wheat",
            "state": "Punjab",
            "season": "Rabi",
            "area": 1000.0,
            "annual_rainfall": 1100.0,
            "fertilizer": 90000.0,
            "pesticide": 700.0
        }
        res = self.client.post("/api/optimize-inputs", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("recommended_fertilizer_kg", data)
        self.assertIn("response_curve", data)
        self.assertEqual(len(data["response_curve"]), 7)

    def test_14_api_batch_predict(self):
        """Verify /api/batch-predict handles multi-farm simulation."""
        payload = {
            "records": [
                {"crop": "Rice", "state": "Punjab", "season": "Kharif", "area": 500, "annual_rainfall": 1200, "fertilizer": 40000, "pesticide": 300},
                {"crop": "Wheat", "state": "Haryana", "season": "Rabi", "area": 800, "annual_rainfall": 900, "fertilizer": 65000, "pesticide": 450}
            ]
        }
        res = self.client.post("/api/batch-predict", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["count"], 2)
        self.assertIn("kpis", data)
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)

    def test_15_api_advisory_report(self):
        """Verify /api/advisory-report generates structured farm report data."""
        payload = {
            "crop": "Rice",
            "state": "Punjab",
            "season": "Kharif",
            "area": 1000.0,
            "annual_rainfall": 1200.0,
            "fertilizer": 80000.0,
            "pesticide": 600.0
        }
        res = self.client.post("/api/advisory-report", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("report_id", data)
        self.assertIn("agronomic_advisory", data)

    def test_16_api_pca(self):
        """Verify /api/pca returns PCA variance and decomposition data."""
        res = self.client.get("/api/pca")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("pca", data)

    def test_17_api_anomaly(self):
        """Verify /api/anomaly returns outlier statistics and diagnostics."""
        res = self.client.get("/api/anomaly")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("anomaly", data)

    def test_18_pca_anomaly_scripts_exist(self):
        """Verify PCA and Anomaly Detection script files exist in 04_Models_and_Pipelines."""
        models_dir = BASE_DIR / "04_Models_and_Pipelines"
        self.assertTrue((models_dir / "pca_analysis.py").is_file(), "pca_analysis.py missing")
        self.assertTrue((models_dir / "26_pca.py").is_file(), "26_pca.py missing")
        self.assertTrue((models_dir / "anomaly_detection.py").is_file(), "anomaly_detection.py missing")
        self.assertTrue((models_dir / "27_anomaly_detection.py").is_file(), "27_anomaly_detection.py missing")


if __name__ == "__main__":
    unittest.main(verbosity=2)


