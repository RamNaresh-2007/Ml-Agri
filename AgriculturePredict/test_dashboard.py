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

    def test_08_api_plots_list(self):
        """Verify /api/plots-list discovers EDA and model visual plots."""
        res = self.client.get("/api/plots-list")
        self.assertEqual(res.status_code, 200)
        plots = res.get_json()
        self.assertIsInstance(plots, list)
        self.assertGreater(len(plots), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
