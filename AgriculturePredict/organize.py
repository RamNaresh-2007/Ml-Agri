import os
import shutil

src_dir = os.path.dirname(os.path.abspath(__file__))

folders = {
    "01_Datasets": [
        "crop_yield.csv",
        "crop_yield_sample.csv",
        "processed_crop_yield.csv"
    ],
    "02_Data_Preprocessing": [
        "categorical.py",
        "categorical_encoding.py",
        "numerical_scaling.py",
        "clean_and_preprocess.py"
    ],
    "03_Exploratory_Data_Analysis": [
        "eda.py",
        "plots",
        "EDA_Outputs"
    ],
    "04_Models_and_Pipelines": [
        "linear_regression.py",
        "linear regression.py",
        "decision_trees.py",
        "random_forest.py",
        "clustering_models.py",
        "complete_pipeline.py",
        "dbscan_clustering.py",
        "25_dbscan.py",
        "hierarchical_clustering.py",
        "logistic_regression.py",
        "Models.py",
        "PPP.py",
        "APP.py"
    ],
    "05_Web_Application": [
        "app.py",
        "dashboard.py",
        "main.py",
        "config.py",
        "static",
        "templates"
    ],
    "06_Outputs_and_Utils": [
        "Utils.py",
        "Model_Outputs",
        "processed_data",
        "outputs",
        "splits",
        "processed",
        "Outputs"
    ]
}


print("=" * 65)
print("  ORGANIZING & VERIFYING AGRICULTUREPREDICT DIRECTORY TREE")
print("=" * 65)

for folder, items in folders.items():
    folder_path = os.path.join(src_dir, folder)
    os.makedirs(folder_path, exist_ok=True)
    status = "OK" if os.path.exists(folder_path) else "MISSING"
    print(f"[{status}] Stage: {folder}")
    for item in items:
        item_path = os.path.join(folder_path, item)
        item_status = "FOUND" if os.path.exists(item_path) else "PENDING"
        print(f"    - {item:35s} [{item_status}]")

print("=" * 65)
print("Organization check complete.")
