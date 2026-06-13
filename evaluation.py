"""
Quantum BioNet 2.0
This repository accompanies the published paper and is provided for
academic inspection and educational purposes.

NOTE:
- Google Colab-specific code has been removed.
- Dataset paths and pretrained weights are intentionally omitted.
- Additional resources are required to reproduce the published results.
"""


# MOUNT GOOGLE DRIVE
import zipfile
import os
import shutil

zip_path = "YOUR_DATASET_PATH"
extract_to = "YOUR_DATASET_PATH"

if os.path.exists(extract_to):
    shutil.rmtree(extract_to)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)

print("Top-level folders/files after extraction:")
for item in os.listdir(extract_to):
    print(item)

# ---- Section ----

def summarize_results(stage1, stage2):
    def extract_metric(report_df, metric):
        return report_df.loc["macro avg", metric] * 100

    def get_auc(result):
        auc_val = result.get("roc_auc")
        if isinstance(auc_val, pd.DataFrame):
            return auc_val["AUC"].mean() * 100
        elif isinstance(auc_val, (float, int)):
            return auc_val * 100
        else:
            return None

    auc_stage1 = get_auc(stage1)
    auc_stage2 = get_auc(stage2)

    summary = {
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "Sensitivity", "Specificity", "AUC"],
        "Stage 1": [
            f"{stage1.get('accuracy', 0)*100:.2f}%",
            f"{extract_metric(stage1['classification_report'], 'precision'):.2f}%",
            f"{extract_metric(stage1['classification_report'], 'recall'):.2f}%",
            f"{extract_metric(stage1['classification_report'], 'f1-score'):.2f}%",
            f"{stage1.get('sensitivity', 0)*100:.2f}%" if stage1.get('sensitivity') is not None else "-",
            f"{stage1.get('specificity', 0)*100:.2f}%" if stage1.get('specificity') is not None else "-",
            f"{auc_stage1:.2f}%" if auc_stage1 is not None else "-"
        ],
        "Stage 2": [
            f"{stage2.get('accuracy', 0)*100:.2f}%",
            f"{extract_metric(stage2['classification_report'], 'precision'):.2f}%",
            f"{extract_metric(stage2['classification_report'], 'recall'):.2f}%",
            f"{extract_metric(stage2['classification_report'], 'f1-score'):.2f}%",
            "-", "-", f"{auc_stage2:.2f}%" if auc_stage2 is not None else "-"
        ]
    }

    df_summary = pd.DataFrame(summary)
    print("\n📊 Comparison of Stage 1 and Stage 2 Metrics:\n")
    print(df_summary.to_string(index=False))
    return df_summary

# ---- Section ----

results_stage1 = evaluate_metrics(model_stage1, test_loader1, device, stage=1, class_names=test_loader1.dataset.classes)
results_stage2 = evaluate_metrics(model_stage2, test_loader2, device, stage=2, class_names=test_loader2.dataset.classes)

df_summarry = summarize_results(results_stage1, results_stage2)

# ---- Section ----

class_names1 = ['Inorganic', 'Organic']
evaluate_classwise_confidence(model_stage1, test_loader1, device, class_names1, stage=1)

# ---- Section ----

class_names2 = ['Food', 'Garden', 'Paper', 'Vegetables', 'Wood']
evaluate_classwise_confidence(model_stage2, test_loader2, device, class_names2, stage=2)