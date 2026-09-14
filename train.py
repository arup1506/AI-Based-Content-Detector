"""
Training and Evaluation Script for AI Text Detection.
Trains the dual-granularity TF-IDF + Calibrated Classifier on data/dataset.csv.
Evaluates accuracy, precision, recall, F1, ROC-AUC, and generates a confusion matrix.
Saves the trained model to models/ai_detector.joblib.
"""

import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score

from src.model import AIDetector, MODEL_PATH


def load_dataset(csv_path: str):
    """Load texts and labels from CSV."""
    texts = []
    labels = []
    metadata = []
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(int(row["label"]))
            metadata.append(row)
            
    return texts, np.array(labels), metadata


def train_and_evaluate():
    csv_file = os.path.join(os.path.dirname(__file__), "data", "dataset.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset not found at: {csv_file}. Run 'python data/generate_dataset.py' first.")

    print(f"Loading dataset from: {csv_file}")
    texts, y, meta = load_dataset(csv_file)
    print(f"Total samples: {len(texts)} (Class 0 [Human]: {sum(y == 0)}, Class 1 [AI]: {sum(y == 1)})")

    # 1. Strict ML Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        texts, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTraining set size: {len(X_train)}, Test set size: {len(X_test)}")

    detector = AIDetector(model_path=MODEL_PATH)
    pipeline = detector.build_pipeline()

    # 2. Stratified 5-Fold Cross-Validation on Training Set
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy')
    cv_f1 = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='f1')

    print("\n--- 5-Fold Cross-Validation Results (Training Data) ---")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"Mean CV F1-Score: {cv_f1.mean():.4f} (+/- {cv_f1.std():.4f})")

    # 3. Fit on Full Training Split
    print("\nFitting final model pipeline on training split...")
    pipeline.fit(X_train, y_train)

    # 4. Out-of-Sample Test Set Evaluation
    y_pred = pipeline.predict(X_test)
    y_probs = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    cm = confusion_matrix(y_test, y_pred)

    print("\n========================================================")
    print("           OUT-OF-SAMPLE TEST EVALUATION                ")
    print("========================================================")
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"Test F1-Score : {f1:.4f}")
    print(f"Test ROC-AUC  : {roc_auc:.4f}")
    print("\nConfusion Matrix (Rows: True [0=Human, 1=AI], Cols: Pred [0=Human, 1=AI]):")
    print(f"             Pred Human   Pred AI")
    print(f"True Human       {cm[0, 0]:<12} {cm[0, 1]}")
    print(f"True AI          {cm[1, 0]:<12} {cm[1, 1]}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Human (0)", "AI (1)"]))

    # 5. Fit on all data for production deployment artifact
    print("\nRetraining model on full dataset for maximum generalization...")
    detector.fit(texts, y.tolist())
    detector.save()
    print("Model training & artifact persistence complete!")

    # 6. Test a quick inference
    print("\n--- Testing Sample Inferences ---")
    human_sample = (
        "I was standing on the subway platform at 8th street, holding a wet umbrella "
        "and listening to a busker butcher a Dylan song on an out-of-tune acoustic guitar. "
        "My shoes were soaked through, and the delay announcement crackled over the intercom."
    )
    ai_sample = (
        "Artificial intelligence plays a pivotal role in modern technological innovation. "
        "Furthermore, by leveraging deep learning architectures, organizations can analyze complex "
        "datasets with remarkable efficiency. In conclusion, navigating these challenges requires robust governance."
    )

    res_human = detector.analyze(human_sample)
    print(f"\n[Human Sample Test]: Verdict = {res_human['verdict']} | AI: {res_human['ai_percentage']}% | Human: {res_human['human_percentage']}%")

    res_ai = detector.analyze(ai_sample)
    print(f"[AI Sample Test]   : Verdict = {res_ai['verdict']} | AI: {res_ai['ai_percentage']}% | Human: {res_ai['human_percentage']}%")


if __name__ == "__main__":
    train_and_evaluate()
