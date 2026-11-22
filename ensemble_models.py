from __future__ import annotations

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

THRESHOLD = 0.5


def make_final_model(seed: int = 42) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500, min_samples_leaf=3, max_features="sqrt", random_state=seed, n_jobs=1
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает модель на train и считает метрики на test."""
    X_train = np.asarray(data.train_features)
    X_test = np.asarray(data.test_features)
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()  #для метрик

    model = make_final_model(seed=data.random_seed)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    pred = (proba >= THRESHOLD).astype(int)

    return EnsembleResult(
        model_name="ExtraTreesClassifier",
        probabilities=proba,
        predictions=pred,
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
    )


def main() -> None:
    import json
    import time
    import warnings

    import pandas as pd
    from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
    from sklearn.model_selection import (
        GridSearchCV, RepeatedStratifiedKFold, train_test_split,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")
    SEED = 42
    df = pd.read_csv("german.csv", sep=";")
    X, y = df.drop(columns="Creditability"), df["Creditability"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    search = {
        "ExtraTrees": (
            ExtraTreesClassifier(n_estimators=500, random_state=SEED),
            {"min_samples_leaf": [1, 3, 5, 10], "max_features": ["sqrt", 0.5]}),
        "RandomForest": (
            RandomForestClassifier(n_estimators=500, random_state=SEED),
            {"min_samples_leaf": [3, 5, 10], "max_features": ["sqrt", 0.3]}),
        "GradientBoosting": (
            GradientBoostingClassifier(subsample=0.7, random_state=SEED),
            {"n_estimators": [100, 300], "learning_rate": [0.02, 0.05], "max_depth": [2, 3]}),
        "AdaBoost": (
            AdaBoostClassifier(random_state=SEED),
            {"n_estimators": [100, 300], "learning_rate": [0.05, 0.5]}),
        "MLP": (
            Pipeline([("scale", StandardScaler()), ("mlp", MLPClassifier(max_iter=2000, random_state=SEED))]),
            {"mlp__hidden_layer_sizes": [(16,), (32,)], "mlp__alpha": [0.1, 1.0, 10.0]}),
    }

    results = {}
    for family, (model, grid) in search.items():
        t = time.time()
        gs = GridSearchCV(model, grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
        gs.fit(X_train, y_train)  # подбор только на train
        i = gs.best_index_
        proba = gs.best_estimator_.predict_proba(X_test)[:, 1]
        pred = (proba >= THRESHOLD).astype(int)
        results[family] = dict(
            best_params={k: (list(v) if isinstance(v, tuple) else v) for k, v in gs.best_params_.items()},
            n_configs=len(gs.cv_results_["params"]),
            cv_roc_auc_mean=float(gs.cv_results_["mean_test_score"][i]),
            cv_roc_auc_std=float(gs.cv_results_["std_test_score"][i]),
            roc_auc=float(roc_auc_score(y_test, proba)),
            accuracy=float(accuracy_score(y_test, pred)),
            precision=float(precision_score(y_test, pred, zero_division=0)),
            recall=float(recall_score(y_test, pred, zero_division=0)),
            f1=float(f1_score(y_test, pred, zero_division=0)),
        )
        r = results[family]
        print(f"{family:17s} {r['n_configs']:2d} конфиг. | CV={r['cv_roc_auc_mean']:.4f}±{r['cv_roc_auc_std']:.3f} "
              f"| test AUC={r['roc_auc']:.4f} | {r['best_params']} ({time.time()-t:.0f}s)", flush=True)

    best = max(results, key=lambda k: results[k]["cv_roc_auc_mean"])
    print("Лучшее семейство по CV:", best, results[best]["best_params"])

    final = run_ensemble_experiment(EnsembleInput(X_train, y_train, X_test, y_test, random_seed=SEED))
    final_metrics = dict(roc_auc=final.roc_auc, accuracy=final.accuracy, precision=final.precision,
                         recall=final.recall, f1=final.f1)
    print("FINAL", final.model_name, final_metrics)


    robust_aucs = []
    for k in range(10):
        X_tr_k, X_te_k, y_tr_k, y_te_k = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=1000 + k)
        res = run_ensemble_experiment(
            EnsembleInput(X_tr_k, y_tr_k, X_te_k, y_te_k, random_seed=SEED))
        robust_aucs.append(res.roc_auc)
    print("10 других разбиений: mean", np.mean(robust_aucs), "min", np.min(robust_aucs))

    out = {
        "model_name": final.model_name, "random_seed": SEED,
        "split": "stratified 80/20, random_state=42 (800 train / 200 test)",
        "selection_criterion": "GridSearchCV ROC-AUC, repeated stratified 5-fold x3, train only",
        "best_family_by_cv": best, "final_test_metrics": final_metrics, "experiments": results,
        "robustness_10_other_splits": {"roc_auc_mean": float(np.mean(robust_aucs)),
                                      "roc_auc_std": float(np.std(robust_aucs)),
                                      "roc_auc_min": float(np.min(robust_aucs))},
    }
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
