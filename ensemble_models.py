from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from grader_contracts.ensembles import EnsembleInput, EnsembleResult

THRESHOLD = 0.5


GERMAN_FEATURES = [
    "Account_Balance", "Duration_of_Credit_monthly", "Payment_Status_of_Previous_Credit",
    "Purpose", "Credit_Amount", "Value_Savings_Stocks", "Length_of_current_employment",
    "Instalment_per_cent", "Sex_Marital_Status", "Guarantors", "Duration_in_Current_address",
    "Most_valuable_available_asset", "Age_years", "Concurrent_Credits", "Type_of_apartment",
    "No_of_Credits_at_this_Bank", "Occupation", "No_of_dependents", "Telephone", "Foreign_Worker",
]

CATEGORICAL = [
    "Account_Balance", "Payment_Status_of_Previous_Credit", "Purpose", "Value_Savings_Stocks",
    "Length_of_current_employment", "Sex_Marital_Status", "Guarantors",
    "Most_valuable_available_asset", "Concurrent_Credits", "Type_of_apartment", "Occupation",
]


def _to_frame(features) -> pd.DataFrame:
    """Массив numpy получает имена столбцов german.csv"""
    if isinstance(features, pd.DataFrame):
        return features.reset_index(drop=True)
    arr = np.asarray(features)
    if arr.ndim == 2 and arr.shape[1] == len(GERMAN_FEATURES):
        return pd.DataFrame(arr, columns=GERMAN_FEATURES)
    return pd.DataFrame(arr)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"Credit_Amount", "Duration_of_Credit_monthly", "Instalment_per_cent", "Age_years"} <= set(df.columns):
        df["log_amount"] = np.log1p(df["Credit_Amount"])
        df["amount_per_month"] = df["Credit_Amount"] / df["Duration_of_Credit_monthly"].clip(lower=1)
        df["amount_x_instalment"] = df["Credit_Amount"] * df["Instalment_per_cent"] / 100.0
        df["log_age"] = np.log(df["Age_years"].clip(lower=1))
    return df


def make_logreg_pipeline(columns, C: float = 0.1, seed: int = 42):
    cols = list(add_features(pd.DataFrame(columns=list(columns), dtype=float)).columns)
    cat = [c for c in CATEGORICAL if c in cols]
    num = [c for c in cols if c not in cat]
    transformer = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), cat), ("num", StandardScaler(), num)]
    )
    return make_pipeline(
        FunctionTransformer(add_features), transformer,
        LogisticRegression(C=C, max_iter=3000, random_state=seed),
    )


def make_random_forest(seed: int = 42) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, max_features=0.3, random_state=seed, n_jobs=1
    )


def make_gradient_boosting(seed: int = 42) -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.02, subsample=0.7, random_state=seed
    )


def make_final_model(columns, seed: int = 42) -> VotingClassifier:
    return VotingClassifier(
        [
            ("rf", make_random_forest(seed)),
            ("gb", make_gradient_boosting(seed)),
            ("lr", make_logreg_pipeline(columns, C=0.1, seed=seed)),
        ],
        voting="soft",
    )


def run_ensemble_experiment(data: EnsembleInput) -> EnsembleResult:
    """Обучает soft-voting ансамбль на train и считает метрики на test.
    test_target используется только для расчёта итоговых метрик.
    """
    X_train = _to_frame(data.train_features)
    X_test = _to_frame(data.test_features)
    X_test = X_test[X_train.columns] if set(X_train.columns) == set(X_test.columns) else X_test
    y_train = np.asarray(data.train_target).ravel()
    y_test = np.asarray(data.test_target).ravel()

    model = make_final_model(X_train.columns, seed=data.random_seed)
    model.fit(X_train, y_train)

    positive_idx = list(model.classes_).index(1)
    proba = model.predict_proba(X_test)[:, positive_idx]
    pred = (proba >= THRESHOLD).astype(int)

    return EnsembleResult(
        model_name="Soft Voting: RandomForest + GradientBoosting + LogisticRegression(one-hot)",
        probabilities=proba,
        predictions=pred,
        roc_auc=float(roc_auc_score(y_test, proba)),
        accuracy=float(accuracy_score(y_test, pred)),
        precision=float(precision_score(y_test, pred, zero_division=0)),
        recall=float(recall_score(y_test, pred, zero_division=0)),
        f1=float(f1_score(y_test, pred, zero_division=0)),
    )


def main() -> None:
    """Воспроизводит эксперименты лабораторной 4 и пишет experiment_results. Метрики на test приводятся для отчёта.
    Импорты, нужные только для экспериментов, лежат внутри функции
    """
    import json
    import time
    import warnings

    from sklearn.ensemble import (
        AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier,
        StackingClassifier,
    )
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, train_test_split
    from sklearn.neural_network import MLPClassifier

    warnings.filterwarnings("ignore")
    SEED = 42
    df = pd.read_csv("german.csv", sep=";")
    X, y = df.drop(columns="Creditability"), df["Creditability"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    cols = X.columns

    def test_metrics(model):
        model.fit(X_tr, y_tr)
        p = model.predict_proba(X_te)[:, 1]
        pred = (p >= THRESHOLD).astype(int)
        return dict(roc_auc=roc_auc_score(y_te, p), accuracy=accuracy_score(y_te, pred),
                    precision=precision_score(y_te, pred), recall=recall_score(y_te, pred),
                    f1=f1_score(y_te, pred))

    def extra_trees():
        return ExtraTreesClassifier(500, min_samples_leaf=3, max_features="sqrt", random_state=SEED)

    candidates = {
        "LogReg (z-score, как в лаб. 3, C=0.1)": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)),
        "LogReg (one-hot + признаки, C=0.1)": make_logreg_pipeline(cols, 0.1, SEED),
        "RandomForest (500, leaf=5, mf=0.3)": make_random_forest(SEED),
        "ExtraTrees (500, leaf=3, sqrt)": extra_trees(),
        "AdaBoost (200, lr=0.05)": AdaBoostClassifier(n_estimators=200, learning_rate=0.05, random_state=SEED),
        "GradientBoosting (300, d=3, lr=0.02, ss=0.7)": make_gradient_boosting(SEED),
        "HistGradientBoosting (d=3, lr=0.03, 200)": HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.03, max_iter=200, l2_regularization=1.0, random_state=SEED),
        "MLP (16, alpha=1)": make_pipeline(StandardScaler(), MLPClassifier((16,), alpha=1.0, max_iter=2000, random_state=SEED)),
        "MLP (32, alpha=0.1)": make_pipeline(StandardScaler(), MLPClassifier((32,), alpha=0.1, max_iter=2000, random_state=SEED)),
        "Voting RF+GB": VotingClassifier([("rf", make_random_forest(SEED)), ("gb", make_gradient_boosting(SEED))], voting="soft"),
        "Voting RF+GB+LR (ВЫБРАНА)": make_final_model(cols, SEED),
        "Voting RF+GB+ET+LR": VotingClassifier(
            [("rf", make_random_forest(SEED)), ("gb", make_gradient_boosting(SEED)),
             ("et", extra_trees()), ("lr", make_logreg_pipeline(cols, 0.1, SEED))], voting="soft"),
        "Stacking RF+GB+LR -> LogReg": StackingClassifier(
            [("rf", make_random_forest(SEED)), ("gb", make_gradient_boosting(SEED)), ("lr", make_logreg_pipeline(cols, 0.1, SEED))],
            final_estimator=LogisticRegression(C=1.0), cv=5, stack_method="predict_proba"),
    }

    results = {}
    for name, model in candidates.items():
        t = time.time()
        s = cross_val_score(model, X_tr, y_tr, cv=cv, scoring="roc_auc")
        m = test_metrics(model)
        results[name] = dict(cv_roc_auc_mean=float(s.mean()), cv_roc_auc_std=float(s.std()),
                             **{k: float(v) for k, v in m.items()})
        print(f"{name:46s} CV={s.mean():.4f}±{s.std():.3f} test AUC={m['roc_auc']:.4f} ({time.time()-t:.0f}s)", flush=True)

    # вызов контрактной функции
    final = run_ensemble_experiment(EnsembleInput(X_tr, y_tr, X_te, y_te, random_seed=SEED))
    final_metrics = dict(roc_auc=final.roc_auc, accuracy=final.accuracy, precision=final.precision,
                         recall=final.recall, f1=final.f1)
    print("FINAL", final.model_name, final_metrics)

    robust = []
    for s in range(10):
        a, b, c, d = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1000 + s)
        r = run_ensemble_experiment(EnsembleInput(a, c, b, d, random_seed=SEED))
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)).fit(a, c)
        robust.append((r.roc_auc, roc_auc_score(d, lr.predict_proba(b)[:, 1])))
    robust = np.array(robust)
    print("robust ensemble mean/std/min", robust[:, 0].mean(), robust[:, 0].std(), robust[:, 0].min())
    print("robust logreg   mean/std/min", robust[:, 1].mean(), robust[:, 1].std(), robust[:, 1].min())

    out = {
        "model_name": final.model_name,
        "random_seed": SEED,
        "split": "stratified 80/20, random_state=42 (800 train / 200 test)",
        "final_test_metrics": final_metrics,
        "selection_criterion": "repeated stratified 5-fold x3 CV ROC-AUC on train only",
        "selected_cv_roc_auc_mean": results["Voting RF+GB+LR (ВЫБРАНА)"]["cv_roc_auc_mean"],
        "experiments": results,
        "robustness_10_other_splits": {
            "ensemble_roc_auc_mean": float(robust[:, 0].mean()), "ensemble_roc_auc_std": float(robust[:, 0].std()),
            "ensemble_roc_auc_min": float(robust[:, 0].min()),
            "logreg_roc_auc_mean": float(robust[:, 1].mean()), "logreg_roc_auc_std": float(robust[:, 1].std()),
            "ensemble_beats_logreg_in": int((robust[:, 0] > robust[:, 1]).sum()),
        },
    }
    with open("experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
