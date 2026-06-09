import warnings

import joblib
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# SVC(probability=True) deprecation warning suppressed for compatibility.
# TODO: migrate to CalibratedClassifierCV when sklearn >= 1.11 is required.
warnings.filterwarnings(
    "ignore",
    message="The `probability` parameter was deprecated",
    category=FutureWarning,
)


def build_svm_pipeline(config):
    """Build a sklearn Pipeline: StandardScaler -> (PCA) -> SVC."""
    svm_cfg = config["svm"]
    steps = [("scaler", StandardScaler())]

    if svm_cfg["pca"].get("enabled", False):
        steps.append(
            (
                "pca",
                PCA(n_components=svm_cfg["pca"].get("n_components", 0.95)),
            )
        )

    model_cfg = svm_cfg["model"]
    steps.append(
        (
            "svc",
            SVC(
                kernel=model_cfg.get("kernel", "rbf"),
                class_weight=model_cfg.get("class_weight", "balanced"),
                probability=model_cfg.get("probability", True),
                random_state=model_cfg.get("random_state", 42),
            ),
        )
    )
    return Pipeline(steps)


def train_svm(X_train, y_train, groups, config):
    """Train (and optionally grid-search) the SVM pipeline.

    When grid_search.enabled is True, a StratifiedGroupKFold cross-validation
    is used so augmented variants of the same image stay in the same fold.

    Returns the best (or only) fitted Pipeline.
    """
    pipeline = build_svm_pipeline(config)
    grid_cfg = config["svm"]["grid_search"]

    if not grid_cfg.get("enabled", True):
        pipeline.fit(X_train, y_train)
        return pipeline

    param_grid = {
        "svc__C": grid_cfg.get("C", [1]),
        "svc__gamma": grid_cfg.get("gamma", ["scale"]),
    }
    cv = StratifiedGroupKFold(
        n_splits=grid_cfg.get("cv", 5),
        shuffle=True,
        random_state=config["svm"]["split"].get("random_state", 42),
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=grid_cfg.get("scoring", "f1_macro"),
        cv=cv,
        n_jobs=grid_cfg.get("n_jobs", 1),
        refit=True,
    )
    search.fit(X_train, y_train, groups=groups)
    print(f"Best SVM params: {search.best_params_}  "
          f"(cv {grid_cfg.get('scoring','f1_macro')}={search.best_score_:.4f})")
    return search.best_estimator_


def save_model_bundle(model, feature_columns, path):
    """Persist model + feature column list to a joblib file."""
    joblib.dump({"model": model, "feature_columns": feature_columns}, path)


def load_model_bundle(path):
    """Load a model bundle saved by save_model_bundle."""
    return joblib.load(path)
