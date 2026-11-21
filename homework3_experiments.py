"""Reproduce Homework 3 using scikit-learn 1.8.0.

Usage: python homework3_experiments.py /path/to/the/three/csvs
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import parallel_config
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
STAGES = (1, 10, 100, 300, 1000)


def wine_regression(data_dir, jobs):
    wine = pd.read_csv(data_dir / "winequality-white.csv", sep=";")
    X = wine.drop(columns="quality").to_numpy()
    y = wine["quality"].to_numpy()
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    train, valid = train_test_split(
        np.arange(len(y_dev)), test_size=1 / 8, random_state=SEED
    )
    print(f"Wine rows: train={len(train)}, validation={len(valid)}, test={len(y_test)}")

    forest = RandomForestRegressor(random_state=SEED, n_jobs=1)
    forest_grid = {
        "n_estimators": [8, 16, 24, 32, 48, 64, 96, 128],
        "max_depth": [4, 8, 16, None],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": [2, 4, 7, 11],
    }
    knn = Pipeline([("scale", StandardScaler()), ("knn", KNeighborsRegressor())])
    knn_grid = {
        "knn__n_neighbors": range(1, 129),
        "knn__weights": ["uniform", "distance"],
        "knn__p": [1, 2],
    }
    for name, model, grid in (("Forest", forest, forest_grid), ("KNN", knn, knn_grid)):
        search = GridSearchCV(
            model, grid, cv=[(train, valid)], scoring="neg_mean_squared_error",
            refit=False, n_jobs=jobs, error_score="raise",
        )
        with parallel_config(backend="loky", inner_max_num_threads=1):
            search.fit(X_dev, y_dev)
        # Refit the winner on the original training rows to evaluate that same rule.
        winner = clone(model).set_params(**search.best_params_).fit(X_dev[train], y_dev[train])
        val_mse = mean_squared_error(y_dev[valid], winner.predict(X_dev[valid]))
        np.testing.assert_allclose(val_mse, -search.best_score_)
        test_mse = mean_squared_error(y_test, winner.predict(X_test))
        print(f"{name}: {len(search.cv_results_['params'])} settings, {search.best_params_}")
        print(f"  validation MSE={val_mse:.6f}, test MSE={test_mse:.6f}")


def boosting_classification(path):
    # Both files have no header. The first column contains the binary response.
    data = np.loadtxt(path, delimiter=",")
    y, X = data[:, 0].astype(int), data[:, 1:]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_errors = []
    for train, held_out in cv.split(X, y):
        model = GradientBoostingClassifier(
            n_estimators=max(STAGES), max_depth=3, learning_rate=0.1,
            loss="log_loss", random_state=SEED,
        ).fit(X[train], y[train])
        # Earlier stages of this fit are the corresponding smaller ensembles.
        predictions = zip(model.staged_predict(X[train]), model.staged_predict(X[held_out]))
        errors = []
        for stage, (train_pred, held_out_pred) in enumerate(predictions, start=1):
            if stage in STAGES:
                errors.append((
                    np.mean(train_pred != y[train]), np.mean(held_out_pred != y[held_out])
                ))
        fold_errors.append(errors)

    quartile = pd.qcut(X[:, 0], q=4, labels=False)
    rates = pd.Series(y).groupby(quartile).mean()
    print(f"{path.stem}: class-1 frequency by X1 quartile:",
          ", ".join(f"{rate:.0%}" for rate in rates))
    print("  trees  train error  held-out error")
    for stage, (train_error, cv_error) in zip(STAGES, np.mean(fold_errors, axis=0)):
        print(f"  {stage:5d}  {train_error:11.4f}  {cv_error:14.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, help="Directory containing the three course CSVs")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel grid-search workers")
    args = parser.parse_args()
    wine_regression(args.data_dir, args.jobs)
    for filename in ("dset0.csv", "dset1.csv"):
        boosting_classification(args.data_dir / filename)
