"""Classical supervised baselines: logistic regression, random forest, LightGBM."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from fraudlab.models.base import FraudModel, ModelData, Preprocessor

N_JOBS = 5  # fixed so results are identical serial or parallel (DECISIONS D028)


class LogReg(FraudModel):
    """L2 logistic regression on quantile-normalised features; C chosen on validation."""

    name, family = "logreg", "classical"

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> LogReg:
        self.prep = Preprocessor(data.categorical_idx, self.seed).fit(data.X[train])
        Xt, Xv = self.prep.transform(data.X[train]), self.prep.transform(data.X[valid])
        best = (-1.0, None)
        for c in (0.01, 0.1, 1.0, 10.0):
            m = LogisticRegression(C=c, max_iter=2000, random_state=self.seed)
            m.fit(Xt, data.y_train[train])
            score = self.valid_auprc(data.y_train[valid], m.predict_proba(Xv)[:, 1])
            if score > best[0]:
                best = (score, m)
        self.model = best[1]
        self.info = {"C": self.model.C, "valid_auprc": best[0]}
        return self

    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(self.prep.transform(data.X[idx]))[:, 1]


class RandomForest(FraudModel):
    """Random forest (300 trees). NaNs are filled with -1."""

    name, family = "random_forest", "classical"

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> RandomForest:
        self.model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=3,
            max_features="sqrt",
            n_jobs=N_JOBS,
            random_state=self.seed,
        ).fit(np.nan_to_num(data.X[train], nan=-1.0), data.y_train[train])
        return self

    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.nan_to_num(data.X[idx], nan=-1.0))[:, 1]


def _lgb_params(seed: int, **overrides) -> dict:
    return {
        "objective": "binary",
        "verbosity": -1,
        "seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": N_JOBS,
        **overrides,
    }


class LightGBMDefault(FraudModel):
    """LightGBM with library defaults (100 trees, 31 leaves, lr 0.1). No tuning."""

    name, family = "lgbm_default", "classical"

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> LightGBMDefault:
        ds = lgb.Dataset(
            data.X[train],
            data.y_train[train],
            categorical_feature=data.categorical_idx,
            free_raw_data=False,
        )
        self.booster = lgb.train(_lgb_params(self.seed), ds, num_boost_round=100)
        return self

    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        return self.booster.predict(data.X[idx])

    def explain(self, data: ModelData, idx: np.ndarray, top: int = 10, **_) -> dict[str, float]:
        gain = self.booster.feature_importance("gain")
        total = gain.sum() or 1.0
        pairs = sorted(zip(data.feature_names, gain / total, strict=True), key=lambda kv: -kv[1])
        return {k: float(v) for k, v in pairs[:top]}


class LightGBMTuned(LightGBMDefault):
    """LightGBM tuned with Optuna TPE (≤30 trials) on validation AUPRC, with early stopping."""

    name = "lgbm_tuned"

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> LightGBMTuned:
        n_trials = int(self.params.get("n_trials", 30))
        cat = data.categorical_idx
        dtrain = lgb.Dataset(
            data.X[train],
            data.y_train[train],
            categorical_feature=cat,
            free_raw_data=False,
            params={"feature_pre_filter": False},  # trials vary min_child_samples
        )
        dvalid = lgb.Dataset(
            data.X[valid],
            data.y_train[valid],
            categorical_feature=cat,
            reference=dtrain,
            free_raw_data=False,
            params={"feature_pre_filter": False},
        )
        yv = data.y_train[valid]

        def objective(trial: optuna.Trial) -> float:
            params = _lgb_params(
                self.seed,
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 8, 256, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 200, log=True),
                feature_fraction=trial.suggest_float("feature_fraction", 0.4, 1.0),
                bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
                bagging_freq=1,
                lambda_l2=trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
                scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, 50.0, log=True),
                metric="average_precision",
            )
            booster = lgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                valid_sets=[dvalid],
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )
            trial.set_user_attr("best_iteration", booster.best_iteration)
            trial.set_user_attr("booster", booster)
            return self.valid_auprc(
                yv, booster.predict(data.X[valid], num_iteration=booster.best_iteration)
            )

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler(seed=self.seed)
        )
        study.optimize(objective, n_trials=n_trials)
        self.booster = study.best_trial.user_attrs["booster"]
        self.best_iteration = study.best_trial.user_attrs["best_iteration"]
        self.info = {
            "best_params": study.best_params,
            "best_iteration": self.best_iteration,
            "valid_auprc": study.best_value,
            "n_trials": n_trials,
        }
        return self

    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        return self.booster.predict(data.X[idx], num_iteration=self.best_iteration)

    def refit(self, data: ModelData, rows: np.ndarray) -> LightGBMTuned:
        """Retrain on ``rows`` with the tuned hyperparameters and round count (no validation).

        Used for periodic retraining on *all* labels available at retrain time (RQ2).
        """
        params = _lgb_params(self.seed, **self.info["best_params"], bagging_freq=1)
        ds = lgb.Dataset(data.X[rows], data.y_train[rows], categorical_feature=data.categorical_idx)
        self.booster = lgb.train(params, ds, num_boost_round=max(1, self.best_iteration))
        self.best_iteration = self.booster.current_iteration()
        return self
