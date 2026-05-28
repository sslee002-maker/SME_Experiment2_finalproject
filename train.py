import gzip
import pickle
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


MODEL_PATH = "model.pkl"


def get_anchor_positions(data):
    BS_positions = np.asarray(data["BS_positions"], dtype=float)

    if BS_positions.shape == (18, 2):
        BS_positions = BS_positions.T

    return BS_positions


def asd_loc_base_algorithm(d_u, BS_positions):
    d_u = np.asarray(d_u, dtype=float).reshape(-1)
    BS_positions = np.asarray(BS_positions, dtype=float)

    if BS_positions.shape == (18, 2):
        BS_positions = BS_positions.T

    valid = np.isfinite(d_u) & (d_u > 0)

    anchors = BS_positions[:, valid]
    distances = d_u[valid]

    if anchors.shape[1] < 3:
        return np.mean(BS_positions, axis=1)

    eps = 1e-6

    distance_weight = 1.0 / ((distances + eps) ** 2)
    distance_weight = distance_weight / (np.max(distance_weight) + eps)

    k = min(8, anchors.shape[1])
    nearest_idx = np.argsort(distances)[:k]

    init_weight = distance_weight[nearest_idx]
    init_weight = init_weight / (np.sum(init_weight) + eps)

    x0 = np.sum(anchors[:, nearest_idx] * init_weight.reshape(1, -1), axis=1)

    margin = 20.0
    x_min = np.min(BS_positions[0, :]) - margin
    x_max = np.max(BS_positions[0, :]) + margin
    y_min = np.min(BS_positions[1, :]) - margin
    y_max = np.max(BS_positions[1, :]) + margin

    bounds = ([x_min, y_min], [x_max, y_max])

    def residual_stage1(x):
        predicted = np.sqrt(
            (x[0] - anchors[0, :]) ** 2 +
            (x[1] - anchors[1, :]) ** 2
        )
        residual = predicted - distances
        return np.sqrt(distance_weight) * residual

    result1 = least_squares(
        residual_stage1,
        x0,
        bounds=bounds,
        loss="soft_l1",
        f_scale=5.0,
        max_nfev=150
    )

    x_stage1 = result1.x

    predicted_stage1 = np.sqrt(
        (x_stage1[0] - anchors[0, :]) ** 2 +
        (x_stage1[1] - anchors[1, :]) ** 2
    )

    residual_abs = np.abs(predicted_stage1 - distances)

    median_res = np.median(residual_abs)
    mad_res = np.median(np.abs(residual_abs - median_res)) + eps
    robust_scale = 1.4826 * mad_res + eps

    residual_weight = 1.0 / (1.0 + (residual_abs / robust_scale) ** 2)

    final_weight = distance_weight * residual_weight
    final_weight = np.clip(final_weight, 0.05, 1.0)

    def residual_stage2(x):
        predicted = np.sqrt(
            (x[0] - anchors[0, :]) ** 2 +
            (x[1] - anchors[1, :]) ** 2
        )
        residual = predicted - distances
        return np.sqrt(final_weight) * residual

    result2 = least_squares(
        residual_stage2,
        x_stage1,
        bounds=bounds,
        loss="soft_l1",
        f_scale=3.0,
        max_nfev=150
    )

    return result2.x


def make_feature(d_u, BS_positions):
    d_u = np.asarray(d_u, dtype=float).reshape(-1)
    BS_positions = np.asarray(BS_positions, dtype=float)

    if BS_positions.shape == (18, 2):
        BS_positions = BS_positions.T

    d_clean = d_u.copy()

    valid = np.isfinite(d_clean) & (d_clean > 0)

    if np.sum(valid) == 0:
        d_clean[:] = 0.0
    else:
        fill_value = np.median(d_clean[valid])
        d_clean[~valid] = fill_value

    p_base = asd_loc_base_algorithm(d_clean, BS_positions)

    predicted_dist = np.sqrt(
        (p_base[0] - BS_positions[0, :]) ** 2 +
        (p_base[1] - BS_positions[1, :]) ** 2
    )

    residual = predicted_dist - d_clean

    sorted_d = np.sort(d_clean)
    sorted_res = np.sort(np.abs(residual))

    eps = 1e-6

    nearest_idx = np.argsort(d_clean)
    nearest3 = nearest_idx[:3]
    nearest5 = nearest_idx[:5]

    nearest3_center = np.mean(BS_positions[:, nearest3], axis=1)
    nearest5_center = np.mean(BS_positions[:, nearest5], axis=1)

    w = 1.0 / (d_clean + eps) ** 2
    w = w / (np.sum(w) + eps)
    weighted_anchor_center = np.sum(BS_positions * w.reshape(1, -1), axis=1)

    nearest_indices_feature = nearest_idx[:5].astype(float)

    diff_base_nearest3 = p_base - nearest3_center
    diff_base_nearest5 = p_base - nearest5_center
    diff_base_weighted = p_base - weighted_anchor_center

    feature = np.concatenate([
        d_clean,
        sorted_d[:8],

        np.array([
            np.mean(d_clean),
            np.std(d_clean),
            np.median(d_clean),
            np.min(d_clean),
            np.max(d_clean),
            np.percentile(d_clean, 10),
            np.percentile(d_clean, 25),
            np.percentile(d_clean, 75),
            np.percentile(d_clean, 90),

            p_base[0],
            p_base[1],

            np.mean(np.abs(residual)),
            np.std(residual),
            np.median(np.abs(residual)),
            np.max(np.abs(residual)),
            np.percentile(np.abs(residual), 75),
            np.percentile(np.abs(residual), 90),
        ]),

        sorted_res[:8],

        nearest3_center,
        nearest5_center,
        weighted_anchor_center,

        diff_base_nearest3,
        diff_base_nearest5,
        diff_base_weighted,

        d_clean[nearest_idx[:5]],
        nearest_indices_feature
    ])

    return feature, p_base


def position_error(p_true, p_pred):
    return np.sqrt(
        (p_true[0, :] - p_pred[0, :]) ** 2 +
        (p_true[1, :] - p_pred[1, :]) ** 2
    )


def build_candidate_models():
    candidates = {}

    candidates["RF_depth12_leaf2"] = RandomForestRegressor(
        n_estimators=500,
        max_depth=12,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=1
    )

    candidates["RF_depth16_leaf1"] = RandomForestRegressor(
        n_estimators=800,
        max_depth=16,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=42,
        n_jobs=1
    )

    candidates["RF_depth10_leaf2"] = RandomForestRegressor(
        n_estimators=700,
        max_depth=10,
        min_samples_leaf=2,
        max_features=0.7,
        random_state=42,
        n_jobs=1
    )

    candidates["ExtraTrees_leaf2"] = ExtraTreesRegressor(
        n_estimators=800,
        max_depth=None,
        min_samples_leaf=2,
        max_features=0.9,
        random_state=42,
        n_jobs=1
    )

    candidates["ExtraTrees_leaf1"] = ExtraTreesRegressor(
        n_estimators=1000,
        max_depth=None,
        min_samples_leaf=1,
        max_features=0.8,
        random_state=42,
        n_jobs=1
    )

    candidates["GBR_depth3"] = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            min_samples_leaf=3,
            subsample=0.85,
            random_state=42
        )
    )

    candidates["GBR_depth2"] = MultiOutputRegressor(
        GradientBoostingRegressor(
            n_estimators=700,
            learning_rate=0.025,
            max_depth=2,
            min_samples_leaf=3,
            subsample=0.9,
            random_state=42
        )
    )

    return candidates


def evaluate_model(model, X_test, p_test_base, p_test_true):
    pred_residual = model.predict(X_test)
    pred_residual = np.clip(pred_residual, -20.0, 20.0)

    p_test_pred = p_test_base + pred_residual.T
    test_errors = position_error(p_test_true, p_test_pred)

    mean_error = np.mean(test_errors)
    rmse = np.sqrt(np.mean(test_errors ** 2))

    return mean_error, rmse, test_errors


def main():
    data = sio.loadmat("DH_FR1.mat", squeeze_me=False)

    p = np.asarray(data["p"], dtype=float)
    d_hat = np.asarray(data["d_hat"], dtype=float)
    BS_positions = get_anchor_positions(data)

    num_user = d_hat.shape[1]

    print("========== train.py Data Summary ==========")
    print(f"Total users loaded     : {num_user}")
    print(f"d_hat shape            : {d_hat.shape}")
    print(f"p shape                : {p.shape}")
    print(f"BS_positions shape     : {BS_positions.shape}")
    print("===========================================")
    print()

    X = []
    p_base_all = np.zeros((2, num_user))

    for u in range(num_user):
        feature, p_base = make_feature(d_hat[:, u], BS_positions)
        X.append(feature)
        p_base_all[:, u] = p_base

    X = np.asarray(X)

    y = (p - p_base_all).T

    rng = np.random.default_rng(42)
    indices = np.arange(num_user)
    rng.shuffle(indices)

    train_count = min(400, num_user)

    train_idx = indices[:train_count]
    test_idx = indices[train_count:]

    X_train = X[train_idx]
    y_train = y[train_idx]

    X_test = X[test_idx]
    p_test_true = p[:, test_idx]
    p_test_base = p_base_all[:, test_idx]

    print("========== Split Summary ==========")
    print(f"Train samples          : {len(train_idx)}")
    print(f"Internal test samples  : {len(test_idx)}")
    print("===================================")
    print()

    candidates = build_candidate_models()

    best_name = None
    best_model = None
    best_mean = np.inf
    best_rmse = np.inf
    best_errors = None

    print("========== Model Candidate Test ==========")

    for name, model in candidates.items():
        model.fit(X_train, y_train)

        mean_error, rmse, test_errors = evaluate_model(
            model,
            X_test,
            p_test_base,
            p_test_true
        )

        print(
            f"{name:20s} | "
            f"Mean: {mean_error:.4f} m | "
            f"RMSE: {rmse:.4f} m | "
            f"Acc<=5m: {np.mean(test_errors <= 5.0) * 100:.2f}% | "
            f"Acc<=10m: {np.mean(test_errors <= 10.0) * 100:.2f}%"
        )

        if mean_error < best_mean:
            best_name = name
            best_model = model
            best_mean = mean_error
            best_rmse = rmse
            best_errors = test_errors

    print("==========================================")
    print()

    base_test_errors = position_error(p_test_true, p_test_base)

    print("========== Best Model Internal Test Result ==========")
    print(f"Best model name        : {best_name}")
    print(f"Train samples          : {len(train_idx)}")
    print(f"Internal test samples  : {len(test_idx)}")
    print("----------------------------------------------------")
    print(f"Base Test Mean Error   : {np.mean(base_test_errors):.4f} m")
    print(f"Base Test RMSE         : {np.sqrt(np.mean(base_test_errors ** 2)):.4f} m")
    print("----------------------------------------------------")
    print(f"ML Test Mean Error     : {best_mean:.4f} m")
    print(f"ML Test RMSE           : {best_rmse:.4f} m")
    print(f"ML Test Median Error   : {np.median(best_errors):.4f} m")
    print(f"ML Test Max Error      : {np.max(best_errors):.4f} m")
    print("----------------------------------------------------")
    print(f"Accuracy <= 1m         : {np.mean(best_errors <= 1.0) * 100:.2f} %")
    print(f"Accuracy <= 2m         : {np.mean(best_errors <= 2.0) * 100:.2f} %")
    print(f"Accuracy <= 3m         : {np.mean(best_errors <= 3.0) * 100:.2f} %")
    print(f"Accuracy <= 5m         : {np.mean(best_errors <= 5.0) * 100:.2f} %")
    print(f"Accuracy <= 10m        : {np.mean(best_errors <= 10.0) * 100:.2f} %")
    print("====================================================")

    model_bundle = {
        "algorithm_name": "ASD-Loc Inspired Candidate-Selected ML Residual Correction with Anchor Spatial Features",
        "best_model_name": best_name,
        "rf_model": best_model,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "train_count": len(train_idx),
        "test_count": len(test_idx),
        "feature_dim": X.shape[1],
        "split_random_seed": 42,
        "compressed_by": "gzip_pickle"
    }

    with gzip.open(MODEL_PATH, "wb") as f:
        pickle.dump(model_bundle, f, protocol=pickle.HIGHEST_PROTOCOL)

    print()
    print(f"Saved gzip-compressed model bundle to {MODEL_PATH}")


if __name__ == "__main__":
    main()