import os
import gzip
import pickle
import numpy as np
import scipy.io as sio
from scipy.optimize import least_squares


MODEL_PATH = "model.pkl"
_MODEL = None


def get_anchor_positions(data):
    BS_positions = np.asarray(data["BS_positions"], dtype=float)

    if BS_positions.shape == (18, 2):
        BS_positions = BS_positions.T

    return BS_positions


def load_model_once():
    global _MODEL

    if _MODEL is None:
        if os.path.exists(MODEL_PATH):
            with gzip.open(MODEL_PATH, "rb") as f:
                _MODEL = pickle.load(f)
        else:
            _MODEL = False

    return _MODEL


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


def your_algorithm(d_u, BS_positions):
    feature, p_base = make_feature(d_u, BS_positions)

    model_bundle = load_model_once()

    if model_bundle is False:
        return p_base

    if isinstance(model_bundle, dict):
        model = model_bundle["rf_model"]
    else:
        model = model_bundle

    residual_pred = model.predict(feature.reshape(1, -1))[0]
    residual_pred = np.clip(residual_pred, -20.0, 20.0)

    p_final = p_base + residual_pred

    return p_final


def main():
    mat_path = "DH_FR1.mat"

    data = sio.loadmat(mat_path, squeeze_me=False)
    BS_positions = np.asarray(data['BS_positions'], dtype=float)
    d_hat = np.asarray(data["d_hat"], dtype=float)
    p = np.asarray(data['p'], dtype=float)

    num_user = d_hat.shape[1]
    p_hat = np.zeros((2, num_user))
    for u in range(num_user):
        p_hat[:, u] = your_algorithm(d_hat[:, u], BS_positions)

    return p_hat


if __name__ == "__main__":
    main()
    """
    p_hat = main()
    data = sio.loadmat("DH_FR1.mat", squeeze_me=False)
    model_bundle = load_model_once()

    print("========== main.py Final Error Analysis ==========")
    print(f"Processed users        : {p_hat.shape[1]}")
    print(f"Prediction shape       : {p_hat.shape}")
    print("--------------------------------------------------")

    if "p" in data:
        p = np.asarray(data["p"], dtype=float)

        # 전체 데이터 기준 오차
        errors_all = np.linalg.norm(p - p_hat, axis=0)
        errors_all = errors_all[np.isfinite(errors_all)]

        print("========== All 700 Data Error ==========")
        print(f"Valid results          : {len(errors_all)}")
        print(f"Mean Error             : {np.mean(errors_all):.4f} m")
        print(f"RMSE                   : {np.sqrt(np.mean(errors_all ** 2)):.4f} m")
        print(f"Median Error           : {np.median(errors_all):.4f} m")
        print(f"Min Error              : {np.min(errors_all):.4f} m")
        print(f"Max Error              : {np.max(errors_all):.4f} m")
        print(f"Accuracy within 1 m    : {np.mean(errors_all <= 1.0) * 100:.2f} %")
        print(f"Accuracy within 2 m    : {np.mean(errors_all <= 2.0) * 100:.2f} %")
        print(f"Accuracy within 3 m    : {np.mean(errors_all <= 3.0) * 100:.2f} %")
        print(f"Accuracy within 5 m    : {np.mean(errors_all <= 5.0) * 100:.2f} %")
        print(f"Accuracy within 10 m   : {np.mean(errors_all <= 10.0) * 100:.2f} %")
        print("----------------------------------------")

        # 학습에 사용하지 않은 internal test 300개 기준 오차
        if isinstance(model_bundle, dict) and "test_idx" in model_bundle:
            test_idx = np.asarray(model_bundle["test_idx"], dtype=int)

            if len(test_idx) > 0 and np.max(test_idx) < p_hat.shape[1]:
                p_test_hat = p_hat[:, test_idx]
                p_test_true = p[:, test_idx]

                errors_test = np.linalg.norm(p_test_true - p_test_hat, axis=0)
                errors_test = errors_test[np.isfinite(errors_test)]

                print("========== Internal Test 300 Error ==========")
                print(f"Internal test users    : {len(errors_test)}")
                print(f"Mean Error             : {np.mean(errors_test):.4f} m")
                print(f"RMSE                   : {np.sqrt(np.mean(errors_test ** 2)):.4f} m")
                print(f"Median Error           : {np.median(errors_test):.4f} m")
                print(f"Min Error              : {np.min(errors_test):.4f} m")
                print(f"Max Error              : {np.max(errors_test):.4f} m")
                print(f"Accuracy within 1 m    : {np.mean(errors_test <= 1.0) * 100:.2f} %")
                print(f"Accuracy within 2 m    : {np.mean(errors_test <= 2.0) * 100:.2f} %")
                print(f"Accuracy within 3 m    : {np.mean(errors_test <= 3.0) * 100:.2f} %")
                print(f"Accuracy within 5 m    : {np.mean(errors_test <= 5.0) * 100:.2f} %")
                print(f"Accuracy within 10 m   : {np.mean(errors_test <= 10.0) * 100:.2f} %")
                print("=============================================")
            else:
                print("Internal test_idx does not match current data size.")
                print("This can happen when running on hidden test data.")
                print("=============================================")
        else:
            print("No test_idx found in model.pkl.")
            print("Internal test error skipped.")
            print("=============================================")

    else:
        print("Ground truth p not found.")
        print("Only p_hat prediction was generated.")
        print("=============================================")
    """