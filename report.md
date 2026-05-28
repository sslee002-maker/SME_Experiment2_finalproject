# ASD-Loc Inspired Candidate-Selected ML Residual Correction with Anchor Spatial Features

## 1. 모티베이션 & 인트로

본 프로젝트의 목적은 18개의 기지국에서 측정한 RTT 기반 거리 정보 d_hat과 기지국 좌표 BS_positions를 이용하여 사용자 위치 p_hat을 추정하는 것이다. 입력 데이터는 사용자별 18개 RTT 측정값과 18개 기지국 좌표이며, 출력은 각 사용자에 대한 2차원 위치 좌표이다. 전체 데이터는 1000개 사용자로 구성되고, 학생에게는 700개만 제공되며 나머지 300개는 hidden test set으로 평가된다. 따라서 알고리즘은 제공된 700개 데이터에만 과적합되지 않고, hidden test 데이터에서도 자동으로 동작해야 한다.

중간발표까지의 실험에서는 단순 삼변측량의 한계를 확인하는 데 초점을 두었다. 삼변측량은 여러 기지국과 사용자 사이의 거리 정보를 이용하여 위치를 계산하는 직관적인 방식이다. 하지만 실제 RTT 측정값에는 이상적인 거리 정보만 들어 있지 않다. 특정 기지국의 측정값이 순간적으로 튀거나, 실제 거리보다 과도하게 크거나 작게 측정되거나, 일부 기지국에서 반복적으로 큰 residual이 발생할 수 있다. 이러한 상황에서는 모든 센서의 거리값을 동일하게 믿고 계산하는 단순 삼변측량이 큰 오차를 만들 수 있다.

중간발표 자료에서는 이 문제를 “삼변측량 한계”로 정리하였다. 거리 측정 오차가 존재하면 각 센서가 만드는 원 또는 거리 제약이 실제 사용자 위치에서 일관되게 만나지 않고, 잘못된 교차점 후보가 동시에 발생한다. 즉, 실제 위치 근처의 후보와 outlier에 의해 만들어진 잘못된 후보가 섞이면서 최종 위치를 판단하기 어려워진다. 따라서 단순히 모든 센서의 거리값을 그대로 넣어 위치를 계산하는 방식은 충분하지 않다고 판단하였다.

중간발표에서 제안한 흐름은 DBSCAN, MAD, Adaptive MCC를 활용한 robust localization이었다. 당시 알고리즘의 큰 흐름은 median 기반 전처리, range filter, DBSCAN 기반 후보 위치 생성, residual 계산, MAD 기반 outlier 판정, Adaptive MCC 기반 최종 최적화였다. 이 흐름의 목적은 특정 알고리즘 자체를 사용하는 것보다, 위치 추정에서 “어떤 센서를 믿고 어떤 센서의 영향력을 줄일 것인가”를 결정하는 것이었다.

중간발표에서 얻은 첫 번째 고찰은 초기 위치 후보가 중요하다는 점이었다. 거리 측정값에 오차가 있으면 모든 센서쌍의 교차점이 실제 위치 주변에만 생기지 않는다. 일부 교차점은 실제 위치 근처에 모이지만, 측정 오차가 큰 센서가 포함된 교차점은 멀리 떨어진 잘못된 후보를 만든다. 중간발표에서는 이 문제를 DBSCAN 기반 후보 생성으로 해결하려고 하였다. 즉, 여러 교차점 중 밀집된 군집을 실제 위치에 가까운 후보로 보는 방식이었다.

두 번째 고찰은 residual 기반 센서 검증의 필요성이었다. 초기 위치 후보가 정해지면, 각 센서에 대해 측정 거리와 예측 거리의 차이를 residual로 계산할 수 있다. 중간발표 자료에서는 정상 센서들은 residual이 작게 모이는 반면, outlier 센서는 residual이 매우 크게 튀는 사례를 확인하였다. 이 결과는 위치 추정에서 모든 센서가 같은 품질의 정보를 제공하지 않는다는 점을 보여준다. 따라서 센서별 residual을 계산하고, residual이 큰 센서의 영향력을 줄이는 과정이 필요하다고 판단하였다.

세 번째 고찰은 평균과 표준편차보다 중앙값과 MAD가 outlier에 강하다는 점이었다. 평균과 표준편차는 큰 outlier 하나만 있어도 값이 크게 흔들릴 수 있다. 그러면 오히려 진짜 outlier가 정상 범위 안에 들어오는 문제가 발생할 수 있다. 반면 중앙값과 MAD는 outlier가 있어도 통계량이 상대적으로 안정적이다. 따라서 중간발표에서는 MAD를 이용해 명확한 outlier를 판정하는 방향을 검토하였다.

네 번째 고찰은 outlier를 단순히 제거하는 것만으로는 충분하지 않을 수 있다는 점이었다. 어떤 센서는 완전히 잘못된 outlier일 수 있지만, 어떤 센서는 애매하게 신뢰도가 낮은 측정값일 수 있다. 이 경우 모든 센서를 in 또는 out으로만 나누는 hard decision보다, residual 크기에 따라 영향력을 부드럽게 줄이는 soft weighting이 더 안정적일 수 있다. 중간발표에서는 이 역할을 Adaptive MCC가 담당하였다. MCC의 핵심은 큰 residual의 영향력이 무한히 커지지 않도록 제한하여, outlier 하나가 최적화 전체를 지배하지 못하게 하는 것이다.

따라서 중간발표에서 최종 프로젝트로 이어진 핵심 아이디어는 다음과 같다. 첫째, 단순 삼변측량은 outlier에 취약하다. 둘째, 위치 추정에서는 센서별 신뢰도를 다르게 반영해야 한다. 셋째, residual은 센서 신뢰도를 판단하는 중요한 기준이 될 수 있다. 넷째, 중앙값과 MAD처럼 outlier에 강한 통계량을 사용해야 한다. 다섯째, 큰 residual의 영향력을 줄이는 robust optimization 또는 weighting 구조가 필요하다.

최종 프로젝트에서는 이 문제의식을 18개 기지국 RTT 데이터에 맞게 재구성하였다. 중요한 점은 최종 코드에서 DBSCAN과 MCC를 직접 구현하지 않았다는 것이다. 중간발표의 DBSCAN-MAD-MCC 구조를 그대로 복사한 것이 아니라, 그 실험에서 얻은 “측정값별 신뢰도를 다르게 반영해야 한다”는 고찰을 바탕으로 최종 알고리즘을 설계하였다. 최종 구현에서는 DBSCAN 기반 교차점 군집화 대신 weighted least squares 기반의 p_base를 계산하였고, MCC 대신 scipy.optimize.least_squares의 soft_l1 robust loss와 residual 기반 가중치를 사용하였다. MAD 개념은 residual 분포의 robust scale을 계산하는 데 반영하였다.

최종 알고리즘의 high-level 구조는 물리 기반 추정과 머신러닝 기반 보정을 결합하는 방식이다. 먼저 거리 기반 신뢰도와 residual 기반 신뢰도를 이용해 ASD-Loc inspired robust weighted trilateration을 수행하고, 이를 통해 1차 위치 p_base를 계산한다. 그 다음 머신러닝 모델이 p_base와 실제 위치 p_true 사이의 residual, 즉 p_true - p_base를 학습한다. 마지막으로 main.py에서는 학습된 model.pkl을 불러와 새로운 d_hat이 들어왔을 때 p_base를 계산하고, 머신러닝 모델이 예측한 residual을 더해 최종 위치 p_hat을 출력한다.

이 방식은 단순히 머신러닝이 좌표를 처음부터 직접 예측하는 방식이 아니다. 먼저 기하학적으로 의미 있는 p_base를 계산한 뒤, 머신러닝은 p_base가 반복적으로 틀리는 패턴만 보정한다. 제공 데이터가 700개로 많지 않기 때문에, 좌표 전체를 직접 학습하는 복잡한 딥러닝 모델보다 residual correction 방식이 더 안정적이라고 판단하였다. 또한 가까운 기지국의 좌표 중심, 거리 가중 anchor 중심, p_base와 anchor 중심의 차이 등을 feature로 추가하여, 단순 거리값뿐 아니라 기지국 배치에 따른 공간적 오차 패턴도 학습하도록 하였다.

| 구분 | 중간발표에서 얻은 아이디어 | 최종 코드에서의 실제 반영 |
|---|---|---|
| 삼변측량 한계 | outlier가 있으면 위치가 크게 흔들림 | 단순 삼변측량을 baseline으로 두고 robust 방식 필요성 확인 |
| DBSCAN | 교차점 후보 중 밀집 영역을 초기 위치로 사용 | 최종 코드에서는 직접 사용하지 않고, hidden test 안정성을 위해 weighted trilateration으로 대체 |
| MAD | 명확한 residual outlier에 강한 통계량 | residual 기반 robust scale 계산에 사용 |
| Adaptive MCC | 큰 residual의 영향력을 부드럽게 감소 | MCC를 직접 쓰지 않고 soft_l1 loss와 residual weighting으로 robust 효과 구현 |
| 센서 신뢰도 | 모든 센서를 동일하게 보지 않음 | 거리 기반 가중치와 residual 기반 가중치 사용 |
| 최종 보정 | robust optimization 중심 | ML residual correction으로 반복적 오차 패턴 보정 |

본 프로젝트의 최종 알고리즘 이름은 ASD-Loc Inspired Candidate-Selected ML Residual Correction with Anchor Spatial Features이다. 이름에서 알 수 있듯이, 이 알고리즘은 ASD-Loc처럼 센서 신뢰도 개념을 반영하고, p_base의 residual을 머신러닝으로 보정하며, 여러 후보 모델 중 internal test 평균 오차가 가장 낮은 모델을 선택한다.

| 항목 | 내용 |
|---|---|
| 알고리즘 이름 | ASD-Loc Inspired Candidate-Selected ML Residual Correction with Anchor Spatial Features |
| 입력 | d_hat[:, u], BS_positions |
| 출력 | 사용자 위치 추정값 p_hat[:, u] |
| 핵심 구조 | Robust weighted trilateration + anchor spatial feature + ML residual correction |
| 중간발표에서 이어진 아이디어 | outlier 대응, residual 기반 센서 신뢰도, robust optimization |
| 최종 코드에서 직접 사용한 기법 | 거리 기반 가중치, MAD 기반 residual scale, soft_l1 robust loss, RandomForest residual correction |
| 최종 코드에서 직접 사용하지 않은 기법 | DBSCAN, MCC |
| 학습 방식 | 제공 700개 중 400개 train, 300개 internal test |
| 최종 모델 선택 기준 | internal test Mean Error 최소 |
| hidden test 대응 | d_hat.shape[1]로 사용자 수 자동 처리 |

## 2. 알고리즘 설명

본 알고리즘은 크게 다섯 단계로 구성된다. 첫 번째는 입력 데이터 전처리, 두 번째는 ASD-Loc inspired robust weighted trilateration, 세 번째는 feature 설계, 네 번째는 residual learning, 다섯 번째는 후보 모델 비교 및 최종 모델 선택이다.

첫 번째 단계에서는 하나의 사용자에 대한 RTT 기반 거리 벡터 d_u를 입력받는다. d_u는 18개의 기지국 측정값으로 구성된다. 만약 거리값 중 NaN, inf, 0 이하의 값이 존재하면 계산 안정성을 위해 유효한 거리값의 중앙값으로 대체한다. 이 처리는 일부 비정상 측정값 때문에 최적화 계산이 실패하거나 위치 추정이 과도하게 흔들리는 것을 방지하기 위한 것이다.

두 번째 단계는 ASD-Loc inspired robust weighted trilateration이다. 사용자 위치 후보를 x = [x, y]^T, i번째 기지국 위치를 b_i = [b_xi, b_yi]^T, i번째 측정 거리를 d_i라고 하면, 위치 x에서 i번째 기지국까지의 예측 거리와 residual은 다음과 같이 정의된다.

| 수식 | 의미 |
|---|---|
| hat_d_i(x) = sqrt((x - b_xi)^2 + (y - b_yi)^2) | 위치 x에서 i번째 기지국까지의 예측 거리 |
| r_i(x) = hat_d_i(x) - d_i | 예측 거리와 측정 거리의 차이 |

기본 least squares는 모든 residual을 같은 중요도로 다룬다. 하지만 중간발표에서 확인했듯이 outlier residual이 하나만 있어도 L2 손실에서는 그 영향이 제곱되어 전체 추정값을 왜곡할 수 있다. 따라서 본 프로젝트에서는 기지국별 신뢰도를 반영하였다. 먼저 가까운 기지국일수록 위치 추정에 더 직접적인 영향을 줄 가능성이 크다고 보고, 거리 기반 초기 가중치를 사용하였다.

| 수식 | 의미 |
|---|---|
| w_i = 1 / (d_i + epsilon)^2 | 거리 기반 초기 신뢰도 |
| w_i = w_i / max(w_i) | 정규화된 거리 기반 가중치 |

거리 기반 가중치를 사용하여 1차 robust least squares를 수행하면 초기 위치 p_stage1을 얻을 수 있다. 이때 최적화에는 일반적인 L2 loss만 사용하지 않고 soft_l1 loss를 사용하였다. soft_l1 loss는 residual이 작은 영역에서는 일반적인 least squares와 유사하게 동작하지만, residual이 큰 영역에서는 outlier의 영향이 과도하게 커지는 것을 완화한다. 따라서 중간발표의 MCC와 같은 식을 직접 사용한 것은 아니지만, 큰 residual의 영향력을 제한한다는 robust optimization 목적은 유사하다.

1차 추정 이후에도 일부 측정값은 여전히 큰 residual을 만들 수 있다. 따라서 1차 결과를 기준으로 각 기지국의 residual 크기 e_i를 계산하고, residual이 큰 기지국의 신뢰도를 낮추었다. 이 과정은 중간발표에서 MAD가 outlier에 강한 통계량으로 사용된 흐름과 연결된다.

| 수식 | 의미 |
|---|---|
| e_i = |r_i(p_stage1)| | 1차 추정 위치에서의 residual 크기 |
| MAD = median(|e_i - median(e)|) | residual 분포의 robust spread |
| s = 1.4826 × MAD + epsilon | residual 기반 robust scale |
| q_i = 1 / (1 + (e_i / s)^2) | residual 기반 신뢰도 |
| w_final_i = w_i × q_i | 최종 기지국 신뢰도 |

최종 가중치 w_final_i는 거리 기반 신뢰도와 residual 기반 신뢰도를 함께 반영한다. 이 가중치를 사용하여 2차 robust least squares를 수행하고, 그 결과를 p_base로 정의하였다. p_base는 머신러닝 보정 이전의 물리 기반 1차 위치 추정값이다.

세 번째 단계는 feature 설계이다. 단순히 18개 d_hat 값만 머신러닝 모델에 넣으면, 모델은 기지국의 공간적 배치와 p_base의 신뢰도를 충분히 알기 어렵다. 따라서 본 프로젝트에서는 RTT 거리 feature뿐 아니라 anchor geometry를 반영하는 feature를 함께 사용하였다.

| Feature 종류 | 설명 |
|---|---|
| 원본 RTT 거리 feature | 18개 d_hat 값 |
| 정렬 거리 feature | 가까운 거리 순서로 정렬한 일부 거리값 |
| 거리 통계 feature | 평균, 표준편차, 중앙값, 최솟값, 최댓값, 분위수 |
| p_base 좌표 | robust trilateration으로 얻은 1차 위치 |
| residual 통계 feature | p_base 기준 예측 거리와 측정 거리 차이의 통계량 |
| 가까운 기지국 중심 | 가까운 3개, 5개 기지국 좌표의 평균 |
| 거리 가중 anchor 중심 | 거리의 역제곱 가중치로 계산한 기지국 중심 |
| 위치 차이 feature | p_base와 anchor 중심 사이의 차이 |
| 가까운 기지국 index | 가장 가까운 기지국들의 index 정보 |

가까운 기지국 중심 feature를 넣은 이유는 사용자의 대략적인 공간 영역을 표현하기 위해서이다. 예를 들어 가장 가까운 3개 기지국이 특정 영역에 몰려 있다면, 사용자의 실제 위치도 그 근처일 가능성이 높다. 또한 p_base와 가까운 기지국 중심의 차이는 “물리 기반 추정 결과가 anchor geometry와 얼마나 어긋나는지”를 나타낸다. 이러한 feature는 tree ensemble 모델이 지역별 오차 패턴을 학습하는 데 도움을 준다.

네 번째 단계는 residual learning이다. 머신러닝 target은 실제 좌표 자체가 아니라, 실제 위치와 p_base 사이의 차이로 정의하였다.

| 수식 | 의미 |
|---|---|
| y_residual = p_true - p_base | 머신러닝이 학습할 위치 보정량 |
| p_final = p_base + f(feature) | residual model 기반 최종 위치 |

이 구조는 좌표를 처음부터 예측하는 direct prediction보다 안정적이다. p_base는 이미 기하학적 계산을 통해 얻은 위치이므로, 머신러닝 모델은 전체 위치를 새로 학습하는 것이 아니라 p_base의 반복적 오차만 학습하면 된다. 이는 제공 데이터 수가 제한적인 상황에서 과적합 위험을 줄이는 데 유리하다고 판단하였다.

다섯 번째 단계는 후보 모델 비교이다. 제공 데이터 700개 중 400개를 학습용으로 사용하고, 나머지 300개를 internal test set으로 사용하였다. 학습 데이터 400개로 여러 후보 모델을 학습하고, 300개 internal test에서 평균 위치 오차가 가장 낮은 모델을 최종 모델로 선택하였다.

| 후보 모델 | 목적 |
|---|---|
| RF_depth12_leaf2 | 안정적인 기본 RandomForest residual correction |
| RF_depth16_leaf1 | 더 깊은 tree를 허용하여 복잡한 residual pattern 확인 |
| RF_depth10_leaf2 | 더 얕은 tree로 과적합 완화 가능성 확인 |
| ExtraTrees_leaf2 | 더 랜덤한 tree ensemble의 일반화 가능성 확인 |
| ExtraTrees_leaf1 | 더 강한 ExtraTrees 모델 확인 |
| GBR_depth3 | 순차적 오차 보정 방식의 GradientBoosting 확인 |
| GBR_depth2 | 더 얕은 GradientBoosting으로 과적합 완화 확인 |

최종적으로 internal test 평균 오차가 가장 낮은 RF_depth12_leaf2를 선택하였다. 이 모델은 n_estimators = 500, max_depth = 12, min_samples_leaf = 2, max_features = 0.8 조건을 사용하였다. max_depth를 제한한 이유는 학습 데이터가 400개로 많지 않기 때문에 너무 깊은 tree가 training data에만 맞춰지는 것을 막기 위해서이다. min_samples_leaf = 2는 leaf node가 너무 작은 샘플에만 반응하는 것을 줄이기 위한 설정이다.

ML 사용 학생은 학습된 모델 파일을 제출해야 하므로, 최종 모델은 model.pkl로 저장하였다. 모델 저장은 Python 기본 내장 모듈인 gzip과 pickle을 사용해 압축 저장하였다. 이 방식은 추가 패키지 설치 없이 동작하며, GitHub의 단일 파일 크기 제한을 피하면서도 main.py에서 동일한 모델을 복원하여 inference할 수 있게 한다.

### 참고 개념 및 본인 기여 구분

본 프로젝트에서는 특정 논문을 그대로 재현하거나 공개된 ASD-Loc 코드를 직접 사용한 것이 아니다. 다만 중간발표에서 다룬 DBSCAN, MAD, Adaptive MCC 기반 robust localization 흐름과 일반적인 weighted least squares, robust estimation, residual learning 개념을 참고하였다. 따라서 참고한 개념과 본 프로젝트에서 새롭게 구성한 부분을 구분하여 정리하였다.

| 참고 개념 | 기존 개념의 핵심 | 본 프로젝트에서의 적용 및 차이점 |
|---|---|---|
| 삼변측량 | 여러 기지국과 사용자 사이의 거리 정보를 이용해 위치를 계산 | baseline으로 사용하였고, outlier에 취약하다는 한계를 확인하는 기준으로 활용 |
| Weighted Least Squares | 측정값마다 다른 가중치를 부여하여 least squares를 수행 | 거리 기반 신뢰도와 residual 기반 신뢰도를 결합하여 기지국별 영향력을 다르게 설정 |
| MAD | 중앙값 기반으로 outlier에 강한 통계량을 계산 | residual 분포의 robust scale을 계산하여 residual이 큰 기지국의 신뢰도를 낮추는 데 활용 |
| MCC / Robust Loss | 큰 residual의 영향력을 제한하여 outlier에 덜 민감하게 최적화 | MCC를 직접 구현하지 않고, soft_l1 loss와 residual weighting을 사용하여 robust optimization 효과를 구현 |
| DBSCAN 기반 후보 생성 | 교차점 후보 중 밀집된 위치 후보를 찾음 | 최종 코드에서는 직접 사용하지 않고, hidden test 안정성과 실행 시간을 고려하여 weighted trilateration 기반 p_base 계산으로 대체 |
| ASD-Loc의 신뢰도 기반 위치 추정 관점 | 센서별 측정 신뢰도를 다르게 반영 | 거리 기반 가중치, residual 기반 가중치, anchor spatial feature, ML residual correction으로 재구성 |
| Fingerprint localization | 측정 패턴과 위치 사이의 관계를 데이터 기반으로 학습 | 18개 RTT 값, 거리 통계, anchor spatial feature를 이용해 p_base의 residual pattern을 학습 |

본 프로젝트의 차별점은 위 참고 개념들을 각각 독립적으로 사용하는 것이 아니라, 제공된 700개 데이터와 hidden test 평가 구조에 맞게 하나의 pipeline으로 결합한 점이다. 중간발표의 DBSCAN-MAD-MCC 흐름은 outlier 대응과 센서 신뢰도 반영의 필요성을 보여주었고, 최종 프로젝트에서는 이를 18개 기지국 데이터에 맞게 거리 기반 가중치, residual 기반 가중치, anchor spatial feature, RandomForest residual correction으로 재구성하였다.

| 본인 기여 | 설명 |
|---|---|
| ASD-Loc inspired base position 구성 | 거리 기반 신뢰도와 residual 기반 신뢰도를 결합하여 p_base를 계산 |
| Anchor spatial feature 설계 | 가까운 기지국 중심, 거리 가중 중심, p_base와 anchor 중심 차이를 feature로 추가 |
| ML residual correction 적용 | 실제 좌표를 직접 예측하지 않고 p_true - p_base를 학습하여 물리 기반 추정 오차만 보정 |
| Candidate-selected model selection | 여러 tree ensemble 후보를 동일한 400/300 split에서 비교하여 internal test 평균 오차가 가장 낮은 모델 선택 |
| Hidden test 대응 구조 | main.py에서 학습을 수행하지 않고, model.pkl을 load하여 d_hat.shape[1] 기준으로 모든 사용자 수에 대응 |

## 3. Agent AI 활용 방안

본 프로젝트에서는 Agent AI를 알고리즘 아이디어 탐색, 코드 구조 정리, 오류 원인 분석, 실험 결과 해석 보조 도구로 활용하였다. 단, 최종 알고리즘 선택과 성능 판단은 제공 데이터에서 직접 실행한 결과를 기준으로 본인이 결정하였다.

Agent AI는 먼저 다양한 후보 접근을 제안하는 데 사용되었다. 예를 들어 기본 삼변측량, robust weighted trilateration, RandomForest residual correction, ExtraTrees, GradientBoosting, MLPRegressor, KNN fingerprint correction, Direct-Residual Hybrid 방식 등이 후보로 검토되었다. 이 중 실제 실험 결과가 낮거나 hidden test 일반화 가능성이 떨어진다고 판단한 방식은 최종 알고리즘에서 제외하였다.

예를 들어 KNN fingerprint correction은 RTT pattern이 비슷한 샘플은 실제 위치도 비슷할 것이라는 직관에서 출발하였다. 하지만 internal test 기준에서 KNN 단독 및 RF+KNN hybrid 방식은 RandomForest residual correction보다 낮은 성능을 보였다. 따라서 KNN은 참신한 아이디어였지만 최종 모델에는 포함하지 않았다.

MLPRegressor 기반 신경망도 실험하였다. 그러나 제공 데이터가 700개로 제한적이고, 그중 internal test를 제외하면 학습에 사용하는 데이터는 400개뿐이므로 MLP는 tree ensemble보다 안정적인 성능을 내지 못하였다. 이에 따라 더 깊은 모델을 사용하는 것보다 데이터 수와 feature 특성에 맞는 모델을 선택하는 것이 더 중요하다고 판단하였다.

또한 Agent AI는 코드 오류를 해결하는 데도 사용되었다. 예를 들어 초기에는 데이터 파일의 기지국 변수명을 정확히 확인하지 못해 p_bs와 BS_positions 중 어떤 이름을 사용해야 하는지 혼동이 있었다. 최종 코드에서는 내가 실제로 사용한 데이터 변수명과 실행 결과를 기준으로 기지국 좌표를 불러오도록 정리하였다. 또한 model.pkl 파일 크기가 GitHub 100MB 제한을 넘는 문제가 있었고, 이에 대해 Agent AI가 가능한 원인을 제시하였다. 본인은 실제 실행 로그를 확인하면서 Python 기본 내장 모듈인 gzip과 pickle을 사용하는 압축 저장 방식으로 최종 정리하였다.

| 구분 | Agent AI 활용 | 본인 판단 및 결정 |
|---|---|---|
| 알고리즘 후보 탐색 | 여러 위치 추정 및 ML 보정 구조 제안 | 실제 validation 결과로 최종 구조 선택 |
| 중간발표 내용 연결 | DBSCAN, MAD, MCC 흐름을 최종 구조와 연결하는 방향 제안 | 그대로 복사하지 않고 최종 데이터셋에 맞게 재구성 |
| feature 설계 | anchor spatial feature 추가 아이디어 제안 | 성능 개선 여부 확인 후 유지 결정 |
| 모델 후보 비교 | RF, ExtraTrees, GBR, MLP 비교 방향 제안 | internal test 평균 오차 기준으로 최종 모델 선택 |
| 코드 오류 해결 | 오류 원인 후보 제시 | 실행 로그 확인 후 직접 수정 |
| 제출 준비 | GitHub 업로드, model size 문제 해결 방향 제안 | 최종 제출 파일 구성 직접 확인 |
| 보고서 작성 | 문장 구조와 표 정리 보조 | 실험 흐름과 결과 해석 직접 반영 |

따라서 본 프로젝트에서 Agent AI는 코드를 대신 완성한 주체가 아니라, 설계 과정에서 가능한 선택지를 넓히고 오류 해결을 돕는 보조 도구였다. 최종 feature 선택, validation 방식, 모델 선택, 결과 해석은 본인이 직접 실행한 결과를 바탕으로 결정하였다.

## 4. 결과 도출 & 디스커션

성능 평가는 예측 좌표 p_hat과 실제 좌표 p 사이의 2D Euclidean distance error를 기준으로 수행하였다. 제공 데이터 700개 중 400개를 학습용으로 사용하고, 나머지 300개를 internal test set으로 사용하였다. 이 방식은 학습 데이터와 검증 데이터를 분리하므로, 전체 700개를 다시 평가하는 방식보다 hidden test 상황에 더 가까운 평가라고 판단하였다.

사용한 지표는 다음과 같다.

| 지표 | 의미 |
|---|---|
| Mean Error | 모든 샘플의 Euclidean distance error 평균 |
| RMSE | 큰 오차에 더 민감한 root mean square error |
| Median Error | 중앙 샘플의 위치 오차 |
| Min Error | 가장 작게 틀린 샘플의 위치 오차 |
| Max Error | 가장 크게 틀린 샘플의 위치 오차 |
| Accuracy within R m | 오차가 R m 이하인 샘플 비율 |

먼저 단순 삼변측량은 평균 오차가 약 23.2115 m로 나타났다. 이는 RTT 측정값에 포함된 큰 오차나 일부 기지국 측정 왜곡에 취약하기 때문으로 해석된다. ASD-Loc inspired robust trilateration을 적용한 base model은 internal test 기준 평균 오차를 9.9173 m까지 낮추었다. 이후 anchor spatial feature와 ML residual correction을 적용하자 internal test 평균 오차는 5.2456 m까지 감소하였다.

| 방법 | 평가 방식 | Mean Error (m) | RMSE (m) | Median Error (m) | Min Error (m) | Max Error (m) | Acc ≤ 1 m | Acc ≤ 2 m | Acc ≤ 3 m | Acc ≤ 5 m | Acc ≤ 10 m |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 단순 삼변측량 | 제공 700개 기준 | 23.2115 | 25.0171 | 21.7254 | 2.2154 | 85.5485 | 0.00% | 0.00% | 0.14% | 0.29% | 2.57% |
| ASD-Loc inspired base | 400/300 internal test | 9.9173 | 11.8921 | - | - | - | - | - | - | - | - |
| ML residual correction with anchor spatial features | 400/300 internal test | 5.2456 | 6.7988 | 4.1309 | 0.4571 | 28.4933 | 6.00% | 21.00% | 36.67% | 60.00% | 88.33% |
| 전체 데이터 재평가 | 제공 700개 전체 | 3.6360 | 5.3367 | 2.4184 | 0.0250 | 41.5302 | 16.29% | 41.14% | 59.29% | 77.86% | 94.14% |

위 결과에서 대표 성능으로 볼 값은 400/300 internal test 결과이다. 전체 데이터 재평가 결과는 학습에 사용된 400개 샘플도 포함하므로 실제 일반화 성능보다 좋게 보일 수 있다. 따라서 hidden test에 가까운 자체 평가로는 internal test 평균 오차 5.2456 m를 사용하는 것이 더 공정하다.

최종 제출 직전 동일한 model.pkl을 사용하여 main.py를 실행한 결과, 제공된 700개 전체 데이터에 대한 Mean Error는 3.6360 m로 확인되었다. 그러나 이 값은 학습에 사용된 400개 sample이 포함된 전체 재평가 결과이므로, hidden test 성능을 대표하는 값으로 사용하지 않고 참고용으로만 해석하였다.

모델 후보 비교 결과는 다음과 같다.

| 후보 모델 | Mean Error (m) | RMSE (m) | Acc ≤ 5 m | Acc ≤ 10 m |
|---|---:|---:|---:|---:|
| RF_depth12_leaf2 | 5.2456 | 6.7988 | 60.00% | 88.33% |
| RF_depth16_leaf1 | 5.2530 | 6.8036 | 61.00% | 88.00% |
| RF_depth10_leaf2 | 5.2494 | 6.7926 | 60.67% | 89.00% |
| ExtraTrees_leaf2 | 5.3053 | 6.9654 | 61.67% | 88.00% |
| ExtraTrees_leaf1 | 5.2686 | 6.8692 | 61.67% | 88.00% |
| GBR_depth3 | 5.4558 | 6.9887 | 57.67% | 86.67% |
| GBR_depth2 | 5.7530 | 7.3149 | 54.33% | 84.00% |

가장 낮은 평균 오차는 RF_depth12_leaf2에서 나타났다. RF_depth16_leaf1과 ExtraTrees 계열은 Accuracy within 5 m가 일부 더 높았지만, 평균 오차와 RMSE가 RF_depth12_leaf2보다 낮지 않았다. 위치 추정 문제에서는 일부 샘플을 5 m 안에 넣는 것뿐만 아니라 전체 평균 오차와 큰 오차를 함께 줄이는 것이 중요하다고 판단하였다. 따라서 최종 모델은 RF_depth12_leaf2로 선택하였다.

본 알고리즘의 장점은 다음과 같다.

| 장점 | 설명 |
|---|---|
| 물리 기반 구조 유지 | robust trilateration으로 1차 위치를 먼저 계산하여 완전히 데이터에만 의존하지 않음 |
| 측정 신뢰도 반영 | 거리 기반 가중치와 residual 기반 가중치를 함께 사용 |
| 지역적 오차 학습 | p_true - p_base residual을 학습하여 반복적인 위치 편향을 보정 |
| 공간 feature 활용 | 가까운 기지국 중심과 거리 가중 anchor 중심을 사용하여 위치 영역 정보를 반영 |
| hidden test 대응 | d_hat.shape[1]로 사용자 수를 받아 300개 hidden test에서도 동작 |
| 역할 분리 | train.py는 학습, main.py는 model load와 inference만 수행 |
| 제출 안정성 | model.pkl을 gzip과 pickle로 압축 저장하여 추가 패키지 의존성을 줄임 |

한계도 존재한다.

| 한계 | 설명 |
|---|---|
| 데이터 수 제한 | 제공 데이터 700개 중 400개만 학습에 사용하면 공간 패턴 학습에 한계가 있음 |
| 큰 오차 샘플 존재 | Internal test Max Error가 28.4933 m로 일부 샘플에서 큰 오차가 남음 |
| split 의존성 | 400/300 split 하나를 기준으로 모델을 선택했기 때문에 다른 split에서는 결과가 달라질 수 있음 |
| hidden test 불확실성 | hidden 300개의 공간 분포가 제공 데이터와 다르면 성능이 달라질 수 있음 |
| 모델 파일 용량 | tree ensemble 모델은 저장 용량이 커질 수 있어 Python 기본 내장 gzip과 pickle을 이용해 압축 저장함 |

본 성능 비교는 단순 삼변측량과 머신러닝 보정 모델을 비교한다는 점에서 완전히 동일한 조건의 비교는 아니다. 단순 삼변측량은 정답 위치 p를 학습에 사용하지 않지만, proposed method는 제공된 학습 데이터의 정답 위치를 이용해 residual pattern을 학습한다. 따라서 본 보고서에서는 머신러닝이 무조건 삼변측량보다 우수하다고 주장하기보다, 물리 기반 robust trilateration으로 얻은 p_base의 반복적 오차를 학습 데이터 기반으로 보정했을 때 internal test에서 성능이 개선되었다고 해석하였다.

또한 전체 700개를 다시 평가하면 평균 오차가 3.6360 m까지 낮아진다. 하지만 이 값은 학습 데이터가 포함된 재평가 결과이므로 hidden test 성능을 대표한다고 보기 어렵다. 본 프로젝트에서는 이 값을 참고용으로만 사용하고, 모델 선택과 성능 논의는 internal test 결과를 중심으로 하였다.

향후 개선 방향은 다음과 같다.

| 개선 방향 | 설명 |
|---|---|
| 5-fold OOF validation | 하나의 400/300 split에 의존하지 않고 전체 700개에 대해 더 안정적인 검증 수행 |
| outlier sample 분석 | Max Error가 큰 샘플의 RTT pattern을 분석하여 별도 보정 |
| confidence-aware correction | residual 크기나 feature 분포에 따라 보정 강도를 샘플별로 조절 |
| anchor geometry feature 개선 | 기지국 배치의 면적, 분산, GDOP 유사 지표 추가 |
| lightweight model 설계 | 성능을 유지하면서 model.pkl 크기와 inference 시간을 줄이는 방향 |

최종적으로 본 알고리즘은 단순 삼변측량 대비 큰 폭의 성능 개선을 보였고, 중간발표에서 도출한 outlier 대응과 residual 기반 센서 신뢰도 아이디어를 최종 데이터셋에 맞게 확장하였다. 특히 DBSCAN-MAD-MCC 흐름에서 얻은 “측정값을 동일하게 신뢰하지 말아야 한다”는 핵심 고찰을 거리 기반 가중치, residual 기반 가중치, anchor spatial feature, ML residual correction으로 재구성하였다. 또한 train.py와 main.py의 역할을 분리하고, main.py에서 학습을 수행하지 않도록 하여 hidden test 및 실행 시간 제한 조건에 맞는 구조로 구현하였다.
