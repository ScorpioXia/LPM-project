"""供全部 2D、3D 和跨层特征共享的 v7 脂肪阈值算法。"""

from typing import Tuple

import numpy as np
from scipy.signal import find_peaks
from skimage.filters import threshold_otsu
from sklearn.mixture import GaussianMixture


def is_bimodal_valid(pix_vals: np.ndarray, min_samples: int = 30) -> bool:
    """使用直方图峰谷检验和 Ashman D 判断信号是否具有双峰性。"""
    values = np.asarray(pix_vals, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < min_samples or np.ptp(values) <= np.finfo(float).eps:
        return False

    hist, _ = np.histogram(values, bins="auto")
    distance = max(1, len(hist) // 10)
    peaks, _ = find_peaks(hist, height=0.05 * hist.max(), distance=distance)
    histogram_valid = False
    if len(peaks) >= 2:
        top_two = peaks[np.argsort(hist[peaks])[-2:]]
        left, right = sorted(top_two)
        valley_min = hist[left:right + 1].min()
        valley_depth = min(hist[left], hist[right]) - valley_min
        histogram_valid = valley_depth > 0.1 * max(hist[left], hist[right])

    gmm_valid = False
    try:
        gmm = GaussianMixture(n_components=2, random_state=0, n_init=3)
        gmm.fit(values.reshape(-1, 1))
        means = gmm.means_.ravel()
        variances = gmm.covariances_.ravel()
        if np.all(variances > 0):
            ashman_d = (
                np.sqrt(2.0) * abs(means[0] - means[1])
                / np.sqrt(variances[0] + variances[1])
            )
            gmm_valid = ashman_d > 2.0
    except (ValueError, FloatingPointError):
        gmm_valid = False

    return bool(histogram_valid or gmm_valid)


def get_fat_threshold_per_muscle(
    pix_vals: np.ndarray,
    valid_range: Tuple[float, float] = (0.05, 2.0),
    otsu_range: Tuple[float, float] = (0.3, 0.9),
    fallback: float = 0.6,
) -> Tuple[float, str]:
    """执行 2D v6 阈值策略并返回 ``(阈值, 决策类型)``。

    2D、3D、跨层和多肌肉特征必须调用同一个函数，以保证所有 FIP
    使用完全相同的定义。
    """
    values = np.asarray(pix_vals, dtype=float)
    valid = values[
        np.isfinite(values)
        & (values > valid_range[0])
        & (values < valid_range[1])
    ]
    if valid.size < 10:
        return float(fallback), "too_few_pixels"

    if is_bimodal_valid(valid):
        try:
            threshold = float(threshold_otsu(valid))
            if otsu_range[0] <= threshold <= otsu_range[1]:
                return threshold, "otsu"
        except ValueError:
            pass

    return float(fallback), "fixed_0.6_fallback"

