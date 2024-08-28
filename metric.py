import numpy as np

from dataclasses import dataclass

@dataclass
class MetricsNearestDisatnceResult:
    value_mean: float
    value_max: float
    value_sum: float

def nearest_distance(point: np.array, target_frame: np.ndarray) -> float:
    diff = target_frame - point
    distances = np.linalg.norm(diff, axis=1)
    return distances.min()

def metric_nearest_distance(generated_frame: np.ndarray, target_frame: np.ndarray) -> MetricsNearestDisatnceResult:
    distances = np.zeros([generated_frame.shape[0]])
    for i in range(generated_frame.shape[0]):
        point = generated_frame[i]
        distances[i] = nearest_distance(point, target_frame)

    return MetricsNearestDisatnceResult(
        value_mean=np.mean(distances),  # type: ignore
        value_max=np.max(distances),    # type: ignore
        value_sum=np.sum(distances),    # type: ignore
    )