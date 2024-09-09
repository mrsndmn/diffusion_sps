import numpy as np

from pydantic import BaseModel
class MetricsNearestDisatnceResult(BaseModel):
    value_mean: float
    value_max: float
    value_sum: float

    value_external_mean: float
    value_external_max: float
    value_external_sum: float

    one_nna: float

def nearest_distance(point: np.array, target_frame: np.ndarray) -> float:
    diff = target_frame - point
    distances = np.linalg.norm(diff, axis=1)
    return distances.min()


def metric_nearest_distance(generated_frame: np.ndarray, target_frame: np.ndarray) -> MetricsNearestDisatnceResult:

    import time
    start_metric_calc = time.time()

    one_nna_generated_count = 0
    one_nna_references_count = 0

    len_generated = generated_frame.shape[0]
    len_target = target_frame.shape[0]

    distances = np.zeros([generated_frame.shape[0]])

    all_points = np.concatenate([generated_frame, target_frame], axis=0)
    all_but_one_distances = np.zeros([ all_points.shape[0]-1 ])
    for i in range(generated_frame.shape[0]):
        point = generated_frame[i]
        # distances[i] = nearest_distance(point, target_frame)

        diff = all_points - point
        distances_to_point = np.linalg.norm(diff, axis=1)

        distances_to_references = distances_to_point[len_generated:]
        distances[i] = distances_to_references.min()


        all_but_one_distances[:i] = distances_to_point[:i]
        all_but_one_distances[i:] = distances_to_point[i+1:]
        nearest_neigh_i = all_but_one_distances.argmin()
        if nearest_neigh_i < len_generated - 1:
            one_nna_generated_count += 1

    external_distances = np.zeros([target_frame.shape[0]])
    for i in range(target_frame.shape[0]):
        point = target_frame[i]
        # external_distances[i] = nearest_distance(point, generated_frame)

        diff = all_points - point
        distances_to_point = np.linalg.norm(diff, axis=1)

        distances_to_generated = distances_to_point[:len_generated]
        external_distances[i] = distances_to_generated.min()

        all_but_one_distances[:len_generated+i] = distances_to_point[:len_generated+i]
        all_but_one_distances[len_generated+i:] = distances_to_point[len_generated+i+1:]
        nearest_neigh_i = all_but_one_distances.argmin()
        if nearest_neigh_i >= len_generated:
            one_nna_references_count += 1

    print("one_nna_generated_count, one_nna_references_count", one_nna_generated_count, one_nna_references_count, "len_generated, len_target", len_generated, len_target)
    one_nna = (one_nna_generated_count + one_nna_references_count) / (len_generated + len_target)

    metric_calc_duration = time.time() - start_metric_calc
    print("metric calc duration", metric_calc_duration)

    return MetricsNearestDisatnceResult(
        value_mean=np.mean(distances),  # type: ignore
        value_max=np.max(distances),    # type: ignore
        value_sum=np.sum(distances),    # type: ignore

        value_external_mean=np.mean(external_distances),  # type: ignore
        value_external_max=np.max(external_distances),    # type: ignore
        value_external_sum=np.sum(external_distances),    # type: ignore

        one_nna=one_nna,
    )