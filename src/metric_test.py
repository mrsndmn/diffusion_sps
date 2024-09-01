
import numpy as np

from metric import nearest_distance, metric_nearest_distance

def test_nearest_distance():

    point_value = np.array([0.5, 0.5])
    target_frame_value = np.array([[0.4, 0.4], [0.7, 0.7]])

    metric_value = nearest_distance(point_value, target_frame_value)

    assert np.allclose(metric_value, 0.1414, atol=1e-4)


def test_metric_nearest_distance():

    generated_frame = np.array([[0.5, 0.5], [0.9, 0.9]])
    target_frame_value = np.array([[0.4, 0.4], [0.7, 0.7]])

    metric_value = metric_nearest_distance(generated_frame, target_frame_value)

    assert np.allclose(metric_value.value_max, 0.2828, atol=1e-4)
    assert np.allclose(metric_value.value_mean, 0.2121, atol=1e-4)
    assert np.allclose(metric_value.value_sum, 0.4242, atol=1e-4)