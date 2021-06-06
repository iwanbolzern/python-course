from unittest.mock import patch

import numpy as np
from ranking import ranking_02_removed_basic_smells as ranking
import pytest



@pytest.mark.parametrize("X_train, X_test, ratio, expected_X_test", [
    (
            np.array([[0, 0], [0, 1], [1, 0]]),
            np.array([[3, 0], [3, 1], [4, 0], [4, 2]]),
            1,
            np.array([[3, 0], [3, 1], [4, 0], [4, 2]])
    ),
    (
            np.array([[0, 0], [0, 1], [1, 0]]),
            np.array([[3, 0], [3, 1], [4, 0], [4, 2]]),
            1.5,
            np.array([[3, 0], [3, 1], [4, 0], [4, 2], [0, 2], [1, 1]])
    ),
    (
            np.array([[0, 0], [0, 1], [1, 0]]),
            np.array([[3, 0], [3, 1], [4, 0], [4, 2]]),
            1.6,
            np.array([[3, 0], [3, 1], [4, 0], [4, 2], [0, 2], [1, 1]])
    ),
    (
            np.array([[0, 0], [0, 1], [1, 1]]),
            np.array([[3, 0], [3, 1], [4, 0], [4, 2]]),
            1.5,
            np.array([[3, 0], [3, 1], [4, 0], [4, 2], [0, 2], [4, 1]])
    ),
])
@patch("random.randint", side_effect=(0, 2, 1, 1, 4, 1))
def test_add_zeros(random_mock, X_train, X_test, ratio, expected_X_test):
    unique_portfolios = np.array([1, 2, 3, 4, 6])
    unique_investments = np.array([1, 2, 4])
    y_test = np.array([1] * len(X_test))
    num_added_examples = len(expected_X_test) - len(X_test)
    expected_y_test = np.array([1] * len(X_test) + [0] * num_added_examples)
    computed_X_test, computed_y_test = ranking.add_zeros(X_test, X_train, unique_investments, unique_portfolios, y_test,
                                                         ratio=ratio)

    np.testing.assert_array_equal(computed_y_test, expected_y_test)
    np.testing.assert_array_equal(computed_X_test, expected_X_test)
