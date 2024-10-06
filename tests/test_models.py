"""Tests for statistics functions within the Model layer."""

import numpy as np
import numpy.testing as npt
import pytest


# def test_daily_mean_zeros():
#     """Test that mean function works for an array of zeros."""
#     from inflammation.models import daily_mean

#     test_input = np.array([[0, 0],
#                            [0, 0],
#                            [0, 0]])
#     test_result = np.array([0, 0])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_mean(test_input), test_result)


# def test_daily_mean_integers():
#     """Test that mean function works for an array of positive integers."""
#     from inflammation.models import daily_mean

#     test_input = np.array([[1, 2],
#                            [3, 4],
#                            [5, 6]])
#     test_result = np.array([3, 4])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_mean(test_input), test_result)

@pytest.mark.parametrize(
    "test, expected",
    [
        ([ [0, 0], [0, 0], [0, 0] ], [0, 0]),
        ([ [1, 2], [3, 4], [5, 6] ], [3, 4]),
    ])
def test_daily_mean(test, expected):
    """Test mean function works for array of zeroes and positive integers."""
    from inflammation.models import daily_mean
    np.testing.assert_array_equal(daily_mean(np.array(test)), np.array(expected))

@pytest.mark.parametrize(
    "test, expected", 
    [
        ([ [1, 2], [3, 10], [5, 7] ], [5, 10]),
        ([ [-1, -1], [-3, -5], [-1, -1] ], [-1, -1])
    ]
)
def test_daily_max(test, expected):
    """Test that max function works for an array of distict positive integers, and  negative integers, not all distinct."""
    from inflammation.models import daily_max
    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


# def test_daily_max_distinct_positive_integer_data():
#     """Test that max function works for an array of distict positive integers."""
#     from inflammation.models import daily_max

#     test_input = np.array([[1, 2],
#                            [3, 10],
#                            [5, 7]])
#     test_result = np.array([5, 10])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_max(test_input), test_result)

# def test_daily_max_nondistinct_negative_integer_data():
#     """Test that max function works for an array of negative integers, not all distinct."""
#     from inflammation.models import daily_max

#     test_input = np.array([[-1, -1],
#                            [-3, -5],
#                            [-1, -1]])
#     test_result = np.array([-1, -1])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_max(test_input), test_result)

@pytest.mark.parametrize(
    "test, expected, expected_error",
    [
        ([ [1, 2], [3, 10], [5, 0] ], [1, 0], None),
        ([ [-1, -1], [-3, -5], [-1, -1] ], [-3, -5], None),
        ([['Hello', 'there'], ['General', 'Kenobi']], float('nan') ,TypeError)
    ])
def test_daily_min(test, expected, expected_error):
    """Test that min function works for an array of distict positive integers."""
    from inflammation.models import daily_min
    if expected_error is not None:
        with pytest.raises(expected_error):
            np.testing.assert_almost_equal(daily_min(np.array(test)), np.array(expected), decimal=2)
    else:
        np.testing.assert_almost_equal(daily_min(np.array(test)), np.array(expected), decimal=2)

# def test_daily_min_distinct_positive_integer_data():
#     """Test that min function works for an array of distict positive integers."""
#     from inflammation.models import daily_min

#     test_input = np.array([[1, 2],
#                            [3, 10],
#                            [5, 0]])
#     test_result = np.array([1, 0])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_min(test_input), test_result)

# def test_daily_min_nondistinct_negative_integer_data():
#     """Test that min function works for an array of negative integers, not all distinct."""
#     from inflammation.models import daily_min

#     test_input = np.array([[-1, -1],
#                            [-3, -5],
#                            [-1, -1]])
#     test_result = np.array([-3, -5])

#     # Need to use Numpy testing functions to compare arrays
#     npt.assert_array_equal(daily_min(test_input), test_result)

# def test_daily_min_string():
#     """Test for TypeError when passing strings"""
#     from inflammation.models import daily_min

#     with pytest.raises(TypeError):
#         error_expected = daily_min([['Hello', 'there'], ['General', 'Kenobi']])

@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], None,),
        ([[1, 1, 1], [1, 1, 1], [1, 1, 1]], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], None, ),
         (
            [[float('nan'), 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[1, 2, 3], [4, 5, float('nan')], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.8, 1, 0], [0.78, 0.89, 1]],
            None,
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]], None,),
        ([[1,1,0.000000001], [0.00000008, 0.000008, 0.0882882], [0.00000000000009, 0.001, 0.002]], [[1,1,0], [0,0,1], [0,0.5, 1]], None, ), # Anda test, to see if for small values, but not zero, result still holds. 
    ])
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers.
       Assumption that test accuracy of two decimal places is sufficient."""
    from inflammation.models import patient_normalise
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            np.testing.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)
    else:
        np.testing.assert_almost_equal(patient_normalise(np.array(test)), np.array(expected), decimal=2)