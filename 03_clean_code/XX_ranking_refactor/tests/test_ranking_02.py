import pandas as pd
import numpy as np
from ranking import ranking_02_removed_basic_smells as ranking


def test_preprocess():
    positions = pd.DataFrame({
        'PortfolioID': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6],
        'InstrumentID': [1, 2, 3, 4, 1, 4, 5, 9, 1, 2, 4, 1, 4, 9, 1, 1, 4, 8],
        'Amount': [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 6, 6, 6]
    })
    expected = pd.DataFrame({
        'PortfolioID': [1, 1, 2, 2, 3, 3, 4, 4, 6, 6],
        'InstrumentID': [1, 4, 1, 4, 1, 4, 1, 4, 1, 4],
        'Amount': [1, 1, 2, 2, 3, 3, 4, 4, 6, 6]
    })
    computed = ranking.preprocess(positions).reset_index(drop=True)
    pd.testing.assert_frame_equal(computed, expected)


def test_create_user_item_df():
    positions = pd.DataFrame({
        'PortfolioID': [1, 1, 2, 2, 3, 3, 4, 4, 6, 6],
        'InstrumentID': [1, 4, 1, 4, 1, 4, 1, 4, 1, 4],
        'Amount': [1, 1, 2, 2, 3, 3, 4, 4, 6, 6]
    })
    expected_portfolio_list = [1, 2, 3, 4, 6]
    expected_investment_list = [1, 4]
    expected_user_item_rating_df = pd.DataFrame({
        'User': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        'Item': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        'Rating': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    })
    investment_list, portfolio_list, user_item_rating_df = ranking.create_user_item_df(positions)
    user_item_rating_df = user_item_rating_df.sort_values(by=['User', 'Item']).reset_index(drop=True)

    np.testing.assert_array_equal(investment_list, expected_investment_list)
    np.testing.assert_array_equal(portfolio_list, expected_portfolio_list)
    pd.testing.assert_frame_equal(user_item_rating_df.reset_index(drop=True), expected_user_item_rating_df)


def test_add_zeros():
    X_train = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 2],
        [2, 1],
        [2, 2]
    ])
    X_test = np.array([
        [3, 0],
        [3, 1],
        [4, 0],
        [4, 2],
    ])
    y_test = np.array([1, 1, 1, 1])
    portfolio_list = np.array([1, 2, 3, 4, 6])
    investment_list = np.array([1, 2, 4])
    computed_X_test, computed_y_test = ranking.add_zeros(X_test, X_train, investment_list, portfolio_list, y_test,
                                                         ratio=1.5)
    expected_y_test = np.array([1, 1, 1, 1, 0, 0])

    np.testing.assert_array_equal(computed_y_test, expected_y_test)
    assert computed_X_test.shape == (6, 2)
    for index in [4, 5]:
        new_item = computed_X_test[index]
        assert not any(np.equal(X_train, new_item).all(1))
        assert not any(np.equal(X_test, new_item).all(1))


def test_predict():
    W = np.array([
        [1, 0, 2],
        [1, 0, 0],
        [0, 0, 2],
        [1, 2, 1],
        [0, 1, 0]
    ])
    H = np.array([
        [3, 0, 0, 1, 1, 1, 1],
        [0, 2, 0, 1, 3, 0, 2],
        [1, 0, 4, 1, 0, 2, 1],
    ])
    X_test = np.array([
        [2, 2],
        [3, 6],
        [4, 6],
        [1, 3]
    ])
    y_expected = np.array([1, 1, 0, 0])
    y_computed = ranking.predict(H, W, X_test, threshold=3)
    np.testing.assert_array_equal(y_computed, y_expected)


def test_compute_metrics():
    y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0])
    y_test = np.array([1, 1, 1, 0, 0, 0, 0, 0, 1])
    precision, recall = ranking.compute_metrics(y_pred, y_test)
    assert precision == 0.6
    assert recall == 0.75


def test_suggest_investments():
    W = np.array([
        [2, 0],
        [1, 0],
        [1, 0],
        [0, 2],
        [0, 1]
    ])
    H = np.array([
        [2, 2, 1, 3, 0, 0, 0],
        [0, 0, 0, 0, 3, 1, 2]
    ])
    portfolio_list = np.array([1, 10, 11, 42, 69])
    investment_list = np.array([2, 4, 6, 8, 10, 12, 14])
    portfolios = pd.DataFrame({
        'PortfolioID': [1, 10, 11, 42, 69],
        'Currency': ['CHF', 'CHF', 'CHF', 'GBP', 'CHF']
    })
    extended_positions = pd.DataFrame({
        'PortfolioID': [1, 1, 1, 10, 10, 11, 11, 42, 42, 42, 69, 69],
        'InstrumentID': [2, 4, 6, 2, 8, 4, 8, 10, 12, 14, 10, 12],
        'Currency': ['EUR'] * 12,
        'Ignore': [False] * 12,
        'Expired': [False] * 12
    })
    potential_investments = investment_list
    potential_investors = [1, 42, 69]
    min_swiss_rating = 10
    max_nonswiss_rating = 30
    prediction_threshold = 2

    expected_result = {'1': ['8'], '42': [], '69': ['14']}
    computed_result = ranking.suggest_investments(H, W, investment_list, portfolio_list, portfolios, extended_positions,
                                                  potential_investments, potential_investors, min_swiss_rating,
                                                  max_nonswiss_rating, prediction_threshold)
    assert len(expected_result) == len(computed_result)
    for investor in computed_result:
        assert expected_result[investor] == computed_result[investor]


def test_compute_instrument_rating():
    instruments = pd.DataFrame({
        'InstrumentID': [1, 2, 3, 4, 5],
        'Currency': ['USD', 'CHF', 'EUR', 'EUR', 'CHF'],
        'Ignore': [False, False, False, True, False],
        'Expired': [False, False, False, False, True]
    })
    assert ranking.compute_instrument_rating_for_non_swiss_clients(instruments) == 50
    assert ranking.compute_instrument_rating_for_swiss_clients(instruments) == 20
