import pandas as pd
import numpy as np
import time
import random
from scipy import sparse
from sklearn.decomposition import NMF
from sklearn.metrics import confusion_matrix
from enum import Enum


class CurrencyRating(Enum):
    CHF = 5
    GBP = 6
    EUR = 7
    USD = 8
    NON_SWISS = 10
    DEFAULT = 1


def suggest_investments(H: np.ndarray, W: np.ndarray, unique_investments: np.ndarray, unique_portfolios: np.ndarray,
                        portfolios: pd.DataFrame, extended_positions: pd.DataFrame, potential_investments,
                        potential_investors, min_swiss_rating=500, max_nonswiss_rating=800, prediction_threshold=0.01):
    result = {}
    for potential_investor in potential_investors:
        current_investments = extended_positions.loc[extended_positions['PortfolioID'] == potential_investor]
        rating_allowed = check_rating(potential_investor, portfolios, current_investments, max_nonswiss_rating,
                                      min_swiss_rating)
        if not rating_allowed:
            continue
        score = np.array([])
        user = find_index(potential_investor, unique_portfolios)
        for potential_investment in potential_investments:
            # only do the prediction if investment is valid
            investment_valid = check_valid_investment(current_investments, potential_investment, unique_investments)
            if not investment_valid:
                score = np.append(score, 0)
                continue
            item = find_index(potential_investment, unique_investments)
            # compute prediction
            dot_product = W[user, :].dot(H[:, item])
            score = np.append(score, dot_product)

        y_pred = np.where(score >= prediction_threshold, 1, 0)
        result[str(potential_investor)] = [str(potential_investments[i]) for i in range(len(y_pred)) if y_pred[i] == 1]
    return result


def find_index(value, array):
    return np.where(array == value)[0].min()


def check_valid_investment(current_investments, potential_investment, unique_investments):
    if potential_investment in current_investments.values:
        return False
    if len(np.where(unique_investments == potential_investment)[0]) == 0:
        return False
    return True


def check_rating(potential_investor, portfolios, current_investments, max_nonswiss_rating,
                 min_swiss_rating) -> bool:
    is_swiss = portfolios.loc[portfolios['PortfolioID'] == potential_investor]['Currency'].values[0] == 'CHF'
    if is_swiss:
        rating = compute_instrument_rating_for_swiss_clients(current_investments)
    else:
        rating = compute_instrument_rating_for_non_swiss_clients(current_investments)
    if is_swiss and rating < min_swiss_rating:
        print("Swiss client " + str(potential_investor) + " rating too low, no investment suggested")
        return False
    elif not is_swiss and rating > max_nonswiss_rating:
        print("Non-Swiss client " + str(potential_investor) + " rating too high, no investment suggested")
        return False
    return True


def compute_instrument_rating_for_non_swiss_clients(instruments: pd.DataFrame):
    rating = len(instruments) * CurrencyRating.NON_SWISS.value
    return rating


def compute_instrument_rating_for_swiss_clients(instruments: pd.DataFrame):
    rating = 0
    valid_instruments = instruments.loc[(~instruments["Ignore"]) & (~instruments["Expired"])]
    for currency in valid_instruments["Currency"]:
        try:
            rating += CurrencyRating[currency].value
        except KeyError:
            rating += CurrencyRating.DEFAULT.value
    return rating


def predict(H: np.ndarray, W: np.ndarray, X_test: np.ndarray, threshold=0.01):
    dot_product = [W[user, :].dot(H[:, item]) for user, item in X_test]
    y_pred = np.array(dot_product)
    return np.where(y_pred >= threshold, 1, 0)


def compute_metrics(y_pred: np.ndarray, y_test: np.ndarray):
    confusion = confusion_matrix(y_test, y_pred)
    TN, FP, FN, TP = np.ravel(confusion)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


def add_zeros(X_test: np.ndarray, X_train: np.ndarray, unique_investments: np.ndarray, unique_portfolios: np.ndarray,
              y_test: np.ndarray, ratio: float = 1):
    # Since our test set also contains only positive examples, we want to add some zero values: we randomly generate a
    # pair (user, item) and, if there isn't a position for it, we add it with rating zero
    new_length = int(len(X_test) * ratio)
    X = np.concatenate((X_train, X_test), axis=0)

    while len(X_test) < new_length:
        random_user_index = random.randint(0, len(unique_portfolios) - 1)
        random_item_index = random.randint(0, len(unique_investments) - 1)
        entry = np.array([random_user_index, random_item_index])
        if not any(np.equal(X, entry).all(1)):
            X_test = np.append(X_test, [entry], axis=0)
            y_test = np.append(y_test, 0)
    return X_test, y_test


def create_user_item_df(positions: pd.DataFrame):
    unique_portfolios, user_indices = compute_unique_values_and_indices(positions["PortfolioID"])
    unique_investments, item_indices = compute_unique_values_and_indices(positions["InstrumentID"])
    ratings = [1] * len(user_indices)
    user_item_rating_df = pd.DataFrame({"User": user_indices,
                                        "Item": item_indices,
                                        "Rating": ratings})
    user_item_rating_df = user_item_rating_df.sample(frac=1).reset_index(drop=True)
    return unique_investments, unique_portfolios, user_item_rating_df


def compute_unique_values_and_indices(column):
    unique_values = column.unique()
    indices_mapping = {portfolio: index for index, portfolio in enumerate(unique_values)}
    indices = column.map(indices_mapping)
    return unique_values, indices


def create_user_item_df_old(positions: pd.DataFrame):
    unique_portfolios = positions["PortfolioID"].unique().tolist()
    unique_investments = positions["InstrumentID"].unique().tolist()
    user_indices = []
    item_indices = []
    for index, row in positions.iterrows():
        user_indices += [unique_portfolios.index(row["PortfolioID"])]
        item_indices += [unique_investments.index(row["InstrumentID"])]
    ratings = [1] * len(user_indices)
    user_item_rating_df = pd.DataFrame({"User": user_indices,
                                        "Item": item_indices,
                                        "Rating": ratings})
    user_item_rating_df = user_item_rating_df.sample(frac=1).reset_index(drop=True)
    return unique_investments, unique_portfolios, user_item_rating_df


def preprocess(positions: pd.DataFrame):
    min_transactions = 3
    min_portfolios = 5
    positions = _filter_col_min_value(positions, 'PortfolioID', min_transactions)
    positions = _filter_col_min_value(positions, 'InstrumentID', min_portfolios)
    return positions


def _filter_col_min_value(df: pd.DataFrame, column: str, min_count: int):
    counts = df[column].value_counts()
    filtered_indices = counts[counts >= min_count].index.tolist()
    return df[df[column].isin(filtered_indices)]


if __name__ == '__main__':
    DATA_FOLDER = "../../../Data/"
    positions = pd.read_csv(DATA_FOLDER + "positions.csv")
    portfolios = pd.read_csv(DATA_FOLDER + "portfolios.csv")
    instruments = pd.read_csv(DATA_FOLDER + "instruments.csv")
    positions = preprocess(positions)
    print(f"{len(positions)} positions after preprocessing")

    # Create a user-item-rating dataframe, where users are portfolios and items are instruments.
    # The ratings will be all 1, because the data we have is only the instruments that have been bought.
    # This means we will only train on positive examples
    t1 = time.time()
    unique_investments, unique_portfolios, user_item_rating_df = create_user_item_df(positions)
    t2 = time.time()
    print("Building user-item frame took " + str(t2 - t1) + " seconds.")

    # # train/test split
    X = user_item_rating_df[["User", "Item"]].values
    y = user_item_rating_df["Rating"].values
    X_train, X_test = X[0:int(len(user_item_rating_df) * 0.8)], X[int(len(user_item_rating_df) * 0.8):]
    y_train, y_test = y[0:int(len(user_item_rating_df) * 0.8)], y[int(len(user_item_rating_df) * 0.8):]

    # # Train model
    X_sparse = sparse.csr_matrix((y_train, (X_train[:, 0], X_train[:, 1])),
                                 shape=(len(unique_portfolios), len(unique_investments)))
    model = NMF(
        n_components=3,
        init='random',
        solver='cd',
        beta_loss='frobenius',
        max_iter=200,
        tol=0.0001,
        alpha=0,
        l1_ratio=0,
        random_state=0,
        verbose=0,
        shuffle=False)
    W = model.fit_transform(X_sparse)
    H = model.components_

    # # Test model
    t1 = time.time()
    X_test, y_test = add_zeros(X_test, X_train, unique_investments, unique_portfolios, y_test)
    t2 = time.time()
    print("Adding zeros took " + str(t2 - t1) + " seconds.")
    y_pred = predict(H, W, X_test)

    # # Visualize metrics
    precision, recall = compute_metrics(y_pred, y_test)
    print("Precision:", precision)
    print("Recall:", recall)

    # # Prediction
    potential_investors = [42, 69, 420]
    potential_investments = range(100, 150)
    extended_positions = pd.merge(positions, instruments, on='InstrumentID')
    result = suggest_investments(H, W, unique_investments, unique_portfolios, portfolios, extended_positions,
                                 potential_investments, potential_investors)
    for client in result:
        print("Investments suggested for client " + client + ":")
        print(", ".join(result[client]))
