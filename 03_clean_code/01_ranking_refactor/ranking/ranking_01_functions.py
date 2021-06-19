import pandas as pd
import numpy as np
import time
import random
from scipy import sparse
from sklearn.decomposition import NMF


def suggest_investments(H, W, investment_list, portfolio_list, portfolios, extended_positions, potential_investments,
                        potential_investors, min_swiss_rating=500, max_nonswiss_rating=800, prediction_threshold=0.01):
    result = {}
    for potential_investor in potential_investors:
        print("\nSuggesting investments for client", potential_investor)
        # check if swiss
        is_swiss = portfolios.loc[portfolios['PortfolioID'] == potential_investor]['Currency'].values[0] == 'CHF'
        # get investments
        current_investments = extended_positions.loc[extended_positions['PortfolioID'] == potential_investor]
        # check rating
        rating = compute_instrument_rating(current_investments, is_swiss)
        if is_swiss and rating < min_swiss_rating:
            print("Swiss client " + str(potential_investor) + " rating too low, no investment suggested")
        elif not is_swiss and rating > max_nonswiss_rating:
            print("Non-Swiss client " + str(potential_investor) + " rating too high, no investment suggested")
        else:
            y_pred = []
            for potential_investment in potential_investments:
                dot_product = 0
                # only do the prediction if investment is valid
                if potential_investment not in current_investments.values:
                    if len(np.where(investment_list == potential_investment)[0]) > 0:
                        # get user-item values
                        user = np.where(portfolio_list == potential_investor)[0].min()
                        item = np.where(investment_list == potential_investment)[0].min()
                        # compute prediction
                        dot_product = W[user, :].dot(H[:, item])
                        print(f"{user}-{item} score: {dot_product}")
                y_pred.append(dot_product)
            for i in range(len(y_pred)):
                if y_pred[i] >= prediction_threshold:
                    y_pred[i] = 1
                else:
                    y_pred[i] = 0
            result[str(potential_investor)] = [str(potential_investments[i]) for i in range(len(y_pred)) if
                                               y_pred[i] == 1]
    return result


def predict(H, W, X_test, threshold=0.01):
    y_pred = []
    for index in X_test:
        # compute dot product
        dot_product = W[index[0], :].dot(H[:, index[1]])
        y_pred.append(dot_product)
    for i in range(len(y_pred)):
        if y_pred[i] > threshold:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    return np.array(y_pred)


def compute_metrics(y_pred, y_test):
    # +
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    index = 0
    for prediction in y_pred:
        true_value = y_test[index]
        if prediction == 0 and true_value == 0:
            TN += 1
        if prediction == 0 and true_value == 1:
            FN += 1
        if prediction == 1 and true_value == 0:
            FP += 1
        if prediction == 1 and true_value == 1:
            TP += 1
        index += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    return precision, recall


def add_zeros(X_test, X_train, investment_list, portfolio_list, y_test, ratio=1):
    # Since our test set also contains only positive examples, we want to add some zero values: we randomly generate a
    # pair (user, item) and, if there isn't a position for it, we add it with rating zero
    # EXTRA TODO: this function is super slow! find a faster way
    l = int(len(X_test) * ratio)
    while len(X_test) < l:
        # random user-item pair
        u = random.randint(0, len(portfolio_list) - 1)
        i = random.randint(0, len(investment_list) - 1)
        already_in = False
        # check it was not in the training set
        for row in X_train:
            if row[0] == u and row[1] == i:
                already_in = True
        # check it was not in the test set
        for row in X_test:
            if row[0] == u and row[1] == i:
                already_in = True
        if not already_in:  # if it's not already in
            X_test = np.append(X_test, [[u, i]], axis=0)
            y_test = np.append(y_test, 0)
    return X_test, y_test


def create_user_item_df(positions):
    portfolio_list = positions["PortfolioID"].unique()
    investment_list = positions["InstrumentID"].unique()
    user_indices = []
    item_indices = []
    for index, row in positions.iterrows():
        user_indices += [portfolio_list.tolist().index(row["PortfolioID"])]
        item_indices += [investment_list.tolist().index(row["InstrumentID"])]
    ratings = [1] * len(user_indices)
    user_item_rating_df = pd.DataFrame({"User": user_indices,
                                        "Item": item_indices,
                                        "Rating": ratings})
    user_item_rating_df = user_item_rating_df.sample(frac=1).reset_index(drop=True)
    return investment_list, portfolio_list, user_item_rating_df


def preprocess(positions):
    # remove portfolios with too few (less than 3) transactions
    counts = positions['PortfolioID'].value_counts()
    filtered_indices = counts[counts >= 3].index.tolist()
    positions = positions[positions['PortfolioID'].isin(filtered_indices)]
    # remove investments owned by too few (less than 5) portfolios
    counts = positions['InstrumentID'].value_counts()
    filtered_indices = counts[counts >= 5].index.tolist()
    positions = positions[positions['InstrumentID'].isin(filtered_indices)]
    return positions


# ## Requirement from business
# we should not suggest new investments to
# - non Swiss clients with ranking too high
# - Swiss clients with ranking too low
RATING_NON_SWISS = 10
RATING_DEFAULT = 1
RATING_CHF = 5
CURRENCY_CHF = 'CHF'
RATING_GBP = 6
CURRENCY_GBP = 'GBP'
RATING_EUR = 7
CURRENCY_EUR = 'EUR'
RATING_USD = 8
CURRENCY_USD = 'USD'


def compute_instrument_rating(instruments, ch_cl=False):
    rating = 0

    # check if swiss client
    if ch_cl:
        # iterate over all the instruments
        for index, instrument in instruments.iterrows():

            # do not process insturments that should be ignored
            if not instrument["Ignore"]:

                # skip all the expired instruments
                if not instrument["Expired"]:

                    instr_curr = instrument["Currency"]

                    if instr_curr == CURRENCY_USD:
                        rating += RATING_USD
                    elif instr_curr == CURRENCY_EUR:
                        rating += RATING_EUR
                    elif instr_curr == CURRENCY_GBP:
                        rating += RATING_GBP
                    elif instr_curr == CURRENCY_CHF:
                        rating += RATING_CHF
                    else:
                        rating += RATING_DEFAULT
    else:
        rating = len(instruments) * RATING_NON_SWISS

    return rating


if __name__ == '__main__':
    DATA_FOLDER = "../../../Data/"
    positions = pd.read_csv(DATA_FOLDER + "positions.csv")
    portfolios = pd.read_csv(DATA_FOLDER + "portfolios.csv")
    instruments = pd.read_csv(DATA_FOLDER + "instruments.csv")
    # # preprocess data
    positions = preprocess(positions)
    # Create a user-item-rating dataframe, where users are portfolios and items are instruments.
    # The ratings will be all 1, because the data we have is only the instruments that have been bought.
    # This means we will only train on positive examples
    t1 = time.time()
    investment_list, portfolio_list, user_item_rating_df = create_user_item_df(positions)
    t2 = time.time()
    print("Building user-item frame took " + str(t2 - t1) + " seconds.")
    # # train/test split
    X = user_item_rating_df[["User", "Item"]].values
    y = user_item_rating_df["Rating"].values
    X_train, X_test = X[0:int(len(user_item_rating_df) * 0.8)], X[int(len(user_item_rating_df) * 0.8):]
    y_train, y_test = y[0:int(len(user_item_rating_df) * 0.8)], y[int(len(user_item_rating_df) * 0.8):]
    # # Train model
    X_sparse = sparse.csr_matrix((y_train, (X_train[:, 0], X_train[:, 1])),
                                 shape=(len(portfolio_list), len(investment_list)))
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
    X_test, y_test = add_zeros(X_test, X_train, investment_list, portfolio_list, y_test)
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
    result = suggest_investments(H, W, investment_list, portfolio_list, portfolios, extended_positions,
                                 potential_investments, potential_investors)
    for client in result:
        print("Investments suggested for client " + client + ":")
        print(", ".join(result[client]))
