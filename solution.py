import pandas as pd
import numpy as np
from os.path import join

# 1. read data

ltable = pd.read_csv(join('data', "ltable.csv"))
rtable = pd.read_csv(join('data', "rtable.csv"))
train = pd.read_csv(join('data', "train.csv"))


# 2. blocking
def pairs2LR(ltable, rtable, candset):
    ltable.index = ltable.id
    rtable.index = rtable.id
    pairs = np.array(candset)
    tpls_l = ltable.loc[pairs[:, 0], :]
    tpls_r = rtable.loc[pairs[:, 1], :]
    tpls_l.columns = [col + "_l" for col in tpls_l.columns]
    tpls_r.columns = [col + "_r" for col in tpls_r.columns]
    tpls_l.reset_index(inplace=True, drop=True)
    tpls_r.reset_index(inplace=True, drop=True)
    LR = pd.concat([tpls_l, tpls_r], axis=1)
    return LR


def block_by_brand(ltable, rtable):
    # ensure brand is str
    ltable['category'] = ltable['category'].astype(str)
    rtable['category'] = rtable['category'].astype(str)
    #'category'
    # get all brands
    brands_l = set(ltable["category"].values)
    brands_r = set(rtable["category"].values)
    brands = brands_l.union(brands_r)

    # map each brand to left ids and right ids
    brand2ids_l = {b.lower(): [] for b in brands}
    brand2ids_r = {b.lower(): [] for b in brands}
    for i, x in ltable.iterrows():
        brand2ids_l[x["category"].lower()].append(x["id"])
    for i, x in rtable.iterrows():
        brand2ids_r[x["category"].lower()].append(x["id"])

    # put id pairs that share the same brand in candidate set
    candset = []
    for brd in brands:
        l_ids = brand2ids_l[brd]
        r_ids = brand2ids_r[brd]
        for i in range(len(l_ids)):
            for j in range(len(r_ids)):
                candset.append([l_ids[i], r_ids[j]])
    return candset

# blocking to reduce the number of pairs to be compared
candset = block_by_brand(ltable, rtable)
print("number of pairs originally", ltable.shape[0] * rtable.shape[0])
print("number of pairs after blocking",len(candset))
candset_df = pairs2LR(ltable, rtable, candset)



# 3. Feature engineering
import Levenshtein as lev

# def jaccard_similarity(row, attr):
#     x = set(row[attr + "_l"].lower().split())
#     y = set(row[attr + "_r"].lower().split())
#     return len(x.intersection(y)) / max(len(x), len(y))

def cosine_similarity(row, attr):
    x = set(row[attr + "_l"].lower().split())
    y = set(row[attr + "_r"].lower().split())
    return len(x.intersection(y)) / np.sqrt(len(x) * len(y))

def levenshtein_distance(row, attr):
    x = row[attr + "_l"].lower()
    y = row[attr + "_r"].lower()
    return lev.distance(x, y)

def feature_engineering(LR):
    LR = LR.astype(str)
    attrs = ["title", "category", "brand", "modelno", "price"]
    features = []
    for attr in attrs:
        j_sim = LR.apply(cosine_similarity, attr=attr, axis=1)
        l_dist = LR.apply(levenshtein_distance, attr=attr, axis=1)
        features.append(j_sim)
        features.append(l_dist)
    features = np.array(features).T
    return features
candset_features = feature_engineering(candset_df)

# also perform feature engineering to the training set
training_pairs = list(map(tuple, train[["ltable_id", "rtable_id"]].values))
training_df = pairs2LR(ltable, rtable, training_pairs)
training_features = feature_engineering(training_df)
training_label = train.label.values

# 4. Model training and prediction
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, truncnorm, randint

# model_params = {
#     # randomly sample numbers from 4 to 204 estimators
#     'n_estimators': randint(4,200),
#     # normally distributed max_features, with mean .25 stddev 0.1, bounded between 0 and 1
#     'max_features': truncnorm(a=0, b=1, loc=0.25, scale=0.1),
#     # uniform distribution from 0.01 to 0.2 (0.01 + 0.199)
#     'min_samples_split': uniform(0.01, 0.199)
# }

rf = RandomForestClassifier(class_weight="balanced", random_state=0)

# clf = RandomizedSearchCV(rf, model_params, n_iter=100, cv=5, random_state=1)

# model = clf.fit(training_features, training_label)

# from pprint import pprint
# pprint(model.best_estimator_.get_params())

rf.fit(training_features, training_label)
y_pred = rf.predict(candset_features)
# y_pred = model.predict(training_features)

# 5. output

matching_pairs = candset_df.loc[y_pred == 1, ["id_l", "id_r"]]
matching_pairs = list(map(tuple, matching_pairs.values))

matching_pairs_in_training = training_df.loc[training_label == 1, ["id_l", "id_r"]]
matching_pairs_in_training = set(list(map(tuple, matching_pairs_in_training.values)))

pred_pairs = [pair for pair in matching_pairs if
              pair not in matching_pairs_in_training]  # remove the matching pairs already in training
pred_pairs = np.array(pred_pairs)
pred_df = pd.DataFrame(pred_pairs, columns=["ltable_id", "rtable_id"])
pred_df.to_csv("output.csv", index=False)
