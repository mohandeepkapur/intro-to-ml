from problem3.SimpleClassifier import SimpleClassifier
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np


def run_wine_inferences(cf: SimpleClassifier):
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features
    y = wine_quality.data.targets

    print(y)

    # metadata
    # print(wine_quality.metadata)

    # variable information
    # print(wine_quality.variables)

    # format pandas data
    features = []
    for i in range(X.shape[0]):
        features.append(X.iloc[i].values)
    labels = []
    for i in range(y.shape[0]):
        labels.append(y.iloc[i].values)
    features = np.array(features)
    labels = np.array(labels).T[0]
    loss_matrix = np.logical_not(np.identity(len(features))).astype(int)

    #print(f"# of unique classes {set(labels)}")

    conf_mat = cf.train_classifier(features, labels, loss_matrix)
    print(f"results: \n {conf_mat}")

    #print(cf.classify_features([X.iloc[6495].values]))

    # run through all columns
    # go through each row, sum up curr value * prior_prob associated with column
    pError = 0
    pps = cf.obs_prior_probs()
    for col in range(3, len(conf_mat[0])):
        for row in range(3, len(conf_mat)):
            if row is not col:
                pError += conf_mat[row][col] * pps[col]



    print("p of error: " + str(pError))
