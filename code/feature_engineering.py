import numpy as np
import generaltools as gt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LabelKFold

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn.metrics import confusion_matrix
import matplotlib.cm as cmap

import pickle

def transform_chi(labels):
    labels[labels == "chi1"] = "chi"
    labels[labels == "chi2"] = "chi"
    labels[labels == "chi3"] = "chi"
    labels[labels == "chi4"] = "chi"
    return labels


def load_features(datadir, tseg, log_features=None, ranking=None):

    features_train_full = np.loadtxt(datadir+"grs1915_%is_features_train.txt"%tseg)
    features_test_full = np.loadtxt(datadir+"grs1915_%is_features_test.txt"%tseg)
    features_val_full = np.loadtxt(datadir+"grs1915_%is_features_val.txt"%tseg)

    labels_test_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_test.txt"%tseg)[0])
    labels_train_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_train.txt"%tseg)[0])
    labels_val_full = np.array(gt.conversion(datadir+"grs1915_%is_labels_val.txt"%tseg)[0])

    tstart_train_full = np.loadtxt(datadir+"grs1915_%is_tstart_train.txt"%tseg)
    tstart_test_full = np.loadtxt(datadir+"grs1915_%is_tstart_test.txt"%tseg)
    tstart_val_full = np.loadtxt(datadir+"grs1915_%is_tstart_val.txt"%tseg)

    nseg_train_full = np.loadtxt(datadir+"grs1915_%is_nseg_train.txt"%tseg)
    nseg_test_full = np.loadtxt(datadir+"grs1915_%is_nseg_test.txt"%tseg)
    nseg_val_full = np.loadtxt(datadir+"grs1915_%is_nseg_val.txt"%tseg)

    features_all_full = np.concatenate((features_train_full, features_val_full,
                                        features_test_full))


    lc_all_full = gt.getpickle(datadir+"grs1915_%is_lc_all.dat"%tseg)
    hr_all_full = gt.getpickle(datadir+"grs1915_%is_hr_all.dat"%tseg)

    lc_train_full = lc_all_full["train"]
    lc_test_full = lc_all_full["test"]
    lc_val_full = lc_all_full["val"]

    hr_train_full = hr_all_full["train"]
    hr_test_full = hr_all_full["test"]
    hr_val_full = hr_all_full["val"]

    # NOTE: This line only works unless I change the features or the
    # order of the features! BEWARE!
    # It takes care of a single segment that seems to be an outlier in
    # terms of the hardness ratios
    # It's strange enough to be fairly certain it's instrumental
    delete_ind_train = np.where(features_train_full[:,17] >= 20)[0]
    delete_ind_val = np.where(features_val_full[:,17] >= 20)[0]
    delete_ind_test = np.where(features_test_full[:,17] >= 20)[0]

    if len(delete_ind_train) > 0:
        print("Found outlier in training set.")
        features_train_full = np.delete(features_train_full,
                                        delete_ind_train, axis=0)
        labels_train_full = np.delete(labels_train_full,
                                        delete_ind_train, axis=0)
        lc_train_full = np.delete(lc_train_full,
                                        delete_ind_train, axis=0)
        hr_train_full = np.delete(hr_train_full,
                                        delete_ind_train, axis=0)
        tstart_train_full = np.delete(tstart_train_full,
                                        delete_ind_train, axis=0)
        nseg_train_full = np.delete(nseg_train_full,
                                        delete_ind_train, axis=0)

    if len(delete_ind_val) > 0:
        print("Found outlier in validation set")
        features_val_full = np.delete(features_val_full,
                                        delete_ind_val, axis=0)
        labels_val_full = np.delete(labels_val_full,
                                        delete_ind_val, axis=0)
        lc_val_full = np.delete(lc_val_full,
                                        delete_ind_val, axis=0)
        hr_val_full = np.delete(hr_val_full,
                                        delete_ind_val, axis=0)
        tstart_val_full = np.delete(tstart_val_full,
                                        delete_ind_val, axis=0)
        nseg_val_full = np.delete(nseg_val_full,
                                        delete_ind_val, axis=0)

    if len(delete_ind_test) > 0:
        print("Found outlier in test set.")
        features_test_full = np.delete(features_test_full,
                                        delete_ind_test, axis=0)
        labels_test_full = np.delete(labels_test_full,
                                        delete_ind_test, axis=0)
        lc_test_full = np.delete(lc_test_full,
                                        delete_ind_test, axis=0)
        hr_test_full = np.delete(hr_test_full,
                                        delete_ind_test, axis=0)
        tstart_test_full = np.delete(tstart_test_full,
                                        delete_ind_test, axis=0)
        nseg_test_full = np.delete(nseg_test_full,
                                        delete_ind_test, axis=0)


    features_all_full = np.concatenate((features_train_full, features_val_full,
                                        features_test_full))

    print("after removal: " + str(np.where(features_all_full[:,17] >= 20)[0]))

    labels_train_full = transform_chi(labels_train_full)
    labels_test_full = transform_chi(labels_test_full)
    labels_val_full = transform_chi(labels_val_full)

    if ranking is None:
        features = {"train": features_train_full,
                    "test": features_test_full,
                    "val": features_val_full}
    else:
        features_train_new = np.array([features_train_full[:,i] for i in ranking]).T
        features_test_new = np.array([features_test_full[:,i] for i in ranking]).T
        features_val_new = np.array([features_val_full[:,i] for i in ranking]).T

        features = {"train": features_train_new,
                    "test": features_test_new,
                    "val": features_val_new}

    labels = {"train": labels_train_full,
              "test": labels_test_full,
              "val": labels_val_full}

    lc = {"train": lc_train_full,
          "test": lc_test_full,
          "val": lc_val_full}

    hr = {"train": hr_train_full,
          "test": hr_test_full,
          "val": hr_val_full}

    tstart = {"train": tstart_train_full,
              "val": tstart_val_full,
              "test": tstart_test_full}

    nseg = {"train": nseg_train_full,
            "val": nseg_val_full,
            "test": nseg_test_full}

    return features, labels, lc, hr, tstart, nseg


def labelled_data(features, labels, lc, hr, tstart, nseg):

    labels_train_full = labels["train"]
    labels_test_full = labels["test"]
    labels_val_full = labels["val"]

    train_ind =  np.where(labels_train_full != "None")[0]
    test_ind =  np.where(labels_test_full != "None")[0]
    val_ind =  np.where(labels_val_full != "None")[0]

    labels_train = labels_train_full[train_ind]
    labels_test = labels_test_full[test_ind]
    labels_val = labels_val_full[val_ind]

    features_train = features["train"][train_ind]
    features_test = features["test"][test_ind]
    features_val = features["val"][val_ind]

    tstart_train = tstart["train"][train_ind]
    tstart_test = tstart["test"][test_ind]
    tstart_val = tstart["val"][val_ind]

    nseg_train = nseg["train"][train_ind]
    nseg_test = nseg["test"][test_ind]
    nseg_val = nseg["val"][val_ind]

    lc_train = np.array([lc["train"][i] for i in train_ind])
    lc_test = np.array([lc["test"][i] for i in test_ind])
    lc_val = np.array([lc["val"][i] for i in val_ind])

    hr_train = np.array([hr["train"][i] for i in train_ind])
    hr_test = np.array([hr["test"][i] for i in test_ind])
    hr_val = np.array([hr["val"][i] for i in val_ind])

    features_lb = {"train": features_train,
                "test": features_test,
                "val": features_val}

    labels_lb = {"train": labels_train,
              "test": labels_test,


              "val": labels_val}

    lc_lb = {"train": lc_train,
          "test": lc_test,
          "val": lc_val}

    hr_lb = {"train": hr_train,
          "test": hr_test,
          "val": hr_val}

    tstart_lb = {"train": tstart_train,
          "test": tstart_test,
          "val": tstart_val}

    nseg_lb = {"train": nseg_train,
          "test": nseg_test,
          "val": nseg_val}


    return features_lb, labels_lb, lc_lb, hr_lb, tstart_lb, nseg_lb


def scale_features(features, features_lb=None):

    features_all = np.vstack([features["train"],
                              features["val"],
                              features["test"]])

    scaler = StandardScaler().fit(features_all)

    fscaled = {"train": scaler.transform(features["train"]),
               "test": scaler.transform(features["test"]),
               "val": scaler.transform(features["val"])}

    if features_lb is not None:
        fscaled_lb = {"train": scaler.transform(features_lb["train"]),
                      "test": scaler.transform(features_lb["test"]),
                      "val": scaler.transform(features_lb["val"])}

        return fscaled, fscaled_lb

    else:
        return fscaled


def choose_label(label):
    chaotic = ["beta", "lambda", "kappa", "mu"]
    deterministic = ["theta", "rho", "alpha", "nu", "delta"]
    stochastic = ["phi", "gamma", "chi"]

    if label in chaotic:
        return "chaotic"
    elif label in deterministic:
        return "deterministic"
    elif label in stochastic:
        return "stochastic"
    else:
        return label

def convert_labels_to_physical(labels):
    labels_train = labels["train"]
    labels_val = labels["val"]
    labels_test = labels["test"]

    labels_train_phys, labels_val_phys, labels_test_phys = [], [], []
    for l in labels_train:
        labels_train_phys.append(choose_label(l))

    for l in labels_test:
            labels_test_phys.append(choose_label(l))

    for l in labels_val:
            labels_val_phys.append(choose_label(l))

    labels_phys = {"train":labels_train_phys,
              "test": labels_test_phys,
              "val": labels_val_phys}

    return labels_phys




def greedy_search(datadir, seg_length_supervised=1024., n_folds=5):

    features, labels, lc, hr, tstart, nseg = load_features(datadir,
                                                     seg_length_supervised)

    features_lb, labels_lb, lc_lb, hr_lb, tstart_lb, nseg_lb = labelled_data(features,
                                                                    labels,
                                                                    lc, hr, tstart, nseg)

    labels_all = np.hstack([labels_lb["train"], labels_lb["val"], labels_lb["test"]])

    fscaled, fscaled_lb = scale_features(features, features_lb)
    features_train = features_lb["train"]
    features_val = features_lb["val"]
    features_test = features_lb["test"]
    labels_train = labels_lb["train"]
    labels_val =  labels_lb["val"]
    labels_test = labels_lb["test"]

    score_all = [] 
    feature_ranking = []
    nfeatures = range(features_train.shape[1])
    features_new_train = []
    features_new_val = []
    features_new_test = []
    best_params_all = []

    for i in range(features_train.shape[1]):
        print("I am on the %ith loop"%i)
        score = []
        best_params = []
        ## first feature
        for j in nfeatures:
            if j in feature_ranking:
                continue
            if len(features_new_train) == 0:
                ft = np.atleast_2d(features_train[:,j]).T
                fv = np.atleast_2d(features_val[:,j]).T
                fte = np.atleast_2d(features_test[:,j]).T
            else:
                ft = np.vstack([features_new_train.T, features_train[:,j]]).T
                fv = np.vstack([features_new_val.T, features_val[:,j]]).T
                fte = np.vstack([features_new_test.T, features_test[:,j]]).T
            ### scale features
            f_all = np.vstack([ft, fv, fte]) 
            ### Random Forest Classifier
 
            scaler_train = StandardScaler().fit(ft)
            fscaled_train = scaler_train.transform(ft)
            fscaled_val = scaler_train.transform(fv)

            lr = LogisticRegression(penalty="l2", class_weight="balanced", multi_class="multinomial",
                        solver="lbfgs")

            group_kfold = LabelKFold(nseg_lb["train"], n_folds=n_folds)

            params = {'C': [0.01, 0.1, 1.0, 10.0]}#,
           
            # instantiate the GridSearchCV object for searching over regularization parameters
            grid_lr = GridSearchCV(lr, params, cv=group_kfold, verbose=3, n_jobs=2,
                                             scoring="f1_weighted")
 
            grid_lr.fit(fscaled_train, labels_train)

            best_params.append(grid_lr.best_params_)
            score.append(grid_lr.score(fscaled_val, labels_val))
        score_all.append(score)
        best_params_all.append(best_params)
        mean_scores = np.array([np.mean(s) for s in score])
        best_ind = np.where(mean_scores == np.max(mean_scores))[0]
        if len(best_ind) > 1:
            best_ind = best_ind[0]
        print("The best score in round " + str(i) + " is " + str(np.max(score)))
        n_best = nfeatures.pop(best_ind)
        print("The best-ranked feature in round " + str(i) + " is " + str(n_best))
        feature_ranking.append(n_best)
        if len(features_new_train) == 0:
            features_new_train = np.atleast_2d(features_train[:,n_best]).T
            features_new_val = np.atleast_2d(features_val[:,n_best]).T
            features_new_test = np.atleast_2d(features_test[:,n_best]).T
        else:
            features_new_train = np.concatenate((features_new_train,
                                                 np.atleast_2d(features_train[:,n_best]).T), 1)
            features_new_val = np.concatenate((features_new_val,
                                               np.atleast_2d(features_val[:,n_best]).T), 1)
            features_new_test = np.concatenate((features_new_test,
                                                np.atleast_2d(features_test[:,n_best]).T), 1)

    features_new_train = np.array([features["train"][:,i] for i in feature_ranking]).T
    features_new_test = np.array([features["test"][:,i] for i in feature_ranking]).T
    features_new_val = np.array([features["val"][:,i] for i in feature_ranking]).T

    res = {"ranking":feature_ranking, "fnew_train":features_new_train,
           "fnew_val":features_new_val, "fnew_test":features_new_test,
           "scores":score_all}  

    f = open(datadir+"grs1915_greedysearch_res.dat", "w")
    pickle.dump(res, f, -1)
    f.close()

    return

def main():
    datadir= "/scratch/daniela/data/grs1915/"
    seg_length_supervised = 1024.
    n_folds = 5

    greedy_search(datadir, seg_length_supervised=seg_length_supervised, n_folds=n_folds)

    return

if __name__ == "__main__":
    main()
