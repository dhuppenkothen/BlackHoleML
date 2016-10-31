import numpy as np
import cPickle as pickle
from pandas.tools.plotting import scatter_matrix
import pandas as pd


import powerspectrum
import generaltools as gt
import feature_extraction

import glob
import scipy.stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LabelKFold
from sklearn.pipeline import Pipeline
import matplotlib.cm as cmap
from sklearn.linear_model import LogisticRegression


def load_data(filename):
 
    with open(filename, "r") as f:
        d_all = pickle.load(f)

    return d_all

class RebinTimeseries(BaseEstimator, TransformerMixin):

    def __init__(self, n=4, method="average"):

        """
        Initialize hyperparameters

        :param n: number of samples to bin
        :param method: "average" or "sum" the samples within a bin?
        :return:
        """

        self.n = n ## save number of bins to average together
        self.method = method

        return



    def fit(self,X):
        """
        I don't really need a fit method!
        """
        
        ## set number of light curves (L) and 
        ## number of samples per light curve (k)
        return self
        
        
    def transform(self, X):
        self.L, self.K = X.shape

    
        ## set the number of binned samples per light curve
        K_binned = int(self.K/self.n)
        
        ## if the number of samples in the original light curve
        ## is not divisible by n, then chop off the last few samples of 
        ## the light curve to make it divisible
        #print("X shape: " + str(X.shape))

        if K_binned*self.n < self.K:
            X = X[:,:self.n*K_binned]
        
        ## the array for the new, binned light curves
        X_binned = np.zeros((self.L, K_binned))
        
        if self.method in ["average", "mean"]:
            method = np.mean
        elif self.method == "sum":
            method = np.sum
        else:
            raise Exception("Method not recognized!")
        
        #print("X shape: " + str(X.shape))
        #print("L: " + str(self.L))
        for i in xrange(self.L):
            t_reshape = X[i,:].reshape((K_binned, self.n))
            X_binned[i,:] = method(t_reshape, axis=1)
        
        return X_binned


    def predict(self, X):
        pass
    
    def score(self, X):
        pass

    def fit_transform(self, X, y=None):

        self.fit(X)
        X_binned = self.transform(X)

        return X_binned

## boundaries for power bands
pcb = {"pa_min":0.0039, "pa_max":0.031,
       "pb_min":0.031, "pb_max":0.25,
       "pc_min":0.25, "pc_max":2.0,
       "pd_min":2.0, "pd_max":16.0}


class MakeFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, n=1, k=10, lamb=0.1, n_components=3):

        """
        Initialize hyperparameters

        Parameters
        ----------
        n : int
            The number of time steps in the light curve to rebin
            
        k : int
            number of samples to use in autoregressive model
            
        lamb : float
            the regularization parameter for the autoregressive model
            
        n_components : int
            number of components in the PCA decomposition
        
        """
        
        self.n = n
        self.k = k 
        self.lamb = lamb
        self.n_components = n_components

        return


    def fit(self,X):
        """
        I don't really need a fit method!
        """
        
        ## set number of light curves (L) and 
        ## number of samples per light curve (k)
        return self
        
        
    def transform(self, X):
        """
        transform time series into features.
        
        X should be a a matrix of dimension (N, L, J), where N is 
        the number of samples, L is the set of energy bins and J is 
        the number of time steps per light curve.
        
        """
        
        features = [] # empty list for features        
        
        # total counts in the light curve are stored in the second column for each 
        # sample
        #print("Extracting counts ...")
        counts = np.array([s[:,1] for s in X])

        #print("rebinning time series ...")
        rt = RebinTimeseries(n=self.n, method="average")
        counts_binned = rt.fit_transform(counts)
                
        #print("extracting weights from AR model ...")
        # weights of the autoregressive model
        ww = feature_extraction.linear_filter(counts_binned, k=self.k, lamb=self.lamb)

        #print("extracting PCA components from PSDs ...")
        # PCA on the power spectrum
        pca = feature_extraction.psd_pca(X, n_components=self.n_components)
        
        #print("extracting summary features ...")
        for s in X:

            features_temp = []

            ## time series summary features
            fmean, fmedian, fvar, fskew, fkurt = feature_extraction.timeseries_features(s)
            features_temp.extend([fmean, fmedian, np.log(fvar), fskew, fkurt])

            ## PSD summary features
            maxfreq, psd_a, psd_b, psd_c, psd_d, pc1, pc2 = feature_extraction.psd_features(s, pcb)
            
            if len(maxfreq) == 0:
                features_temp.extend([np.log(psd_a), np.log(psd_b), np.log(psd_c), np.log(psd_d), 
                                      np.log(pc1), np.log(pc2)])
            else:
                features_temp.extend([np.log(maxfreq), np.log(psd_a), np.log(psd_b), 
                                      np.log(psd_c), np.log(psd_d), np.log(pc1), np.log(pc2)])

            mu1, mu2, cov, skew, kurt = feature_extraction.hr_fitting(s)
            features_temp.extend([mu1, mu2])
            features_temp.extend([np.log(cov[0]), cov[1], np.log(cov[3])])
            features_temp.extend(list(skew))
            features_temp.extend(list(kurt))

            features.append(features_temp)

        features_all = np.hstack((np.array(features), np.array(ww)))
        features_all = np.hstack((np.array(features_all), np.array(pca)))
        
        return features_all


    def predict(self, X):
        pass
    
    def score(self, X):
        pass

    def fit_transform(self, X, y=None):
        
        self.fit(X)
        features = self.transform(X)
        return features

def run_parameterestimation(d_all, seg_length=1024., overlap=256., dt=0.125):

    # split the data into train, test and validation sets 
    d_all_train, d_all_val, d_all_test = feature_extraction.split_dataset(d_all, 
                                                                          train_frac = 0.5,  
                                                                          validation_frac = 0.25, 
                                                                          test_frac = 0.25, seed=20160615)


    # extract the segments
    segments_train, labels_train, nsegments_train = feature_extraction.extract_segments(d_all_train, 
                                                                                        seg_length=seg_length, 
                                                                                        overlap=overlap, 
                                                                                        dt=dt)

    segments_test, labels_test, nsegments_test = feature_extraction.extract_segments(d_all_test, 
                                                                                    seg_length=seg_length, 
                                                                                    overlap=overlap, 
                                                                                    dt=dt)

    segments_val, labels_val, nsegments_val = feature_extraction.extract_segments(d_all_val, 
                                                                                  seg_length=seg_length, 
                                                                                  overlap=overlap, 
                                                                                  dt=dt)
 
    # find all human annotated samples
    label_mask_train = labels_train != "None"
    label_mask_val= labels_val != "None"
    label_mask_test = labels_test != "None"
    
    segments_train_labelled = np.array(segments_train)[label_mask_train]
    labels_train_labelled = labels_train[label_mask_train]
    nsegments_train_labelled = nsegments_train[label_mask_train]

    segments_test_labelled = np.array(segments_test)[label_mask_test]
    labels_test_labelled = labels_test[label_mask_test]
    nsegments_test_labelled = nsegments_test[label_mask_test]

    segments_val_labelled = np.array(segments_val)[label_mask_val]
    labels_val_labelled = labels_val[label_mask_val]
    nsegments_val_labelled = nsegments_val[label_mask_val]


    # parameters to search using GridSearchCV
    nn = [1,4,8,80] ## number of samples to rebin
    kk = [2,5,10,20,50] ## number of weights in the linear filter
    lamb = [1.0, 10.,100., 1000., 10000.0 ] ## regularization parameter for ridge regression
    n_components = [1, 2, 3, 5, 10, 20, 50]
    cc = [0.001, 0.01, 0.1, 1., 10., 100., 1000.0]

    ## parameters to search using GridSearchCV
    #nn = [1.]#[1,4,8,80] ## number of samples to rebin
    #kk = [7]#[2,5,7,10,20] ## number of weights in the linear filter
    #lamb = [100]#[1.0, 10.,100., 1000. ] ## regularization parameter for ridge regression
    #n_components = [3]#[1, 2, 3, 5, 10, 20]
    #cc = [0.001, 0.01, 0.1, 1., 10., 40., 100., 1000.0]

    # folds to use in cross-validation
    n_folds = 5

    # empty lists for validation scores
    all_scores = []

    # stack all segments together
    seg_all_labelled = np.concatenate([segments_train_labelled, segments_val_labelled, segments_test_labelled])
    # stack all segment indices together
    nseg_all_labelled = np.hstack([nsegments_train_labelled, nsegments_val_labelled, nsegments_test_labelled])

    # store number of samples in each set, for use below
    ntrain = len(segments_train_labelled)
    nval = len(segments_val_labelled)
    ntest = len(segments_test_labelled)

    # loop over all parameters
    for i, n in enumerate(nn):
        for j, k in enumerate(kk):
            for l, lm in enumerate(lamb):
                for m, nc in enumerate(n_components):
                    # make features using whole feature set
                    # don't need to repeat this for each iteration of finding 
                    # the regularization parameter for the logistic regression estimator
                    mf = MakeFeatures(n=n, k=k, lamb=lm, n_components=nc)
                    features = mf.fit_transform(seg_all_labelled)

                    pars = {"n":n, "k":k, "lamb":lm, "n_components":nc}

                    # make the GroupKFold cross validation generator
                    group_kfold = LabelKFold(nsegments_train_labelled, n_folds=n_folds)

                    # make a LogisticRegression object
                    lr = LogisticRegression(penalty="l2", class_weight="balanced", multi_class="multinomial",  
                                            solver="lbfgs")

                    # instantiate the GridSearchCV object for searching over regularization parameters
                    grid_lr = GridSearchCV(lr, dict(C=cc), cv=group_kfold, verbose=3, n_jobs=1,
                                             scoring="f1_weighted")
                    # do grid search
                    grid_lr.fit(features[:ntrain], labels_train_labelled)

                    val_score = grid_lr.score(features[ntrain:ntrain+nval], labels_val_labelled)

                    score_dict = {"train_scores": grid_lr.grid_scores_, "val_score":val_score, "pars":pars}

                    all_scores.append(score_dict)

    val_scores = np.array([s["val_score"] for s in all_scores])
    mean_scores = np.array([[m.mean_validation_score for m in s["train_scores"]] for s in all_scores])
 
    print("The best cross-validation score of %.3f was achieved for parameters "%(np.max(val_scores)))
    print("The validation score for the best model is " + str(np.max(mean_scores))) 

    return all_scores


def main():
    datadir = "../../"
    filename = "grs1915_125ms_clean.dat"
    d_all = load_data(datadir+filename)

    estimator = run_parameterestimation(d_all, seg_length=1024., overlap=256., dt=0.125)

    with open(datadir+"grs1915_best_estimator.dat", "w") as f:
        pickle.dump(estimator, f, -1)

    return 

if __name__ == "__main__":
    main()


