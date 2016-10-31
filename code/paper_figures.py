
import matplotlib.pyplot as plt

import seaborn as sns

import numpy as np
import pandas as pd

import feature_engineering
import plotting

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier


try:
    import cPickle as pickle
except ImportError:
    import pickle


def plot_scores(datadir, scores):
    max_scores = []
    for s in scores:
        max_scores.append(np.max(s))

    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    fig, ax = plt.subplots(1,1,figsize=(9,7))

    ax.scatter(np.arange(len(max_scores)), max_scores, marker="o",
               c=sns.color_palette()[0], s=40)
    ax.set_xlabel("Number of features")
    ax.set_ylabel("Validation accuracy")

    plt.savefig(datadir+"grs1915_greedysearch_scores.pdf", format="pdf")
    plt.close()


def load_data(datadir, tseg=1024.0, log_features=None, ranking=None):
    features, labels, lc, \
    hr, tstart, nseg = feature_engineering.load_features(datadir, tseg,
                                                   log_features=log_features,
                                                   ranking=ranking)

    features_lb, labels_lb, lc_lb, \
    hr_lb, tstart_lb, nseg_lb = feature_engineering.labelled_data(features, labels,
                                                         lc, hr, tstart, nseg)

    fscaled, fscaled_lb = feature_engineering.scale_features(features,
                                                             features_lb)

    fscaled_full = np.vstack([fscaled["train"], fscaled["val"],
                              fscaled["test"]])

    labels_all = np.hstack([labels["train"], labels["val"], labels["test"]])

    return features, labels, lc, hr, tstart, nseg, \
           features_lb, labels_lb, lc_lb, hr_lb, nseg_lb, \
           fscaled, fscaled_lb, fscaled_full, labels_all

class AlgorithmUnrecognizedException(Exception):
    pass

def features_pca(fscaled, labels, axes=None,
                 alpha=0.8, palette="Set3", algorithm="pca"):

    #_, _, _, _, _, _, _, _, _, _, _, fscaled_full, labels_all = \
    #        load_data(datadir, tseg=tseg, log_features=log_features,
    #                  ranking=ranking)

    if algorithm == 'pca':
        pc = PCA(n_components=2)
        fscaled_trans = pc.fit(fscaled).transform(fscaled)
    elif algorithm == "tsne":
        fscaled_trans = TSNE(n_components=2).fit_transform(fscaled)
    else:
        raise AlgorithmUnrecognizedException("Not recognizing method of "+
                                             "dimensionality reduction.")

    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    # make a Figure object
    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(16,6), sharey=True)

    ax1, ax2 = axes[0], axes[1]

    labels_all = np.hstack([labels["train"], labels["val"], labels["test"]])
    
    ax1 = plotting.scatter(fscaled_trans, labels_all, ax=ax1)

    # second panel: physical labels:
    labels_phys = feature_engineering.convert_labels_to_physical(labels)

    labels_all_phys = np.hstack([labels_phys["train"], labels_phys["val"],
                                 labels_phys["test"]])

    ax2 = plotting.scatter(fscaled_trans, labels_all_phys, ax=ax2)

    plt.tight_layout()

    return ax1, ax2

def features_pca_classified(fscaled, labels_true, labels_predict, axes=None,
                            algorithm="pca"):
    if algorithm == 'pca':
        pc = PCA(n_components=2)
        fscaled_trans = pc.fit(fscaled).transform(fscaled)
    elif algorithm == "tsne":
        fscaled_trans = TSNE(n_components=2).fit_transform(fscaled)
    else:
        raise AlgorithmUnrecognizedException("Not recognizing method of "+
                                             "dimensionality reduction.")

    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    # make a Figure object
    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(16,6), sharey=True)

    ax1, ax2 = axes[0], axes[1]

    ax1 = plotting.scatter(fscaled_trans, labels_true, ax=ax1)

    # second panel: physical labels:

    ax2 = plotting.scatter(fscaled_trans, labels_predict, ax=ax2)

    plt.tight_layout()

    return ax1, ax2



def states_distribution(labels_trained_full, ax=None):
    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(9,6))

    st = pd.Series(labels_trained_full)
    nstates = st.value_counts()
    nstates.plot(kind='bar', color=sns.color_palette()[0], ax=ax)
    ax.set_ylim(0, 1.05*np.max(nstates))
    ax.set_title("Distribution of states from the supervised classification")

    return ax

def plot_eta_omega(labels_all, labels_predict_phys, axes=None):

    labels_eta_phys = labels_predict_phys[(labels_all == "eta")]
    labels_omega_phys = labels_predict_phys[(labels_all == "omega")]

    if axes is None:
        fig, axes = plt.subplots(1,2,figsize=(16,6), sharey=True)

    st = pd.Series(labels_eta_phys)
    nstates = st.value_counts()
    nstates.plot(kind='bar', color=sns.color_palette()[0], ax=axes[0])
    axes[0].set_ylim(0, 1.05*np.max(nstates))

    axes[0].set_title("Distribution for state Eta")
    axes[0].set_xlabel("States")
    axes[0].set_ylabel("Number of samples")

    st = pd.Series(labels_omega_phys)
    nstates = st.value_counts()
    nstates.plot(kind='bar', color=sns.color_palette()[0], ax=axes[1])
    axes[1].set_ylim(0, 1.05*np.max(nstates))
    axes[1].set_title("Distribution for stat Omega")
    axes[1].set_xlabel("States")
    #ax2.set_ylabel("Number of samples")

    return axes





    return ax

def supervised_validation(fscaled, fscaled_lb, labels, labels_lb, lc_lb,
                          hr_lb, datadir="./", namestr="grs1915_supervised",
                          misclassified=False):

   # full set of scaled features
    fscaled_full = np.vstack([fscaled["train"], fscaled["val"],
                              fscaled["test"]])

    # all labels in one array
    labels_all = np.hstack([labels["train"], labels["val"], labels["test"]])

    fscaled_train = fscaled_lb["train"]
    fscaled_test = fscaled_lb["test"]
    fscaled_val = fscaled_lb["val"]

    labels_train = labels_lb["train"]
    labels_test = labels_lb["test"]
    labels_val = labels_lb["val"]

    # Do RF classification
    max_depth=50
    rfc = RandomForestClassifier(n_estimators=500, max_depth=max_depth)
    rfc.fit(fscaled_train, labels_train)

    print("Training score: " + str(rfc.score(fscaled_train, labels_train)))
    print("Validation score: " + str(rfc.score(fscaled_val, labels_val)))

    lpredict_val = rfc.predict(fscaled_val)
    lpredict_test = rfc.predict(fscaled_test)

    print("Test score: " + str(rfc.score(fscaled_test, labels_test)))

    # plot the confusion matrix
#    fig, ax = plt.subplots(1,1,figsize=(9,9))
#    ax = plotting.confusion_matrix(labels_val, lpredict_val, log=True, ax=ax)
#    fig.subplots_adjust(bottom=0.15, left=0.15)
#    plt.tight_layout()

    fig, ax1 = plt.subplots(1,1, figsize=(10.5,9))

    sns.set_style("whitegrid") 
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20) 
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20) 
    plt.rc('ytick', labelsize=20) 

    im = ax1.pcolormesh(log_cm, cmap=cmap.viridis, 
                        vmin=vmin, vmax=vmax)
    #ax1.set_title('Confusion matrix')
    ax1.set_ylabel('True label')
    ax1.set_xlabel('Predicted label')
    ax1.set_xticks(np.arange(len(unique_labels))+0.5)
    ax1.set_xticklabels(unique_labels, rotation=90)

    ax1.set_yticks(np.arange(len(unique_labels))+0.5)
    ax1.set_yticklabels(unique_labels)

    tick_pos = np.array([2, 5, 10, 20, 50, 100])

    ticks = [0]
    ticks.extend(np.log10(tick_pos))

    tick_pos = np.hstack([[0], tick_pos])
    cbar = fig.colorbar(im, ticks=ticks)

    labels = ["%i"%f for f in tick_pos]

    cbar.ax.set_yticklabels(labels);
    plt.tight_layout()

    plt.savefig(datadir+namestr+"_cm.pdf", format="pdf")
    plt.close()

    if misclassified:
        # plot misclassified examples:
        plotting.plot_misclassifieds(fscaled_val, lpredict_val, labels_val,
                                     lc_lb["val"], hr_lb["val"],
                                     nexamples=20, namestr="misclassified",
                                     datadir=datadir)

    return


def plot_asm_with_classes(labels_predict, tstart, fig=None, ax=None,
                          datadir="./", palette="Set3"):

    pred_label_set = np.unique(labels_predict)
    colors = sns.color_palette(palette, len(pred_label_set))

    tstart_all = np.concatenate([tstart["train"],
                                 tstart["val"],
                                 tstart["test"]])

    # load ASM light curve
    asm = np.loadtxt(datadir+"grs1915_asm_lc.txt",skiprows=5)
    asm_time = asm[:,0]
    asm_cr = asm[:,1]
    asm_total = asm_time[-1]-asm_time[0]
    print("The ASM light curve covers a total of %i days"%asm_total)

    mjdrefi = 49353.
    tstart_all_days = tstart_all/(60.*60.*24.)
    tstart_all_mjd = tstart_all_days + mjdrefi

    ## each light curve covers 500 days
    plot_len = 500.
    start_time = asm_time[0]
    end_time = start_time + plot_len
    i = 0

    if fig is  None and ax is None:
        fig, ax = plt.subplots(1,1,figsize=(12,15))
    sns.set_style("white")
    # Turn off axis lines and ticks of the big subplot

    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top='off', bottom='off', left='off', right='off')

    sns.set_context("notebook", font_scale=1.0, rc={"axes.labelsize": 16})

    sns.set_style("whitegrid")

    plt.rc("font", size=16, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=16, labelsize=16)
    plt.rc("text", usetex=True)

    plt.subplots_adjust(top=0.95, bottom=0.08, left=0.08, right=0.97, hspace=0.2)

    current_palette = sns.color_palette(palette, len(pred_label_set))
    while end_time <= asm_time[-1]:
        print("I am on plot %i."%i)
        ax1 = fig.add_subplot(11,1,i+1)
        ax1.errorbar(asm[:,0], asm[:,1], yerr = asm[:,2], linestyle="steps-mid")
        for k, col in zip(pred_label_set, colors):
            tstart_members = tstart_all_mjd[labels_predict == k]
            ax1.plot(tstart_members, np.ones(len(tstart_members))*240.,"o", color=col, label="state " + str(k))
        ax1.set_xlim([start_time, end_time])
        ax1.set_ylim([1.0, 299.0])
        plt.yticks(np.arange(3)*100.0+100.0, [100, 200, 300]);

        start_time +=plot_len
        end_time += plot_len
        i+=1

    ax.set_xlabel("Time in MJD", fontsize=18)
    ax.set_ylabel("Count rate [counts/s]", fontsize=18)

    return fig, ax

def supervised_all(fscaled_full, labels_all, tstart,
                   datadir="./", namestr="grs1915_supervised"):

    # classification on the whole data set at the same time:
    fscaled_cls = fscaled_full[labels_all != "None"]
    labels_cls = labels_all[labels_all != "None"]

    print("Running the classifier ...")

    max_depth = 200.0
    rfc = RandomForestClassifier(n_estimators=500,max_depth=max_depth)
    rfc.fit(fscaled_cls, labels_cls)

    labels_trained_full = rfc.predict(fscaled_full)

    print("Plotting the distribution of states ...")
    fig, ax = plt.subplots(1,1,figsize=(9,7))
    ax = states_distribution(labels_trained_full, ax=None)
    fig.subplots_adjust(bottom=0.15)
    plt.savefig(datadir+namestr+"_states_histogram.pdf", format="pdf")
    plt.close()

    print("Plotting PCA representation of states ...")
    # do the PCA plots with the full classified states:
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    ax1, ax2 = features_pca_classified(fscaled_full, labels_all, labels_trained_full,
                            axes=axes, algorithm="pca")
    ax2.set_ylabel("")
    plt.tight_layout()
    plt.savefig(datadir+namestr+"_features_pca.pdf", format="pdf")
    plt.close()

    print("Plotting ASM light curves with classified states ...")
    # Plot ASM light curve with classified states:
    fig, ax = plt.subplots(1,1,figsize=(16,22))
    fig, ax = plot_asm_with_classes(labels_trained_full, tstart, datadir=datadir,
                                    fig=fig, ax=ax, palette="Set3")
    plt.savefig(datadir+namestr+"asm_lc_all.pdf")
    plt.close()

    print("plotting transition matrix ...")
    fig, ax = plt.subplots(1,1, figsize=(9,9))
    ax = plotting.transition_matrix(labels_trained_full, ax=ax, log=True)
    fig.subplots_adjust(bottom=0.15, left=0.15)
    plt.tight_layout()
    plt.savefig(datadir+namestr+"_transmat.pdf", format="pdf")
    plt.close()

    print("Finished!")

    return

def all_figures():
    datadir = "../../"

    # read in the results from the greedy feature engineering
    with open(datadir+"grs1915_greedysearch_res.dat" ,'r') as f:
        data = pickle.load(f)

    scores = data["scores"]
    ranking = data["ranking"]

    # Plot the maximum validation score for each feature in the greedy search
    plot_scores(datadir, scores)

    # Read in the data:
    features, labels, lc, hr, tstart, \
        features_lb, labels_lb, lc_lb, hr_lb, \
        fscaled, fscaled_lb, fscaled_full, labels_all = \
            load_data(datadir, tseg=1024.0, log_features=None, ranking=None)

    # First the comparison between Belloni classification + other classification
    # using PCA
    ax1, ax2 = features_pca(fscaled_full, labels_all, axes=None,
                            alpha=0.8, palette="Set3", algorithm="pca")

    ax2.set_ylabel("")
    plt.tight_layout()
    plt.savefig(datadir+"grs1915_features_pca.pdf", format="pdf")
    plt.close()

    # Same as before, using t-SNE
    ax1, ax2 = features_pca(fscaled_full, labels_all, axes=None,
                            alpha=0.8, palette="Set3", algorithm="tsne")

    ax2.set_ylabel("")
    plt.tight_layout()
    plt.savefig(datadir+"grs1915_features_tsne.pdf", format="pdf")
    plt.close()

    # get classified features + labels
    supervised_validation(fscaled, fscaled_lb, labels, labels_lb, lc_lb,
                          hr_lb, datadir="./", namestr="grs1915_supervised",
                          misclassified=False)

    # supervised learning of the whole data set:
    supervised_all(fscaled_full, labels_all, tstart,
                   datadir=datadir, namestr="grs1915_supervised")

    # same using physical labels:
    labels_phys = feature_engineering.convert_labels_to_physical(labels)
    labels_phys_lb = feature_engineering.convert_labels_to_physical(labels_lb)

    labels_all_phys = np.hstack([labels_phys["train"], labels_phys["val"],
                                 labels_phys["test"]])

    supervised_validation(fscaled, fscaled_lb, labels_phys, labels_phys_lb,
                          lc_lb, hr_lb, datadir="./",
                          namestr="grs1915_supervised_phys",
                          misclassified=False)

    supervised_all(fscaled_full, labels_all_phys, tstart,
                   datadir=datadir, namestr="grs1915_supervised_phys")

    # classification on the whole data set at the same time:
    fscaled_cls = fscaled_full[labels_all != "None"]
    labels_cls = labels_all[labels_all != "None"]

    max_depth = 200.0
    rfc = RandomForestClassifier(n_estimators=500,max_depth=max_depth)
    rfc.fit(fscaled_cls, labels_cls)

    labels_trained_full = rfc.predict(fscaled_full)

    # Figures showing distributions of eta and omega
    fig, axes = plt.subplots(1,2,figsize=(16,6))
    plot_eta_omega(labels_all, labels_trained_full, axes=axes)
    plt.tight_layout()
    plt.savefig(datadir+"grs1915_supervised_eta_omega.pdf", format="pdf")
    plt.close()

    return
