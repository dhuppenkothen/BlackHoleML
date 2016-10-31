import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import seaborn as sns

import numpy as np
import itertools

import sklearn.metrics

from collections import Counter


def _compute_trans_matrix(labels, order="row"):

    print("order: " + order)
    unique_labels = np.unique(labels)
    nlabels = len(unique_labels)

    labels_numerical = np.array([np.where(unique_labels == l)[0][0] \
                                 for l in labels])
    labels_numerical = labels_numerical.flatten()

    transmat = np.zeros((nlabels,nlabels))
    for (x,y), c in Counter(zip(labels_numerical,
                                labels_numerical[1:])).iteritems():
        transmat[x,y] = c

    if order == "row":
        print("I am here!") 
        transmat_p = transmat/np.sum(transmat, axis=1)
    elif order == "column": 
        print("This is column")
        transmat_p = transmat/np.sum(transmat, axis=0)
    else:
        raise Exception("argument to keyword 'order' not recognized!")
 
    return unique_labels, transmat, transmat_p



def transition_matrix(labels, labels_for_plotting, ax=None, log=False, fig=None, order="column", title="Transition Matrix"):
    """
    Plot a transition matrix. 


    """

    if ax is None and fig is None:
        fig, ax = plt.subplots(1,1, figsize=(9,9))

    unique_labels, transmat, transmat_p = _compute_trans_matrix(labels, order=order)

    if log:
        if np.any(transmat_p == 0):
            transmat_p = transmat_p + \
                         np.min(transmat_p[np.nonzero(transmat_p)])/10.0

        transmat_p = np.log(transmat_p)
    sns.set_style("white")

    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    if log:
         im = ax.imshow(transmat_p.T, interpolation="nearest", cmap=cmap.Blues, vmin=np.min(transmat_p), vmax=np.max(transmat_p))
    else:
         im = ax.imshow(transmat_p.T, interpolation="nearest", cmap=cmap.Blues, vmin=0.0, vmax=np.max(transmat_p))
    print(transmat_p)
    ax.set_title(title)
    fig.colorbar(im)

    ax.set_ylabel('Initial state')
    ax.set_xlabel('Final state')


    thresh = transmat_p.max() / 2.
    for i, j in itertools.product(range(transmat_p.shape[0]), range(transmat_p.shape[1])):
        ax.text(j, i, r"$%.3f$"%transmat_p.T[i, j], fontdict={"size":16},
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if transmat_p.T[i, j] > thresh else "black")

#    labels_for_plotting = [r"$\%s$"%l for l in unique_labels]

    tick_marks = np.arange(len(unique_labels))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(labels_for_plotting)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(labels_for_plotting)


    return fig, ax


def confusion_matrix(labels_true, labels_pred, classes, log=False, normalize=False,
                     title='Confusion matrix', 
                     fig=None, ax=None, cmap=cmap.viridis):


    """
    Plot a confusion matrix between true and predicted labels
    from a machine learning classification task.

    Parameters
    ----------
    labels_true : iterable
        List or array with true labels

    labels_pred : iterable
        List or array with predicted labels

    log : bool
        Plot original confusion matrix or the log of the confusion matrix?
        Default is False
  
    fig : matplotlib.Figure object 
        A Figure object to plot into

    ax : matplotlib.Axes object
        An axes object to plot into

    cm : matplotlib.colormap
        A matplotlib colour map

    Returns
    -------
    ax : matplotlib.Axes object
        The Axes object with the plot

    """
    sns.set_style("white")
    if ax is None or fig is None:
        fig, ax = plt.subplots(1,1,figsize=(9,6))

    unique_labels = np.unique(labels_true)
    confmatrix = sklearn.metrics.confusion_matrix(labels_true, labels_pred, labels=unique_labels)

    if log:
        confmatrix = np.log10(confmatrix)
        cm_mask = np.isfinite(confmatrix) == False
        confmatrix[cm_mask] = -5.
        cm_mask_new = confmatrix != -5
        vmin = np.min(confmatrix[cm_mask_new])-0.3
        vmax = np.max(confmatrix[cm_mask_new])
    else:
        vmin = np.min(confmatrix)
        vmax = np.max(confmatrix)

    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    #im = ax.pcolormesh(confmatrix, cmap=cmap.viridis, 
    #                    vmin=vmin, vmax=vmax)

    if ax is None and fig is None:
        fig, ax = plt.subplots(1,1, figsize=(10, 10))
    im = ax.imshow(confmatrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    fig.colorbar(im)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    if normalize:
        confmatrix = confmatrix.astype('float') / confmatrix.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(confmatrix)

    thresh = confmatrix.max() / 2.
    for i, j in itertools.product(range(confmatrix.shape[0]), range(confmatrix.shape[1])):
        ax.text(j, i, confmatrix[i, j], fontdict={"size":16},
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if confmatrix[i, j] > thresh else "black")


    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    plt.tight_layout()
    return ax


def scatter(features, labels, ax=None, palette="Set3", alpha=0.8, colours=None, ylabel=True):
    """
    Make a scatter plot of dimensions 0 and 1 in features, with scatter
    points coloured by labels.

    Parameters
    ----------
    features : matrix (N, M)
        A (N, M) matrix of `features` with N samples and M features for each
        sample.

    labels : iterable
        A list or array of N labels corresponding to the N feature vectors
        in `features`.

    ax : matplotlib.Axes object
        The Axes object to plot into

    palette : str
        The string of the color palette to use for the different classes
        By default, "Set3" is used.

    alpha : {0,1} float
        Float between 0 and 1 controlling the transparency of the scatter
        points.

    Returns
    -------
    ax : matplotlib.Axes
        The Axes object with the plot.

    """
    sns.set_style("whitegrid")
    plt.rc("font", size=24, family="serif", serif="Computer Sans")
    plt.rc("axes", titlesize=20, labelsize=20)
    plt.rc("text", usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    unique_labels = np.unique(labels)

    # make a Figure object
    if ax is None:
        fig, ax = plt.subplots(1,1,figsize=(9,6), sharey=True)

    xlim = [np.min(features[:,0])-0.5, np.max(features[:,0])+3.5]
    ylim = [np.min(features[:,1])-0.5, np.max(features[:,1])+0.5]

    # If None is in labels, delete from set of unique labels and
    # plot all samples with label None in grey
    if "None" in unique_labels:
        unique_labels = np.delete(unique_labels,
                                  np.where(unique_labels == "None")[0])

        # first plot the unclassified examples:
        ax.scatter(features[labels == "None",0],
                   features[labels == "None",1],
                   color="grey", alpha=alpha, label="unclassified")

    if colours is None:
        # now make a color palette:
        current_palette = sns.color_palette(palette, len(unique_labels))
    else:
        current_palette = colours

    for l, c in zip(unique_labels, current_palette):
        ax.scatter(features[labels == l,0],
                   features[labels == l,1], s=40,
                   color=c, alpha=alpha, label=l)

    ax.set_xlabel("PCA Component 1")
    if ylabel:
        ax.set_ylabel("PCA Component 2")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.legend(loc="upper right", prop={"size":14})

    return ax


import powerspectrum

def plot_misclassifieds(features, trained_labels, real_labels, lc_all, hr_all,
                        nexamples=6, namestr="misclassified", datadir="./"):

    """
    Find all mis-classified light curves and plot them with examples of the
    real and false classes.

    Parameters
    ----------
    features : numpy.ndarray
        The (N,M) array with N samples (rows) and M features (columns) per
        sample

    trained_labels : iterable
        The list or array with the trained labels

    real_labels : iterable
        The list or array with the true labels

    lc_all : list
        A list of N light curves corresponding to each sample

    hr_all : list
        A list of N hardness ratio measurements corresponding to each sample

    nexamples : int
        The number of examples to plot; default is 6

    namestr : str
        The string to append to each plot for saving to disc,
        default: "misclassified"

    datadir : str
        The path of the directory to save the figures in


    """
    misclassifieds = []
    for i,(f, lpredict, ltrue, lc, hr) in enumerate(zip(features,
                                                        trained_labels,
                                                        real_labels, lc_all,
                                                        hr_all)):
        if lpredict == ltrue:
            continue
        else:
            misclassifieds.append([f, lpredict, ltrue, lc, hr])

    for j,m in enumerate(misclassifieds):
        pos_human = np.random.choice([0,3], p=[0.5, 0.5])
        pos_robot = int(3. - pos_human)

        f = m[0]
        lpredict = m[1]
        ltrue = m[2]
        times = m[3][0]
        counts = m[3][1]
        hr1 = m[4][0]
        hr2 = m[4][1]
        print("Predicted class is: " + str(lpredict))
        print("Human classified class is: " + str(ltrue))
        robot_all = [[lp, lt, lc, hr] for lp, lt, lc, hr in \
                     zip(real_labels, trained_labels, lc_all, hr_all)\
                     if lt == lpredict ]
        human_all = [[lp, lt, lc, hr] for lp, lt, lc, hr in \
                     zip(real_labels, trained_labels, lc_all, hr_all)\
                     if lt == ltrue ]

        np.random.shuffle(robot_all)
        np.random.shuffle(human_all)
        robot_all = robot_all[:6]
        human_all = human_all[:6]

        sns.set_style("darkgrid")
        current_palette = sns.color_palette()
        fig = plt.figure(figsize=(10,15))

        def plot_lcs(times, counts, hr1, hr2, xcoords, ycoords,
                     colspan, rowspan):
            #print("plotting in grid point " + str((xcoords[0], ycoords[0])))
            ax = plt.subplot2grid((9,6),(xcoords[0], ycoords[0]),
                                  colspan=colspan, rowspan=rowspan)
            ax.plot(times, counts, lw=2, linestyle="steps-mid", rasterized=True)
            ax.set_xlim([times[0], times[-1]])
            ax.set_ylim([0.0, 12000.0])
            #print("plotting in grid point " + str((xcoords[1], ycoords[1])))

            ax = plt.subplot2grid((9,6),(xcoords[1], ycoords[1]),
                                  colspan=colspan, rowspan=rowspan)
            ax.scatter(hr1, hr2, facecolor=current_palette[1],
                       edgecolor="none", rasterized=True)
            ax.set_xlim([.27, 0.85])
            ax.set_ylim([0.04, 0.7])

            #print("plotting in grid point " + str((xcoords[2], ycoords[2])))
            ax = plt.subplot2grid((9,6),(xcoords[2], ycoords[2]),
                                  colspan=colspan, rowspan=rowspan)
            dt = np.min(np.diff(times))
            ps = powerspectrum.PowerSpectrum(times, counts=counts/dt,
                                             norm="rms")
            ax.loglog(ps.freq[1:], ps.ps[1:], linestyle="steps-mid",
                      rasterized=True)
            ax.set_xlim([ps.freq[1], ps.freq[-1]])
            ax.set_ylim([1.e-6, 10.])

            return

        ## first plot misclassified:
        plot_lcs(times, counts, hr1, hr2, [0,0,0], [0,2,4], 2, 2)

        ## now plot examples
        for i in range(4):
            r = robot_all[i]
            h = human_all[i]

            plot_lcs(h[2][0], h[2][1], h[3][0], h[3][1], [i+2, i+2, i+2],
                     [pos_human, pos_human+1, pos_human+2], 1, 1)
            plot_lcs(r[2][0], r[2][1], r[3][0], r[3][1], [i+2, i+2, i+2],
                     [pos_robot, pos_robot+1, pos_robot+2], 1, 1)

        ax = plt.subplot2grid((9,6),(8,pos_human+1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel("Human: %s"%ltrue, fontsize=20)
        ax = plt.subplot2grid((9,6),(8,pos_robot+1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xlabel("Robot: %s"%lpredict, fontsize=20)
        plt.savefig(datadir,"misclassified%i.pdf"%j, format="pdf")
        plt.close()

    return
