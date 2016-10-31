
import matplotlib.pyplot as plt
import numpy as np
import glob
import copy

import pandas as pd
import astroML.stats
import scipy.stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    # Python 2
    import cPickle as pickle

except:
    # Python 3
    import pickle

import linearfilter
import parameter_estimation as pe

## set the seed for the random number generator
## such that my "random choices" remain reproducible

from BayesPSD import lightcurve, powerspectrum

#from BayesPSD import lightcurve, powerspectrum

def split_dataset(d_all, train_frac = 0.5, validation_frac = 0.25, test_frac = 0.25, seed=20160615):
    """
    Split data set into train, test and validation sets. 

    Parameters
    ----------
    d_all : iterable
        A list of data segments from the GRS 1915+105 data set
 
    train_frac : float (0,1)
        Fraction of data to be placed in the training data set
 
    validation_frac : float (0,1)
        Fraction of data to be placed in the validation data set

    test_frac : float (0,1)
        Fraction of data to be placed in the test data set

    seed : int
        Seed for the random number generator shuffling the data before splitting into 
        train, test and validation sets

    Returns
    -------
    d_all_train, d_all_val, d_all_test : iterables
        The split data set

    """
   
    np.random.seed(seed)

    ## total number of light curves
    n_lcs = len(d_all)

    indices = np.arange(len(d_all))
    np.random.shuffle(indices)

    ## let's pull out light curves for three data sets into different variables.
    d_all_train = [d_all[i] for i in indices[:int(train_frac*n_lcs)]]
    d_all_test = [d_all[i] for i in indices[int(train_frac*n_lcs):int((train_frac + test_frac)*n_lcs)]]

    d_all_val = [d_all[i] for i in indices[int((train_frac + test_frac)*n_lcs):]]

    print("There are %i light curves in the validation set."%len(d_all_val))
    print("There are %i light curves in the test set."%len(d_all_test))

    states = [d[1] for d in d_all_train]
    st = pd.Series(states)
    print("The number of states in the training set is %i"%len(st.value_counts()))

    states = [d[1] for d in d_all_test]
    st = pd.Series(states)
    print("The number of states in the test set is %i"%len(st.value_counts()))

    states = [d[1] for d in d_all_val]
    st = pd.Series(states)
    print("The number of states in the validation set is %i"%len(st.value_counts()))

    for da,n in zip([d_all_train, d_all_val, d_all_test], ["training", "validation", "test"]):
        print("These is the distribution of states in the %s set: "%n)
        states = [d[1] for d in da]
        st = pd.Series(states)
        print(st.value_counts())
        print("================================================================")

    return d_all_train, d_all_val, d_all_test


## This function is also in grs1915_utils.py!
def extract_segments(d_all, seg_length = 256., overlap=64., dt=0.125):
    """ Extract light curve segments from a list of light curves.
        Each element in the list is a list with two elements:
        - an array that contains the light curve in three energy bands
        (full, low energies, high energies) and
        - a string containing the state of that light curve.

        Parameters
        ==========
        seg_length : float
            The length of each segment. Bits of data at the end of a light curve
            that are smaller than seg_length will not be included. 
        
        overlap : float 
            This is actually the interval between start times of individual segments,
            i.e. by default light curves start 64 seconds apart. The actual overlap is 
            seg_length-overlap
            
        dt : float
            Set the time resolution of the light curve by hand. I allow this because 
            in the GRS 1915+105 data, there are rounding errors affecting the time 
            resolution, which in turn affects the length of the segments.

    """
    segments, labels, nsegments = [], [], [] ## labels for labelled data

    for i,d_seg in enumerate(d_all):

        ## data is an array in the first element of d_seg
        data = d_seg[0]
        ## state is a string in the second element of d_seg
        state = d_seg[1]

        ## if the length of the light curve is shorter than the
        ## chosen segment length, discard this light curve
        length = data[-1,0] - data[0,0]
        if length < seg_length:
            continue


        ## compute the number of time bins in a segment
        nseg = int(seg_length/dt)
        ## compute the number of time bins to start of next segment
        noverlap = overlap/dt

        istart = 0
        iend = nseg

        while iend < len(data):
            if iend-istart != nseg:
                istart += noverlap
                iend += noverlap
                continue
            dtemp = data[istart:iend]
            segments.append(dtemp)
            labels.append(state)
            istart += noverlap
            iend += noverlap

    	    nsegments.append(i)

    ## convert labels to strings so I don't have a weird
    ## combination of None objects and strings
    labels = np.array([str(l) for l in labels])

    return segments, labels, np.array(nsegments)

def extract_data(d_all, val=True, train_frac=0.5, validation_frac=0.25, test_frac = 0.25,
                  seg=True, seg_length=1024., overlap = 128., seed=20160615):

    """
    Split data into training, test (and validation) sets and make segments of equal duration with 
    a set overlap.

    Parameters
    ----------
    d_all : iterable
        A list of data segments from the GRS 1915+105 data set
 
    val : bool
        If true, construct a validation set along with training and test sets

    train_frac : float (0,1)
        Fraction of data to be placed in the training data set
 
    validation_frac : float (0,1)
        Fraction of data to be placed in the validation data set

    test_frac : float (0,1)
        Fraction of data to be placed in the test data set

    seg : bool
        If true, extract segments of equal length from the data

    seg_length : float
        The duration, in seconds, of each segment

    overlap : float
        The interval, in seconds, between starting points of each segment

    seed : int
        Seed for the random number generator shuffling the data before splitting into 
        train, test and validation sets

    Returns
    -------
    [[seg_train, labels_train, nseg_train],
     [seg_val, labels_val, nseg_val],
     [seg_test, labels_test, nseg_test]]
         A list of training, validation and test sets with:
             * the segments extracted
             * the class label for each segment
             * a numeric identifier to match each segment to the observation it came from

    """
    ## print light curve statistics
    print("Number of light curves:  " + str(len(d_all)))
    states = [d[1] for d in d_all]
    st = pd.Series(states)
    st.value_counts()

    d_all_train, d_all_val, d_all_test = split_dataset(d_all, train_frac=train_frac, validation_frac=validation_frac, 
                                                       test_frac=test_frac, seed=seed)

    if not val:
        d_all_train  = d_all_train + d_all_val

    if seg:
        seg_train, labels_train, nseg_train = extract_segments(d_all_train, seg_length=seg_length, 
                                                               overlap=overlap, dt=dt)
        seg_test, labels_test, nseg_test = extract_segments(d_all_test, seg_length=seg_length, 
                                                            overlap=overlap, dt=dt)

        if val:
            seg_val, labels_val, nseg_val = extract_segments(d_all_val, seg_length=seg_length, 
                                                             overlap=overlap, dt=dt)


    else:
        seg_train = [d[0] for d in d_all_train]
        labels_train = [d[1] for d in d_all_train]
        nseg_train = [1 for d in d_all_train]

        seg_test = [d[0] for d in d_all_test]
        labels_test = [d[1] for d in d_all_test]
        nseg_test = [1 for d in d_all_test]

        if val:
            seg_val = [d[0] for d in d_all_val]
            labels_val = [d[1] for d in d_all_val]
            nseg_val = [1 for d in d_all_val]


        ## Let's print some details on the different segment data sets
        print("There are %i segments in the training set."%len(seg_train))
        if val:
            print("There are %i segments in the validation set."%len(seg_val))
        print("There are %i segments in the test set."%len(seg_test))
        if val:
            labelset = [labels_train, labels_val, labels_test]
            keys = ["training", "validation", "test"]
        else:
            labelset = [labels_train, labels_test]
            keys = ["training", "test"]

        for la,n in zip(labelset, keys):
            print("These is the distribution of states in the %s set: "%n)
            st = pd.Series(la)
            print(st.value_counts())
            print("================================================================")

    if val:
        return [[seg_train, labels_train, nseg_train],
                [seg_val, labels_val, nseg_val],
                [seg_test, labels_test, nseg_test]]
    else:
        return [[seg_train, labels_train, nseg_train],
                [seg_test, labels_test, nseg_test]]

######################################################################################################################
#### FUNCTIONS FOR FEATURE EXTRACTION ################################################################################
######################################################################################################################

### FIRST PART: TIME SERIES FEATURES

def timeseries_features(seg):
    """ 
    Extract summary features from the time series.

    Parameters:
    -----------
    seg : numpy.ndarray
        A single data segment

    Returns
    -------
    fmean : float
        The mean count rate of the segment

    fmedian : float
        The median count rate of the segment

    fvar : float
        The variance in the count rate of the segment

    skew : float
        The skewness of the count rate of the segment
 
    kurt : float
        The kurtosis of the count rate of the segment

    """
    counts = seg[:,1]
    fmean = np.mean(counts)
    fmedian = np.median(counts)
    fvar = np.var(counts)
    skew = scipy.stats.skew(counts)
    kurt = scipy.stats.kurtosis(counts)
    return fmean, fmedian, fvar, skew, kurt

def linear_filter(counts_scaled, k=10, lamb=None):
    """
    Extract features from an autoregressive process.
    :param counts_scaled: numpy ndarray (nsamples, ntimebins) with SCALED TOTAL COUNTS
    :return ww_all: numpy ndarray (nsamples, k) of weights
    """

    ## initialize the LinearFilter object
    lf = linearfilter.LinearFilter(k=k, lamb=lamb)

    ww_all = np.zeros((len(counts_scaled), k))

    ## loop through all light curves and compute the weight vector for each
    for i,c in enumerate(counts_scaled):
        ww_all[i,:] = lf.fit_transform(c)[0]
    return ww_all




#### FOURIER DOMAIN FEATURES


## boundaries for power bands
## From Heil et al, 2014
pcb = {"pa_min":0.0039, "pa_max":0.031,
       "pb_min":0.031, "pb_max":0.25,
       "pc_min":0.25, "pc_max":2.0,
       "pd_min":2.0, "pd_max":16.0}

def rebin_psd(freq, ps, n=10, type='average'):

    nbins = int(len(freq)/n)
    df = freq[1] - freq[0]
    T = freq[-1] - freq[0] + df
    bin_df = df*n
    binfreq = np.arange(nbins)*bin_df + bin_df/2.0 + freq[0]

    nbins_new = int(len(ps)/n)
    ps_new = ps[:nbins_new*n]
    binps = np.reshape(np.array(ps_new), (nbins_new, int(n)))
    binps = np.sum(binps, axis=1)
    if type in ["average", "mean"]:
        binps = binps/np.float(n)
    else:
        binps = binps

    if len(binfreq) < len(binps):
        binps= binps[:len(binfreq)]

    return binfreq, binps



def logbin_periodogram(freq, ps, percent=0.01):
    df = freq[1]-freq[0]
    minfreq = freq[0]*0.5
    maxfreq = freq[-1]
    binfreq = [minfreq, minfreq+df]
    while binfreq[-1] <= maxfreq:
        binfreq.append(binfreq[-1] + df*(1.+percent))
        df = binfreq[-1]-binfreq[-2]
    binps, bin_edges, binno = scipy.stats.binned_statistic(freq, ps, statistic="mean", bins=binfreq)

    nsamples = np.array([len(binno[np.where(binno == i)[0]]) for i in xrange(np.max(binno))])
    df = np.diff(binfreq)
    binfreq = binfreq[:-1]+df/2.
    #return binfreq, binps, std_ps, nsamples
    return np.array(binfreq), np.array(binps), nsamples


def psd_features(seg, pcb):
    """
    Computer PSD-based features.
    seg: data slice of type [times, count rates, count rate error]^T
    pcb: frequency bands to use for power colours
    """

    times = seg[:,0]
    dt = times[1:] - times[:-1]
    dt = np.min(dt)

    counts = seg[:,1]*dt

    ps = powerspectrum.PowerSpectrum(times, counts=counts, norm="rms")
    ps.freq = np.array(ps.freq)
    ps.ps = np.array(ps.ps)*ps.freq

    bkg = np.mean(ps.ps[-100:])

    freq = np.array(ps.freq[1:])
    ps = ps.ps[1:]

    binfreq, binps, nsamples = logbin_periodogram(freq, ps)

    bkg = np.mean(binps[-30:])

    binps -= bkg

    fmax_ind = np.where(binfreq*binps == np.max(binfreq*binps))
    maxfreq = binfreq[fmax_ind[0]]

    ## find power in spectral bands for power-colours
    pa_min_freq = freq.searchsorted(pcb["pa_min"])
    pa_max_freq = freq.searchsorted(pcb["pa_max"])

    pb_min_freq = freq.searchsorted(pcb["pb_min"])
    pb_max_freq = freq.searchsorted(pcb["pb_max"])

    pc_min_freq = freq.searchsorted(pcb["pc_min"])
    pc_max_freq = freq.searchsorted(pcb["pc_max"])

    pd_min_freq = freq.searchsorted(pcb["pd_min"])
    pd_max_freq = freq.searchsorted(pcb["pd_max"])

    psd_a = np.sum(ps[pa_min_freq:pa_max_freq])
    psd_b = np.sum(ps[pb_min_freq:pb_max_freq])
    psd_c = np.sum(ps[pc_min_freq:pc_max_freq])
    psd_d = np.sum(ps[pd_min_freq:pd_max_freq])
    pc1 = np.sum(ps[pc_min_freq:pc_max_freq])/np.sum(ps[pa_min_freq:pa_max_freq])
    pc2 = np.sum(ps[pb_min_freq:pb_max_freq])/np.sum(ps[pd_min_freq:pd_max_freq])

    return maxfreq, psd_a, psd_b, psd_c, psd_d, pc1, pc2

def make_psd(segment, navg=1):

    times = segment[:,0]
    dt = times[1:] - times[:-1]
    dt = np.min(dt)

    counts = segment[:,1]*dt

    tseg = times[-1]-times[0]
    nlc = len(times)
    nseg = int(nlc/navg)

    if navg == 1:
        ps = powerspectrum.PowerSpectrum(times, counts=counts, norm="rms")
        ps.freq = np.array(ps.freq)
        ps.ps = np.array(ps.ps)*ps.freq
        return ps.freq, ps.ps
    else:
        ps_all = []
        for n in xrange(navg):
            t_small = times[n*nseg:(n+1)*nseg]
            c_small = counts[n*nseg:(n+1)*nseg]
            ps = powerspectrum.PowerSpectrum(t_small, counts=c_small, norm="rms")
            ps.freq = np.array(ps.freq)
            ps.ps = np.array(ps.ps)*ps.freq
            ps_all.append(ps.ps)

        ps_all = np.average(np.array(ps_all), axis=0)

    return ps.freq, ps_all

epsilon = 1.e-8

def total_psd(seg, bins):
    times = seg[:,0]
    dt = times[1:] - times[:-1]
    dt = np.min(dt)
    counts = seg[:,1]*dt

    ps = powerspectrum.PowerSpectrum(times, counts=counts, norm="rms")
    binfreq, binps, binsamples = ps.rebin_log()
    bkg = np.mean(ps.ps[-100:])
    binps -= bkg

    return np.array(binfreq), np.array(binps)

def compute_hrlimits(hr1, hr2):
    min_hr1 = np.min(hr1)
    max_hr1 = np.max(hr1)

    min_hr2 = np.min(hr2)
    max_hr2 = np.max(hr2)
    return [[min_hr1, max_hr1], [min_hr2, max_hr2]]

def hr_maps(seg, bins=30, hrlimits=None):
    times = seg[:,0]
    counts = seg[:,1]
    low_counts = seg[:,2]
    mid_counts = seg[:,3]
    high_counts = seg[:,4]
    hr1 = np.log(mid_counts/low_counts)
    hr2 = np.log(high_counts/low_counts)

    if hrlimits is None:
        hr_limits = compute_hrlimits(hr1, hr2)
    else:
        hr_limits = hrlimits

    h, xedges, yedges = np.histogram2d(hr1, hr2, bins=bins,
                                       range=hr_limits)
    h = np.rot90(h)
    h = np.flipud(h)
    hmax = np.max(h)
    hmask = np.where(h > hmax/20.)
    hmask1 = np.where(h < hmax/20.)
    hnew = copy.copy(h)
    hnew[hmask[0], hmask[1]] = 1.
    hnew[hmask1[0], hmask1[1]] = 0.0
    return xedges, yedges, hnew

def hr_fitting(seg):
    counts = seg[:,1]
    low_counts = seg[:,2]
    mid_counts = seg[:,3]
    high_counts = seg[:,4]
    hr1 = mid_counts/low_counts
    hr2 = high_counts/low_counts

    hr1_mask = np.where(np.isfinite(hr1) == True)
    hr1 = hr1[hr1_mask]
    hr2 = hr2[hr1_mask]

    hr2_mask = np.where(np.isfinite(hr2) == True)
    hr1 = hr1[hr2_mask]
    hr2 = hr2[hr2_mask]

    mu1 = np.mean(hr1)
    mu2 = np.mean(hr2)

    cov = np.cov(hr1, hr2).flatten()

    skew = scipy.stats.skew(np.array([hr1, hr2]).T)
    kurt = scipy.stats.kurtosis(np.array([hr1, hr2]).T)

    if np.any(np.isnan(cov)):
        print("NaN in cov")

    return mu1, mu2, cov, skew, kurt

def hid_maps(seg, bins=30):
    counts = seg[:,1]
    low_counts = seg[:,2]
    high_counts = seg[:,3]
    hr2 = high_counts/low_counts
    hid_limits = compute_hrlimits(hr2, counts)

    h, xedges, yedges = np.histogram2d(hr2, counts, bins=bins,
                                       range=hid_limits)
    h = np.rot90(h)
    h = np.flipud(h)

    return xedges, yedges, h


def lcshape_features(seg, dt=1.0):

    times = seg[:,0]
    counts = seg[:,1]

    dt_small = times[1:]-times[:-1]
    dt_small = np.min(dt_small)

    nbins = np.round(dt/dt_small)

    bintimes, bincounts = rebin_psd(times, counts, nbins)

    return bincounts

def extract_lc(seg):
    times = seg[:,0]
    counts = seg[:,1]
    low_counts = seg[:,2]
    high_counts = seg[:,3]
    hr1 = low_counts/counts
    hr2 = high_counts/counts
    return [times, counts, hr1, hr2]


def psd_pca(seg, n_components=12):
    """
    Extract a PCA representation of the light curves
    """
    
    freq_all, ps_all, maxfreq_all = [], [], []
    for s in seg:
        times = s[:,0] 
        dt = times[1:] - times[:-1]
        dt = np.min(dt)

        counts = s[:,1]*dt
    
        ps = powerspectrum.PowerSpectrum(times, counts=counts, norm="rms")
        ps.freq = np.array(ps.freq)
        ps.ps = np.array(ps.ps)*ps.freq
     
        freq = np.array(ps.freq[1:])
        ps = ps.ps[1:]

        binfreq, binps, nsamples = logbin_periodogram(freq, ps)

        bkg = np.mean(binps[-50:])

        binps -= bkg
    
        fmax_ind = np.where(binfreq*binps == np.max(binfreq*binps))
        maxfreq = binfreq[fmax_ind[0]]
        freq_all.append(binfreq)
        ps_all.append(binps)
        maxfreq_all.append(maxfreq)
 
    freq_all = np.array(freq_all)
    ps_all = np.array(ps_all)
    maxfreq_all = np.hstack(maxfreq_all)

    ps_scaled = StandardScaler().fit_transform(ps_all)
    pc = PCA(n_components=n_components) 
    ps_pca = pc.fit(ps_scaled).transform(ps_scaled)
 
    return ps_pca

def make_features(seg, k=10, bins=30, lamb=None, n=1,
                  hr_summary=True, ps_summary=True, 
                  lc=True, hr=True, hrlimits=None, n_components=3):
    """
    Make features from a set of light curve segments, except for the linear filter!
    
    Parameters
    ----------
    seg : iterable
        list of all segments to be used
        
    bins : int
        bins used in a 2D histogram if hr_summary is False
        
    hr_summary : bool
        if True, summarize HRs in means and covariance matrix
        
    ps_summary : bool
        if True, summarize power spectrum in frequency of maximum 
        power and power spectral bands
        
    lc : bool
        if True, store light curves
        
    hr : bool
        if True, store hardness ratios
        
    hrlimits : iterable
        limits for the 2D histogram if hr_summary is False
    
    Returns
    -------
    fdict : dictionary 
        contains keywords "features", "lc" and "hr"


    """
    features = []
    if lc:
        lc_all = []
    if hr:
        hr_all = []


    ## MAKE FEATURES BASED ON THE AUTOREGRESSIVE MODEL

    ## first, extract the total counts out of each segment
    counts = np.array([s[:,1] for s in seg])

    ## next, transform the array such that *each light curve* is scaled
    ## to zero mean and unit variance
    ## We can do this for all light curves independently, because we're
    ## averaging *per light curve* and not *per time bin*
    # counts_scaled = StandardScaler().fit_transform(counts.T).T

    ## transform the counts into a weight vector
    rt = pe.RebinTimeseries(n=n, method="average")
    counts_binned = rt.fit_transform(counts)
                
    #print("extracting weights from AR model ...")
    # weights of the autoregressive model
    ww = linear_filter(counts_binned, k=k, lamb=lamb)
 
    pca = psd_pca(seg, n_components=n_components)
    for s in seg:

        features_temp = []

        ## time series summary features
        fmean, fmedian, fvar, fskew, fkurt = timeseries_features(s)
        features_temp.extend([fmean, fmedian, np.log(fvar), fskew, fkurt])

        if ps_summary:
            ## PSD summary features
            maxfreq, psd_a, psd_b, psd_c, psd_d, pc1, pc2 = psd_features(s, pcb)
            if len(maxfreq) == 0: 
                features_temp.extend([psd_a, psd_b, psd_c, psd_d, pc1, pc2])
            else: 
                features_temp.extend([np.log(maxfreq), np.log(psd_a), np.log(psd_b), np.log(psd_c), np.log(psd_d), np.log(pc1), np.log(pc2)])

        else:
            ## whole PSD
            binfreq, binps = total_psd(s, 24)
            features_temp.extend(binps[1:])


        if hr_summary:
            mu1, mu2, cov, skew, kurt = hr_fitting(s)
            features_temp.extend([mu1, mu2])
            features_temp.extend([np.log(cov[0]), cov[1], np.log(cov[3])])
            features_temp.extend(list(skew))
            features_temp.extend(list(kurt))

        else:
            xedges, yedges, h = hr_maps(s, bins=bins, hrlimits=hrlimits)
            features_temp.extend(h.flatten())

        features.append(features_temp)


        if lc or hr:
            lc_temp = extract_lc(s)
        if lc:
            lc_all.append([lc_temp[0], lc_temp[1]])
        if hr:
            hr_all.append([lc_temp[2], lc_temp[3]])

    features_all = np.hstack((np.array(features), np.array(ww)))
    features_all = np.hstack((np.array(features_all), np.array(pca)))

    fdict = {"features": features_all}
    if lc:
        #features.append(lc_all)
        fdict["lc"] = lc_all
    if hr:
        #features.append(hr_all)
        fdict["hr"] = hr_all
    return fdict


def check_nan(features, labels, hr=True, lc=True):
    """ 
    Check for NaN values in the features and throw out any 
    sample that contains NaNs.

    """
  
    inf_ind = []
    fnew, lnew, tnew = [], [], []

    nseg = features["nseg"]
    nsum = np.array([np.sum(nseg[:i])-1 for i in xrange(1, len(nseg)+1)])
    if lc:
        lcnew = []
    if hr:
        hrnew = []

    for i,f in enumerate(features["features"]):

        try:
            if any(np.isnan(f)):
                print("NaN in sample row %i"%i)
                aind = np.array(nsum).searchsorted(i)
                nseg[aind] -= 1
                continue
            elif any(np.isinf(f)):
                print("inf sample row %i"%i)
                aind = np.array(nsum).searchsorted(i)
                nseg[aind] -= 1
                continue
            else:
                fnew.append(f)
                lnew.append(labels[i])
                tnew.append(features["tstart"][i])
                if lc:
                    lcnew.append(features["lc"][i])
                if hr:
                    hrnew.append(features["hr"][i])
        except ValueError:
            print("f: " + str(f))
            print("type(f): " + str(type(f)))
            raise Exception("This is breaking! Boo!")
    features_new = {"features":np.array(fnew), "tstart":tnew, "nseg":nseg}
    if lc:
        features_new["lc"] = lcnew
    if hr:
        features_new["hr"] = hrnew
    return features_new, lnew


def make_all_features(d_all, k=10, lamb=0.1, n=1, n_components=1,
                      val=True, train_frac=0.6, validation_frac=0.2, test_frac = 0.2,
                      seg=True, seg_length=1024., overlap = 64., dt=0.125,
                      bins=30, navg=4, hr_summary=True, ps_summary=True, lc=True, hr=True,
                      save_features=True, froot="grs1915", seed=20160615):
    """
    Get out features for the entire data set from start to finish. 
    Stores the results to disk in separate files for the features, labels, light curves, 
    hardness ratios, start times and observation identifiers.
    """

    ## Set the seed to I will always pick out the same light curves.
    np.random.seed(seed)

    ## shuffle list of light curves
    indices = np.arange(len(d_all))
    np.random.shuffle(indices)

    n_lcs = len(d_all)

    ## let's pull out light curves for three data sets into different variables.
    d_all_train = [d_all[i] for i in indices[:int(train_frac*n_lcs)]]
    d_all_test = [d_all[i] for i in indices[int(train_frac*n_lcs):int((train_frac + test_frac)*n_lcs)]]
 
    seg_train, labels_train, nseg_train = extract_segments(d_all_train, seg_length=seg_length,
                                                           overlap=overlap, dt=dt)
    seg_test, labels_test, nseg_test = extract_segments(d_all_test, seg_length=seg_length,
                                                        overlap=overlap, dt=dt)


    states = [d[1] for d in d_all_train]
    st = pd.Series(states)
    print("The number of states in the training set is %i"%len(st.value_counts()))

    states = [d[1] for d in d_all_test]
    st = pd.Series(states)
    print("The number of states in the test set is %i"%len(st.value_counts()))

    tstart_train = np.array([s[0,0] for s in seg_train])
    tstart_test = np.array([s[0,0] for s in seg_test])


    if val:
        d_all_val = [d_all[i] for i in indices[int((train_frac + test_frac)*n_lcs):]]
        seg_val, labels_val, nseg_val = extract_segments(d_all_val, seg_length=seg_length,
                                                         overlap=overlap, dt=dt)
        tstart_val = np.array([s[0,0] for s in seg_val])

        states = [d[1] for d in d_all_val]
        st = pd.Series(states)
        print("The number of states in the validation set is %i"%len(st.value_counts()))

    ### hrlimits are derived from the data, in the GRS1915_DataVisualisation Notebook
    hrlimits = [[-2.5, 1.5], [-3.0, 2.0]]
 
    features_train = make_features(seg_train, k, bins, lamb, n, hr_summary, ps_summary, lc, hr, hrlimits, n_components)
    features_test = make_features(seg_test,k, bins, lamb, n,  hr_summary, ps_summary, lc, hr, hrlimits, n_components)

    features_train["tstart"] = tstart_train
    features_test["tstart"] = tstart_test

    features_train["nseg"] = nseg_train
    features_test["nseg"] = nseg_test

    ## check for NaN
    print("Checking for NaN in the training set ...")
    print("%i samples in training data set before checking for NaNs."%features_train["features"].shape[0])
    print("%i samples in test data set before checking for NaNs."%features_test["features"].shape[0])

    features_train_checked, labels_train_checked = check_nan(features_train, labels_train,
                                                             hr=hr, lc=lc)
    print("Checking for NaN in the test set ...")
    features_test_checked, labels_test_checked= check_nan(features_test, labels_test,
                                                          hr=hr, lc=lc)

    print("%i samples in training data set after checking for NaNs."%features_train_checked["features"].shape[0])
    print("%i samples in test data set before after for NaNs."%features_test_checked["features"].shape[0])

    labelled_features = {"train": [features_train_checked["features"], labels_train_checked],
                     "test": [features_test_checked["features"], labels_test_checked]}

    if val: 
        features_val = make_features(seg_val, k, bins, lamb, n, hr_summary, ps_summary, lc, hr, hrlimits, n_components)
        features_val["tstart"] = tstart_val
        features_val["nseg"] = nseg_val

        print("Checking for NaN in the validation set ...")
        features_val_checked, labels_val_checked = check_nan(features_val, labels_val, hr=hr, lc=lc)
        print("%i samples in validation data set after checking for NaNs."%features_val_checked["features"].shape[0])

        labelled_features["val"] =  [features_val_checked["features"], labels_val_checked],

    if save_features:
        np.savetxt(froot+"_%is_features_train.txt"%int(seg_length), features_train_checked["features"])
        np.savetxt(froot+"_%is_features_test.txt"%int(seg_length), features_test_checked["features"])
        np.savetxt(froot+"_%is_nseg_train.txt"%int(seg_length), features_train_checked["nseg"])
        np.savetxt(froot+"_%is_nseg_test.txt"%int(seg_length), features_test_checked["nseg"])

        np.savetxt(froot+"_%is_tstart_train.txt"%int(seg_length), features_train_checked["tstart"])
        np.savetxt(froot+"_%is_tstart_test.txt"%int(seg_length), features_test_checked["tstart"])

        ltrainfile = open(froot+"_%is_labels_train.txt"%int(seg_length), "w")
        for l in labels_train_checked:
            ltrainfile.write(str(l) + "\n")
        ltrainfile.close()

        ltestfile = open(froot+"_%is_labels_test.txt"%int(seg_length), "w")
        for l in labels_test_checked:
            ltestfile.write(str(l) + "\n")
        ltestfile.close()


        if val:
            np.savetxt(froot+"_%is_features_val.txt"%int(seg_length), features_val_checked["features"])
            np.savetxt(froot+"_%is_tstart_val.txt"%int(seg_length), features_val_checked["tstart"])
            np.savetxt(froot+"_%is_nseg_val.txt"%int(seg_length), features_val_checked["nseg"])

            lvalfile = open(froot+"_%is_labels_val.txt"%int(seg_length), "w")
            for l in labels_val_checked:
                lvalfile.write(str(l) + "\n")
            lvalfile.close()


        if lc:
            lc_all = {"train":features_train_checked["lc"], "test":features_test_checked["lc"]}
            if val:
                lc_all["val"] = features_val_checked["lc"]

            f = open(froot+"_%is_lc_all.dat"%int(seg_length), "w")
            pickle.dump(lc_all, f, -1)
            f.close()

        if hr:
            hr_all = {"train":features_train_checked["hr"], "test":features_test_checked["hr"]}
            if val:
                hr_all["val"] = features_val_checked["hr"]

            f = open(froot+"_%is_hr_all.dat"%int(seg_length), "w")
            pickle.dump(hr_all, f, -1)
            f.close()

    return labelled_features



######################################################################################################################
#### EXTRACT FEATURE FILES ###########################################################################################
######################################################################################################################

def extract_all(d_all, seg_length_all=[256., 1024.], overlap=128.,
                val=True, train_frac=0.5, validation_frac = 0.25, test_frac = 0.25,
                k = 10, lamb=0.1, n=1, n_components=1, seed=20160615,
                datadir="./"):

    if np.size(overlap) != np.size(seg_length_all):
        overlap = [overlap for i in xrange(len(seg_length_all))]
 

    for ov, sl in zip(overlap, seg_length_all):
        lf = make_all_features(d_all, k, lamb, n, n_components, val, train_frac, validation_frac, test_frac,
                  seg=True, seg_length=sl, overlap=ov,
                  hr_summary=True, ps_summary=True, lc=True, hr=True,
                  save_features=True, froot=datadir+"grs1915", seed=seed)

    return
