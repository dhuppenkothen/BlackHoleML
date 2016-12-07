__author__ = 'danielahuppenkothen'

import glob
import numpy as np
from astropy.io import fits
import cPickle as pickle

import convert_belloni


def rebin_lightcurve(times, counts, n=10, type='average'):

    nbins = int(len(times)/n)
    dt = times[1] - times[0]
    T = times[-1] - times[0] + dt
    bin_dt = dt*n
    bintimes = np.arange(nbins)*bin_dt + bin_dt/2.0 + times[0]

    nbins_new = int(len(counts)/n)
    counts_new = counts[:nbins_new*n]
    bincounts = np.reshape(np.array(counts_new), (nbins_new, n))
    bincounts = np.sum(bincounts, axis=1)
    if type in ["average", "mean"]:
        bincounts = bincounts/np.float(n)
    else:
        bincounts = bincounts

    #bincounts = np.array([np.sum(counts[i*n:i*n+n]) for i in range(nbins)])/np.float(n)
    #print("len(bintimes): " + str(len(bintimes)))
    #print("len(bincounts: " + str(len(bincounts)))
    if len(bintimes) < len(bincounts):
        bincounts = bincounts[:len(bintimes)]

    return bintimes, bincounts



################################################################################################################
#### FIRST PART: COMBINE RXTE LIGHT CURVES INTO ASCII FILES THAT I CAN USE #####################################
################################################################################################################

def find_todo(obsids, datadir="./"):
    """
    Find all light curves that do not have a *combined.dat file and still
    need one!

    :param obsids: ObsIDs of the observations to be combined
    :param datadir: Directory that contains the data
    :return:
    """
    combined_files = glob.glob(datadir+"*combined.dat")
    combined = [f.split("_")[1] for f in combined_files]
    todo = set(obsids).difference(combined)
    return list(todo)

def equalize_resolution(times, counts, nbins):
    """
    Rebin a light curve to equalize the resolution of that light curve with all
    the other light curves.
    :param times: numpy array with time stamps
    :param counts: numpy array with (Poisson) counts
    :param nbins: total number of bins in the new resolution light curve
    :return: binned light curve (times_new, counts_new) the new light curve has fewer bins than the old one,
    else return the original light curve
    """
    if nbins < len(times):
        n = int(len(times)/nbins)
        times_new, counts_new = rebin_lightcurve(times, counts, n=n, type="average")
        return times_new[:nbins], counts_new[:nbins]
    else:
        return times, counts


def combine_lightcurves(obsids, datadir="./"):
    """
    Run through all ObsIDs and combine the light curves in different energy ranges
    into a single ascii file with low, mid and high band counts
    :param obsids: list of ObsIDs for the observations in the directory datadir
    :param datadir: directory with the data
    :return: saves file to disk
    """

    hdulist = fits.open(datadir+"GRS1915+105.fits")
    h = hdulist[1]
    d = h.data
    all_obs = d["OBSID"]
    t_start = d["T_START_OBS"]
    t_mid = d["TIME"]
    t_end = d["T_STOP_OBS"]
    exposure = d["EXPOSURE"]

    for i,o in enumerate(obsids):
        fobs = glob.glob(datadir+"*%s*.fits"%o)
        emin_all, emax_all = [], []
        for f in fobs:
            fsplit = f.split("_")
            obsid = fsplit[1]
            ebands = fsplit[3].split("-")
            emin_all.append(np.float(ebands[0]))
            emax_all.append(np.float(ebands[1][:-3]))
            time_res = fsplit[4].split("div")
            tres = np.float(time_res[0])/np.float(time_res[1][:-1])

        if len(emin_all) < 3 or len(emax_all) < 3:
            continue

        emin_all = np.array(emin_all)
        emax_all = np.array(emax_all)

        ### below, you can find totally stupid code, with the energy ranges for the
        ### RXTE data of GRS 1915+105 as made by Lucy, because the energy ranges are not
        ### *quite* the same for the different observations (since the bins shifted with time,
        ### so I need to some complicated statements to figure out which is low, mid and high band.

        total_band_ind = np.where((emin_all < 6.2) & (emax_all > 10.))[0]
        if len(total_band_ind) > 1:
            total_band_ind = total_band_ind[0]

        low_band_ind = np.where((emin_all < 6.2) & (emax_all < 10.))[0]

        mid_band_ind = np.where((emin_all > 6.2) & (emin_all < 10.0) & (emax_all >10.) & (emax_all < 25.0))[0]
        if len(mid_band_ind) > 1:
            mid_band_ind = mid_band_ind[0]

        high_band_ind1 = np.where((emin_all > 10.) & (emax_all > 12.) & (emax_all < 20.0))[0]
        if len(high_band_ind1) == 0:
            high_band_ind1 = None

        high_band_ind2 = np.where((emin_all > 13.0) & (emax_all > 50.))[0]
        if len(high_band_ind2) == 0:
            print("No high_band_ind2")
            continue

        mlf_ind = np.where(obsid == d["OBSID"])[0]
        print("index of ObsID file: " + str(mlf_ind))
        if len(mlf_ind) == 0:
            print("don't know where the observation is. Continuing ...")
            continue

        start_time = t_start[mlf_ind]*3600.0*24.0

        total_data = fits.open(fobs[total_band_ind])
        total_times, total_counts = total_data[1].data.field(0), total_data[1].data.field(1)
        gti_start = total_data[2].data.field(0)[0]

        total_times += start_time + gti_start

        dt_total = np.min(total_times[1:] - total_times[:-1])
        nbins_total = len(total_times)

        low_data = fits.open(fobs[low_band_ind])
        low_times, low_counts = low_data[1].data.field(0), low_data[1].data.field(1)
        low_times += start_time + gti_start

        dt_low = np.min(low_times[1:] - low_times[:-1])
        nbins_low= len(low_times)


        mid_data = fits.open(fobs[mid_band_ind])
        mid_times, mid_counts = mid_data[1].data.field(0), mid_data[1].data.field(1)
        mid_times += start_time + gti_start

        dt_mid = np.min(mid_times[1:] - mid_times[:-1])
        nbins_mid = len(mid_times)

        high_data = fits.open(fobs[high_band_ind2])
        high_times, high_counts = high_data[1].data.field(0), high_data[1].data.field(1)
        high_times += start_time + gti_start

        dt_high = np.min(high_times[1:] - high_times[:-1])
        nbins_high = len(high_times)

        if not nbins_total == nbins_low == nbins_mid == nbins_high:
            nbins = np.min([nbins_total, nbins_low, nbins_mid, nbins_high])
            total_times, total_counts = equalize_resolution(total_times, total_counts, nbins)
            low_times, low_counts = equalize_resolution(low_times, low_counts, nbins)
            mid_times, mid_counts = equalize_resolution(mid_times, mid_counts, nbins)
            high_times, high_counts = equalize_resolution(high_times, high_counts, nbins)

        data = np.transpose(np.array([total_times, total_counts, low_counts, mid_counts, high_counts]))

        np.savetxt(datadir+"LC_%s_combined.dat"%o,data,
                    header = "Times \t total counts \t low band counts \t mid band counts \t high band counts\n")

    return




################################################################################################################
#### SECOND PART: SPLIT LIGHT CURVES AND SAVE AS A WHOLE OBJECT TO BE USED #####################################
################################################################################################################

def bin_lightcurve(dtemp, nbins):
    tbinned_times, tbinned_counts = rebin_lightcurve(dtemp[:,0], dtemp[:,1], n=nbins, type="average")
    lbinned_times, lbinned_counts = rebin_lightcurve(dtemp[:,0], dtemp[:,2], n=nbins, type="average")
    mbinned_times, mbinned_counts = rebin_lightcurve(dtemp[:,0], dtemp[:,3], n=nbins, type="average")
    hbinned_times, hbinned_counts = rebin_lightcurve(dtemp[:,0], dtemp[:,4], n=nbins, type="average")

    dshort = np.transpose(np.array([tbinned_times, tbinned_counts, lbinned_counts, mbinned_counts, hbinned_counts]))
    return dshort


def extract_obsmode_data(files, bin_data=True, bin_res=0.125, label_only=False, labels="clean"):
    """
    Extract Observing Mode data from a list of text files with columns
    #times \t total count rate \t low energy count rate \t high energy count rate

    if bin_data=True, the data will be rebinned to the resolution specified in bin_res.
    If label_only is True, only data with (manually determined) labels will be extracted;
    either from the whole set of observations used in Belloni+ 2000 (labels="all") or from
    the cleaned set with only those observations where the state does not change during the
    observation (labels="clean")

    """

    if labels == "clean":
        belloni_turned = convert_belloni.convert_belloni_clean()
    else:
        belloni_states = convert_belloni.main()
        belloni_turned = convert_belloni.turn_states(belloni_states)


    d_all = []
    for f in files:
        fstring = f.split("_")[1]
        if fstring in belloni_turned:
            state = belloni_turned[fstring]
        else:
            state = None
            if label_only:
                continue

        d = np.loadtxt(f)
        dt_data = d[1:,0]-d[:-1,0]

        dt_min = np.min(dt_data)

        ## compute nbins, if nbins is <=1, don't bin
        ## because target resolution is smaller than
        ## native resolution, and we don't resample.
        nbins = int(bin_res/dt_min)
        if nbins <= 1:
            print("Target resolution smaller than native time resolution. Not binning!")
            bin_data=False

        ### split data with breaks
        breaks = np.where(dt_data > 0.008)[0]
        if len(breaks) == 0:
            dtemp = d
            if bin_data:
                dshort = bin_lightcurve(dtemp, nbins)
            else:
                dshort = dtemp
            d_all.append([dshort, state])
        else:
            for i,b in enumerate(breaks):
                if i == 0:
                    if b == 0:
                        continue
                    else:
                        dtemp = d[:b]
                        if bin_data:
                            dshort = bin_lightcurve(dtemp, nbins)
                        else:
                            dshort = dtemp

                else:
                    dtemp = d[breaks[i-1]+1:b]
                    if bin_data:
                        dshort = bin_lightcurve(dtemp, nbins)
                    else:
                        dshort = dtemp

                d_all.append([dshort, state])

            ## last segment
            dtemp = d[b+1:]
            if bin_data:
                dshort = bin_lightcurve(dtemp, nbins)
            else:
                dshort = dtemp

            d_all.append([dshort, state])

    return d_all


def remove_zeros(d):
    """
    Remove all zero-valued counts from a segment d.
    :param d:
    :return:
    """
    data = d[0]
    labels = d[1]

    max_all = []
    for m in [np.where(d[0][:,1] <= 0.0)[0], np.where(d[0][:,2] <= 0.0)[0],  np.where(d[0][:,3] <= 0.0)[0]]:
        if len(m) > 0:
            max_all.extend(m)
    if len(max_all) == 0:
        return[data, labels]
    else:
        max_zero = np.max(max_all)
        data = data[max_zero:]
        return [data, labels]


def remove_nans(d_all):


    d_all_new = []
    for i,d in enumerate(d_all):
        zero_counts = np.where(d[0][:,1] <= 0.0)[0]
        zero_hr1 =  np.where(d[0][:,2] <= 0.0)[0]
        zero_hr2 =  np.where(d[0][:,3] <= 0.0)[0]
        if len(zero_counts) > 0 or len(zero_hr1) > 0 or len(zero_hr2) > 0:
            print("Found some zeros in light curve %i. Removing ..."%i)
            d_new = remove_zeros(d)
            d_all_new.append(d_new)

        else:
            d_all_new.append(d)

    return d_all_new

def add_total_countrate(d_all):
    """
    This function takes the "total" column and 
    the "high band" column and adds them together 
    since currently the "total" column only goes to
    13 keV.

    """
    d_all_new = []
    for data in d_all:
       lcs = data[0]
       lcs[:,1] = lcs[:,1] + lcs[:,-1]
       d_all_new.append([lcs, data[1]])
   
    return d_all_new



def extract_all_segments(clean=True, datadir="./", bin_data=True, bin_res=0.125):
    """
    Do the entire data extraction and save in a pickle file.
    :param clean:
    :param datadir:
    :return:
    """


    ## make a list of all ObsIDs
    files = glob.glob(datadir+"LC*.fits")
    obsids = [f.split("_")[1] for f in files]
    obsids = set(obsids)
    print("Total number of ObsIDs: " + str(len(obsids)))

    ### combine low-,mid- and high-energy light curves and save in ascii files
    combine_lightcurves(obsids, datadir=datadir)

    ### find all newly combined data files and extract the segments
    files = glob.glob(datadir+"*combined.dat")
    d_all = extract_obsmode_data(files, bin_data=True, bin_res=bin_res, label_only=clean, labels="clean")

    d_all_new = remove_nans(d_all)
    d_all_new = add_total_countrate(d_all_new)
 
    if clean:
        outfile = "grs1915_clean_label_%ims.dat"%int(bin_res*1000.0)
    else:
        outfile = "grs1915_all_%ims.dat"%int(bin_res*1000.0)

    f = open(datadir+outfile, "w")
    pickle.dump(d_all_new, f, -1)
    f.close()

    return
