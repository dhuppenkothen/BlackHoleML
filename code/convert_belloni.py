import numpy as np
import cPickle as pickle

def convert_obsids(states):
    """
    Convert the ObsIDs from the Belloni paper, which contain letters,
    to actual RXTE OBsIDs.

    Parameters
    ----------
    states : iterable
        A list of ObsIDs

    Returns
    -------
    states_new : iterable
        A list with the full ObsIDs without abbreviations
    """

    i = "10408-01"
    j = "20187-02"
    k = "20402-01"

    states_new = []
    for s in states:
        if s[0] == "I":
            states_new.append(i+s[1:])
        elif s[0] == "J":
            states_new.append(j+s[1:])
        else:
            states_new.append(k+s[1:])
    return states_new



def convert_belloni_clean(turned=True):
    """
    I have two files: one with the full set of classifications, and 
    one with *only* the classifications from Belloni et al (2000) and 
    Klein-Wolt et al (2002) where the full observation was covered by a 
    *single* state, i.e. this excludes observations that were annotated, 
    but where the source's state changed throughout the observation.
    This function returns a "clean" sample so that I can classify without 
    having to worry about mis-attributions because the source switched 
    state.

    Note: the file has *one* line per GRS 1915+105 source state, with a 
    list of observations in that state following the state descriptor.
    
    Parameters
    ----------
    turned : bool
       Turn dictionary from {"state":[ObsIDs]} to {"ObsID":state} ?

    Returns
    -------
    belloni_clean : dict
        A dictionary with the ObsIDs and corresponding states.
    """
    ## cleaned version; without observations that have light curves
    ## with more than one class
    file = open("1915Belloniclass_updated.dat")
    lines = file.readlines()
    header = lines[0].split()
    belloni_clean = {}
    for h,l in zip(header, lines[1:]):
        belloni_clean[h] = l.split()
    
    if turned:
        belloni_clean_turned = turn_states(belloni_clean)
        return belloni_clean_turned
    else:
        return belloni_clean
    
## turn around conversion (just in case):
def turn_states(states, remove_dashes = False):
    """
    Turn the state dictionary from the form {"state":[ObsIDs]} 
    to the form {"ObsID": state}
    """
    turned_states = {}
    for k,lis in states.iteritems():
        for l in lis:
            if remove_dashes:
                turned_states[l.translate(None, "-")] = k
            else:
                turned_states[l] = k
    return turned_states



def main():
    alpha_state = ["J-01-00","J-01-01","J-02-00","K-22-00","K-23-00", "K-24-01","K-26-00","K-27-00",
         "K-28-00", "K-30-01", "K-30-02"]
    beta_state = ["I-10-00","I-21-00","I-21-01","K-43-00","K-43-02", "K-44-00","K-45-00", "K-45-03", 
        "K-46-00","K-52-01","K-52-02","K-53-00","K-59-00"]
    gamma_state = ["I-07-00","K-37-00","K-37-02","K-38-00", "K-39-00","K-39-02","K-40-00","K-55-00",
            "K-56-00", "K-57-00"]
    delta_state = ["I-13-00","I-14-00","I-14-01","I-14-02","I-14-03", "I-14-04","I-14-05","I-14-07",
         "I-14-08","I-14-09","I-17-00", "I-17-03","I-18-00","I-18-01","I-18-04","K-41-00",
         "K-41-01","K-41-02","K-41-03","K-42-00","K-53-02", "K-54-00"]
    theta_state = ["I-15-00","I-15-01","I-15-02","I-15-03","I-15-04","I-15-05", "I-16-00","I-16-01",
         "I-16-02","I-16-03","I-16-04", "I-21-01","K-45-02"]
    kappa_state = ["K-33-00","K-35-00"]

    lambda_state = ["I-37-00", "I-38-00","K-36-00","K-36-01","K-37-01"]
    mu_state = ["I-08-00","I-34-00","I-35-00","I-36-00", "K-43-00","K-43-01","K-45-01", "K-53-01",
            "K-53-02","K-59-00"]
    nu_state = ["I-37-00","I-40-00","I-41-00", "I-44-00","K-01-00","K-02-02"]
    rho_state = ["K-03-00","K-27-02","K-30-00","K-31-00", "K-31-01","K-31-02","K-32-00","K-32-01",
             "K-34-00", "K-34-01"]
    phi_state = ["I-09-00","I-11-00","I-12-00","I-13-00", "I-17-01","I-17-02","I-18-00","I-19-00",
             "I-19-01","I-19-02", "I-20-00","I-20-01"]


    chi1_state = ["I-22-00","I-22-01","I-22-02","I-23-00","I-24-00","I-25-00", "I-27-00","I-28-00",
              "I-29-00","I-30-00","I-31-00"]

    chi2_state = ["K-04-00","K-05-00","K-07-00","K-08-00","K-08-01","K-09-00","K-10-00", "K-11-00",
              "K-12-00","K-13-00","K-14-00","K-15-00", "K-16-00","K-17-00","K-18-00","K-19-00",
              "K-20-00","K-21-01"]

    chi3_state = ["K-49-00","K-49-01","K-50-00","K-50-01","K-51-00","K-52-00"]

    chi4_state = ["I-31-00","I-32-00","I-33-00","I-38-00","I-42-00","I-43-00","I-45-00", "J-02-00",
              "K-01-00", "K-02-02","K-24-00","K-25-00","K-26-00", "K-26-01","K-26-02","K-27-01",
              "K-27-03","K-29-00", "K-30-01","K-48-00"]


    alpha_new = convert_obsids(alpha_state)
    beta_new = convert_obsids(beta_state)
    gamma_new = convert_obsids(gamma_state)
    delta_new = convert_obsids(delta_state)
    theta_new = convert_obsids(theta_state)
    kappa_new = convert_obsids(kappa_state)
    lambda_new = convert_obsids(lambda_state)
    mu_new = convert_obsids(mu_state)
    nu_new = convert_obsids(nu_state)
    rho_new = convert_obsids(rho_state)
    phi_new = convert_obsids(phi_state)
    chi1_new = convert_obsids(chi1_state)
    chi2_new = convert_obsids(chi2_state)
    chi3_new = convert_obsids(chi3_state)
    chi4_new = convert_obsids(chi4_state)

    states_dict = {"alpha":alpha_new, "beta":beta_new, "gamma":gamma_new, "delta":delta_new, "theta":theta_new,
                "kappa":kappa_new, "lambda":lambda_new, "mu":mu_new, "nu":nu_new, "rho":rho_new, "phi":phi_new,
                "chi1":chi1_new, "chi2":chi2_new, "chi3":chi3_new, "chi4":chi4_new}

    f = open("grs1915_belloni_states.dat", "w")
    pickle.dump(states_dict,f)
    f.close()

    return states_dict

if __name__ == "__main__":
    states_belloni = main()


