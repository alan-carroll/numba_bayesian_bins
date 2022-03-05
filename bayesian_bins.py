from llvmlite import binding
binding.set_option('SVML', '-vector-library=SVML')

import numpy as np
import scipy.special as ss
import numba as nb
import ctypes

# Implement scipy-cython betaln as JIT
betaln_addr = nb.extending.get_cython_function_address(
    "scipy.special.cython_special", "betaln")
betaln_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, 
                                   ctypes.c_double)
betaln_fn = betaln_functype(betaln_addr)

@nb.njit('float64(float64, float64)')
def nb_betaln(a, b):
    return betaln_fn(a, b)

@nb.vectorize('float64(float64, float64)', target='cpu')
def vec_betaln(a, b):
    return betaln_fn(a, b)


# Implement scipy-cython betainc as JIT
betainc_addr = nb.extending.get_cython_function_address(
    "scipy.special.cython_special", "betainc")
betainc_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double,
                                    ctypes.c_double, ctypes.c_double)
betainc_fn = betainc_functype(betainc_addr)

@nb.jit('float64(float64, float64, float64)')
def nb_betainc(a, b, c):
    return betainc_fn(a, b, c)


# Implement scipy-cython gammaln as JIT
gammaln_addr = nb.extending.get_cython_function_address(
    "scipy.special.cython_special", "gammaln")
gammaln_functype = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double)
gammaln_fn = gammaln_functype(gammaln_addr)

@nb.jit('float64(float64)')
def nb_gammaln(a):
    return gammaln_fn(a)


# Define JIT-compiled convenience functions 
@nb.jit
def nb_n_choose_k(n, k):
    return nb_gammaln(n + 1) - nb_gammaln(k + 1) - nb_gammaln(n - k + 1)

@nb.jit(nopython=True, nogil=True, parallel=True)
def nb_calc_m_priors(max_m, max_t, sigma, gamma):
    m_priors = np.zeros(max_m + 1)
    m_pub = 1
    bf = np.log(nb_betainc(sigma, gamma, m_pub)) + nb_gammaln(sigma) + \
        nb_gammaln(gamma) - nb_gammaln(sigma + gamma)
    
    for idx in nb.prange(max_m + 1):
        m_priors[idx] = -(idx + 1)*bf - nb_n_choose_k(max_t, idx)
        
    return m_priors


# Two versions of logsumexp: normal, and max-val protection
# Normal is faster, but max prevents inf errors
#    Normal: ~60 us -- Max: ~100 us
@nb.njit
def nb_logsumexp(arr):
    return np.log(np.sum(np.exp(arr)))

@nb.njit(fastmath=True, parallel=False)
def nb_max_logsumexp(arr):
    max_val = np.max(arr)
    return np.log(np.sum(np.exp(arr - max_val))) + max_val


# Big E
@nb.jit
def nb_big_e_k_func(sub_e, m, event_counts, num_sweeps, sigma, gamma, length, 
                    max_t):
    """
    Big E
    Define k-loop func
    Same as SDF -- each iteration of k depends on sub_e, but not each other
    Copying sub_e ahead of calculations to be explicit and allocate space for
    the new array values
    """
    
    new_sub_e = sub_e.copy()

    for k_idx in np.arange(length, max_t):

        if k_idx == m:
            new_sub_e[k_idx] = 0
        else:
            # All other cases proceed to their r-funcs
            r_idx = np.arange(m, k_idx)
            events = event_counts[k_idx] - event_counts[r_idx]
            gaps = num_sweeps*(k_idx - r_idx) - events
            
            # Calculate 'running_val'
            # Using vec betaln so it can have arrays instead of writing
            # another loop, but this may be inefficient or slower than 
            # explicitly using JIT + for loop
            running_val = sub_e[r_idx] + vec_betaln((events + sigma), 
                                                    (gaps + gamma))

            new_sub_e[k_idx] = nb_max_logsumexp(running_val)

    return new_sub_e



@nb.jit
def calc_sub_e_parallel(event_counts, num_sweeps, max_t, sigma, gamma):
    """
    JITing initial big E sub_e calculation just to be explicit about
    parallelization. I don't know if numba would be upset about it or not,
    but this keeps it simple. The rest of the m-loop and big_e can't be 
    parallelized, but this simple for loop can, so here you go
    """

    sub_e = np.zeros(max_t)
    
    for k_end in nb.prange(max_t):
        events = event_counts[k_end]
        gaps = num_sweeps*(k_end+1) - events
        
        sub_e[k_end] = nb_betaln((events + sigma), (gaps + gamma))
                    
    return sub_e


# Big E main function call
@nb.jit
def nb_calc_big_e(m_priors, event_counts, num_sweeps, max_m, max_t, sigma,
                  gamma):

    sub_e = calc_sub_e_parallel(event_counts, num_sweeps, max_t, sigma, gamma)
    big_e = np.zeros(max_m + 1)
    big_e[0] = sub_e[-1] + m_priors[0]
    
    # Can't parallelize this loop. Each iteration depends on the last
    for m in range(max_m):
        if m == (max_m - 1):
            length = max_t - 1
        else:
            length = m
            
        # Now k-loop!
        sub_e = nb_big_e_k_func(sub_e, m, event_counts, num_sweeps, sigma, 
                                gamma, length, max_t)

        # Finish m-iteration with big_e update
        big_e[m+1] = sub_e[-1] + m_priors[m+1]
        
    return big_e



# Latency
@nb.njit(nogil=True, fastmath=True)
def nb_calc_latency(big_e, m_priors, event_counts, num_sweeps, lat_start, 
                    lat_end, max_m, max_t, sigma, gamma, l_bound, u_bound, 
                    signal, lat_idx):
    
    lat_sub_e = np.zeros(max_t)
    running_val = np.zeros(max_t)
    
    for k_end in nb.prange(lat_idx):
        events = event_counts[k_end]
        gaps = num_sweeps*(k_end+1) - events
        lat_sub_e[k_end] = vec_betaln((events + sigma), (gaps + gamma)) + \
            np.log(nb_betainc((events + sigma), (gaps + gamma), signal))
    
    lat_big_e = m_priors.copy()
    lat_big_e[0] = lat_sub_e[max_t-1] + m_priors[0]

    for m in range(max_m):
        if m == (max_m - 1):
            length = max_t - 1
        else:
            length = m
        
        for k_idx in np.arange(max_t-1, length-1, -1):
            
            lat_sub_e[k_idx] = 0
            
            if k_idx <= m:
                continue
            
            for r_idx in np.arange(m, k_idx):
                events = event_counts[k_idx] - event_counts[r_idx]
                gaps = num_sweeps*(k_idx - r_idx) - events
                
                if r_idx >= lat_idx:
                    if lat_sub_e[r_idx] == 0:
                        continue
                    else:
                        running_val[r_idx] = lat_sub_e[r_idx] + vec_betaln(
                            (events + sigma), (gaps + gamma))
                elif k_idx < lat_idx:
                    if lat_sub_e[r_idx] == 0:
                        continue
                    else:
                        running_val[r_idx] =  lat_sub_e[r_idx] + vec_betaln(
                            (events + sigma), (gaps + gamma)) + np.log(
                                nb_betainc(events+sigma, gaps+gamma, signal)) 
                elif ((r_idx+1)==lat_idx):
                    if lat_sub_e[r_idx] == 0:
                        continue
                    else:
                        running_val[r_idx] =  lat_sub_e[r_idx] + vec_betaln(
                            (events + sigma), (gaps + gamma)) + np.log(1 - \
                                nb_betainc(events+sigma, gaps+gamma, signal))
                else:
                    continue
            
            if k_idx >= lat_idx:
                if m == 0:
                    r_val_start = lat_idx - 1
                    r_val_end = lat_idx
                elif m < lat_idx:
                    r_val_start = lat_idx - 1
                    r_val_end = k_idx
                else:
                    r_val_start = m
                    r_val_end = k_idx
            else:
                r_val_start = m
                r_val_end = k_idx
            
            lat_sub_e[k_idx] = nb_max_logsumexp(
                running_val[r_val_start:r_val_end])
        
        lat_big_e[m+1] = lat_sub_e[max_t-1] + m_priors[m+1]
        
    # That's it! Finish with a couple computations!
    lat_sum = nb_max_logsumexp(lat_big_e[l_bound:(u_bound+1)])
    
    # Get mEV sum using *original* big_e values (not our new lat_big_e)
    mev_sum = nb_max_logsumexp(big_e[l_bound:(u_bound+1)])
    
    return np.exp(lat_sum - mev_sum)


@nb.njit(nogil=True, parallel=True)
def nb_para_calc_lats(big_e, m_priors, event_counts, num_sweeps, lat_start,
                      lat_end, max_m, max_t, sigma, gamma, l_bound, u_bound,
                      signal):
    results = np.empty(max_t)
    for lat_idx in nb.prange(lat_start, lat_end):
        results[lat_idx] = nb_calc_latency(big_e, m_priors, event_counts, 
                                           num_sweeps, lat_start, lat_end, 
                                           max_m, max_t, sigma, gamma, l_bound, 
                                           u_bound, signal, lat_idx)
        
    return results



# SDF
@nb.vectorize(target='cpu')
def nb_cython_r_calc(sub_e_val, r_idx, sdf_idx, k, events, gaps, sigma, gamma):
    """
    Define r-loop func.
    Currently this is a 'vectorized u-func', in order to avoid using for loops
    However, it seems like straight-up JIT + loops tends to be faster than the
    vectorized versions of things. Other optimizations might change this, but 
    JIT repeatedly seems to just do a better job than everything else.
    Leaving it as written now, but might change to JIT + loop later
    """
    
    if sub_e_val == 0:
        return 0

    if r_idx < sdf_idx <= k:
        return sub_e_val + vec_betaln((events + sigma), (gaps + gamma)) + \
            np.log(sigma + events) - np.log(sigma + gamma + events + gaps)
    
    else:
        return sub_e_val + vec_betaln((events + sigma), (gaps + gamma))


@nb.njit(nogil=True, fastmath=True)
def nb_k_func(sdf_sub_e, sdf_idx, sub_m, event_counts, num_sweeps, sigma, 
              gamma, length, max_t):
    """
    Define k-loop func.
    Literally just run the calc r function for each value of k in t_k_idx now.
    Using fastest implementation from numba jit tests. 
    Parallel + prange won by ~4x improvement on jit alone.
    """
    
    new_sdf_sub_e = sdf_sub_e.copy()
    
    for k_idx in nb.prange(length, max_t):

        if k_idx == sub_m:
            new_sdf_sub_e[k_idx] = 0
        else:
            # All other cases proceed to their r-funcs
            r_idx = np.arange(sub_m, k_idx)
            events = event_counts[k_idx] - event_counts[r_idx]
            gaps = num_sweeps*(k_idx - r_idx) - events
            
            # Vectorized r-loop
            running_val = nb_cython_r_calc(sdf_sub_e[r_idx], r_idx, sdf_idx, 
                                           k_idx, events, gaps, sigma, gamma)
            
            # logsumexp running_val
            running_val_max = np.max(running_val)
            new_sdf_sub_e[k_idx] = np.log(np.sum(np.exp(
                running_val - running_val_max))) + running_val_max
            
    return new_sdf_sub_e


@nb.njit(nogil=True, fastmath=True)
def calc_sdf_sub_e(sdf_idx, event_counts, num_sweeps, max_t, sigma, gamma):
    """
    SDF sub_E initial calculation JIT. Only called once per SDF, but there are
    a lot of SDF calls, so little differences add up.
    This parallel version beat the non-parallel JITs by ~25-30 us
    """
    sdf_sub_e = np.zeros(max_t)
    
    for k_end in nb.prange(max_t):
        events = event_counts[k_end]
        gaps = num_sweeps*(k_end+1) - events
        
        if sdf_idx <= k_end:
            sdf_sub_e[k_end] = vec_betaln((events + sigma), (gaps + gamma)) + \
                np.log(sigma + events) - np.log(sigma + gamma + events + gaps)
        else:
            sdf_sub_e[k_end] = vec_betaln((events + sigma), (gaps + gamma))
                    
    return sdf_sub_e


# Defining SDF main function call for each SDF index.
@nb.njit
def nb_calc_sdf(big_e, m_priors, event_counts, num_sweeps, max_m, max_t, sigma,
                gamma, l_bound, u_bound, sdf_idx):
    sdf_sub_e = calc_sdf_sub_e(sdf_idx, event_counts, num_sweeps, max_t, sigma,
                               gamma)
    sdf_big_e = np.zeros(max_m + 1)
    sdf_big_e[0] = sdf_sub_e[-1] + m_priors[0]
    
    # Can't parallelize this loop. Each iteration depends on the last.
    for sub_m in range(max_m):
        if sub_m == (max_m - 1):
            length = max_t - 1
        else:
            length = sub_m
            
        # Now k-loop!
        # The k-func copies the original sdf_sub_e and only updates the indices
        # it is supposed to (depending on the value of k within the 'k-loop'). 
        # The returned 'new' sdf_sub_e should therefore still have the correct
        # sizing.
        sdf_sub_e = nb_k_func(sdf_sub_e, sdf_idx, sub_m, event_counts, 
                              num_sweeps, sigma, gamma, length, max_t)
        sdf_big_e[sub_m+1] = sdf_sub_e[-1] + m_priors[sub_m+1]
        
    sdf_big_e_val_max = np.max(sdf_big_e[l_bound:(u_bound+1)])
    sdf_sum = np.log(np.sum(np.exp(sdf_big_e[l_bound:(u_bound+1)] - \
        sdf_big_e_val_max))) + sdf_big_e_val_max
    
    # Get mEV sum using *original* big_e values (not our new sdf_big_e)
    mev_val_max = np.max(big_e[l_bound:(u_bound+1)])
    mev_sum = np.log(np.sum(np.exp(big_e[l_bound:(u_bound+1)] - mev_val_max))
                     ) + mev_val_max
    
    return np.exp(sdf_sum - mev_sum)


@nb.njit(nogil=True, parallel=True)
def get_sdf(t_big_e, t_m_priors, t_event_counts, t_num_sweeps, t_max_m, 
            t_max_t, t_sigma, t_gamma, t_l_bound, t_u_bound):
    """
    Simple JIT for the for loop involved in calculating an entire SDF curve
    Don't know if this is really faster or not yet, but it doesn't hurt and it
    has the potential to make the code parallelizable.
    """
    t_sdf = np.zeros(t_max_t)
    for sdf_idx in nb.prange(t_max_t):
        t_sdf[sdf_idx] = nb_calc_sdf(t_big_e, t_m_priors, t_event_counts,
                                     t_num_sweeps, t_max_m, t_max_t, t_sigma,
                                     t_gamma, t_l_bound, t_u_bound, sdf_idx)

    return t_sdf
