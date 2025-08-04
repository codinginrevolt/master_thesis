def find_XV_subop(self, p_target, Omega=None, bounds=None, i_range=None,
                  nu=None, max_iterations=200, num_samples=1000,
                  batch_size=512, sampling_alg='ess'):

    """
    Find each constraint.Xv so that min_x P_c(x) >= p_target.

    Omega = Finite set of candidate points. If Omega = None then
    global optimization is performed in the region defined by 'bounds'
    
    bounds = bounds on input space
    p_target = target constraint probability
    """

    if (Omega is None) ^ (bounds is None):
        raise ValueError("Either supply Omega (finite set of points) or bounds (on input space)")

    # Determine which sub-operators to include
    if i_range is None:
        i_range = list(range(len(self.constraints)))

    # Set nu for widened constraint margins
    if nu is None:
        nu = max(self.constr_likelihood * sp.stats.norm.ppf(p_target), 0)

    # Reset all previous virtual points
    self.reset_XV()

    rows = []
    i_add_pts = 0

    for j in range(max_iterations):

        pc_list, x_list = [], []

        # Loop over each sub-operator and find the most violated point
        for i in i_range:
            if Omega is None:
                # Global optimization
                success, x_min, pc_min = self._argmin_pc_subop(
                    i, nu, bounds, sampling_alg=sampling_alg,
                    num_samples=num_samples
                )
            else:
                # Finite candidate set
                pc_min, x_min = self._argmin_pc_subop_finite(
                    i, nu, Omega, batch_size=batch_size,
                    sampling_alg=sampling_alg, num_samples=num_samples,
                )
                success = True

            if not success: return None
            pc_list.append(pc_min)
            x_list.append(x_min)

        # Choose the operator with the worst violation (lowest p_c)
        pc_min = min(pc_list)
        i_min = pc_list.index(pc_min)
        x_min = x_list[i_min]

        # Store current progress
        rows.append([j, i_min] + list(x_min) + pc_list)

        # Check if all constraints are now satisfied
        if pc_min >= p_target:
            break

        # Add new virtual point to the appropriate constraint
        i_add_pts += 1
        self.constraints[i_min].add(x_min)

        # Invalidate any cached matrices dependent on Xv
        self.reset_XV() # need to make this

    # Format and return result table
    df_out = pd.DataFrame(rows)
    df_out.columns = ['num_Xv', 'update_constr'] + [f'Xv[{i+1}]' for i in range(len(x_min))] + [f'pc_{i+1}' for i in i_range]
    return df_out, i_add_pts, pc_min

def _argmin_pc_subop(self, i, nu, bounds,
                     sampling_alg='ess',
                     num_samples=1000):

    min_prob_log = 1E-10

    if not self.cached:
        self._prepare()

    if not self._has_virtual_obs():
        args = (i, nu)
        def optfun(x, *args):
            i, nu = args
            p_c = self._constrprob_xs_1(np.array(x).reshape(1, -1), i, nu)[0]
            return np.log(max(p_c, min_prob_log))
    else:
        args = (i, nu, num_samples, sampling_alg)
        def optfun(x, *args):
            i, nu, n, alg = args
            p_c = self._constrprob_xs_2(np.array(x).reshape(1, -1), i, nu, n, alg, False)[0]
            return np.log(max(p_c, min_prob_log))

    res = sp.optimize.differential_evolution(optfun, bounds=bounds, args=args)
    return res.success, res.x, np.exp(res.fun)

def _has_virtual_obs(self) -> bool:
    """Check if any constraint has virtual observation points"""
    return any(c.Xv is not None for c in self.constraints)

def _constrprob_xs_1(self, XS, i, nu):
    # done
    """
    Return the probability that the i-th constraint is satisfied at XS
    For use when no constraint point supplied by user
    C~(XS) | Y
    """
    
    # Get mean and cov
    mu, cov = self._constr_posterior_dist_1(XS, i)
    std = np.sqrt(np.diagonal(cov))

    # Get bound vectors for constraint distribution
    LB, UB = self.calc_constr_bounds_subop(XS, i)
    
    # Widen intervals with nu
    LB = LB - nu
    UB = UB + nu

    # Calculate probability that the constraint holds at each XV
    return norm_cdf_int_approx(np.array(mu)[:,0], std, LB, UB) # Aprroximation within E-7 error

def _constr_posterior_dist_1(self, XS, i):
    # done
    """
    Return mean and covariance of the i-th constraint at XS
    
    C~(XS) | Y
    """
    
    if not self.cached:
        self._prepare()

    # c_v2, c_A2 and c_B2
    c_v2, c_A2, c_B2 = self._constr_prep_1(XS, i)
    
                        
    # Get mean vector
    Lmu_all, _ = self._calc_constr_mean()

    # Get i-th block of Lmu
    constraint = self.constraints[i]
    n_XS = XS.shape[0]

    # Slice out Lmu_i from stacked Lmu_all
    if constraint.Xv is not None:
        sizes = [c.Xv.shape[0] for c in self.constraints if c.Xv is not None]
        idx_start = sum(sizes[:i])
        idx_end = idx_start + sizes[i]
        Lmu_i = Lmu_all[idx_start:idx_end]
    else:
        Lmu_i = np.zeros((n_XS, 1))    

    # Posterior mean
    mu = Lmu_i + c_A2@self.f_tilde
    
    # Return posterior mean and covariance
    return mu, c_B2

def _constr_prep_1(self, XS, i): # i think _calc_L1L2 and _calc_L2T can be generalised to do this or maybe it would be easier to just write this out
    """
    Return c_v2, c_A2 and c_B2 for constraint distribution
    """
    # done
    constraint = self.constraint[i]
    L2T_K_X_XS = constraint.op.K_with_f(self.kernel, self.X, XS)
    L1L2T_K_XS_XS = constraint.op.K_with_L(self.kernel, self.X, constraint.op, XS)

    c_v2 = cho_solve((self.L, True), L2T_K_X_XS) 
    c_A2 = cho_solve((self.L.T, False), c_v2).T 
    c_B2 = L1L2T_K_XS_XS - c_v2.T@c_v2
    
    return c_v2, c_A2, c_B2

def calc_constr_bounds_subop(self, XS, i):
    # done
    """ Return lower/upper bounds for the i-th suboperator only at XS """
    
    constraint = self.constraints[i]
    LB = constraint.LBXV(XS)
    UB = constraint.UBXV(XS)
    return LB, UB


def _constrprob_xs_2(self, XS, i, nu, num_samples, algorithm, verbatim = False):
    # done
    """
    Return the probability that the i-th constraint is satisfied at XS
    For use when constraint points supplied by user

    C~(XS) | Y, C
    """
    
    # Calculations only depending on (X, Y)
    if not self.cached:
        self._prepare()

    # Calculations only depending on (X, XV) - v1, A1 and B1
    L2T_K_x_xv = self._calc_L2T(self.X_training)
    L1L2T_K_xv_xv = self._calc_L1L2()
    v1 = cho_solve((L, True), L2T_K_x_xv, check_finite=False)
    A1 = cho_solve((L.T, False), v1).T

    n = L1L2T_K_xv_xv.shape[0]
    B1 = L1L2T_K_xv_xv + (self.var_constr*np.identity(n)) - v1.T@v1

    # Calculate mean of constraint distribution at XV (covariance is B1)
    Lmu_XV, constr_mean = self._calc_constr_mean()
    
    # Get bound vectors for constraint distribution
    LB, UB = self._calc_constr_bounds()
    
    # Sample from truncated constraint distribution 
    self._sample_constr_XV(m = num_samples, mu = constr_mean, sigma = B1, LB = LB, UB = UB, algorithm = algorithm, resample = False, verbatim = verbatim)

    # c_v2, c_A2 and c_B2
    # Only compute diagonal elements of constraint covariance

    constraint = self.constraints[i]

    # Compute covariance blocks
    L2T_K_X_XS = constraint.op.K_with_f(self.kernel, constraint.Xv, self.X_training).T
    K_XS_XS = constraint.op.K_with_L(self.kernel, XS, constraint.op, XS)

    # Take diagonal
    L1L2T_K_XS_XS_diag = np.diag(K_XS_XS).reshape(-1, 1)


    c_v2 = cho_solve((self.L, True), L2T_K_X_XS) 
    c_A2 = cho_solve((self.L.T, False), c_v2).T

    # c_A, c_B and c_Sigma
    L1L2T_XS_XV = self._calc_FiL2T(XS, i)
    
    c_B3 = L1L2T_XS_XV - c_v2.T@v1
    
    self._prep_L1() # Compute L_1
    L_1 = self._compute_cholesky(B1, only_L = True)

    c_v3 = cho_solve((L_1, True), c_B3.T)

    c_A = cho_solve((L_1.T, False), c_v3).T
    
    c_B = c_A2 - c_A@A1
    
    c_Sigma_diag = (
                    L1L2T_K_XS_XS_diag.flatten() 
                    - np.sum(np.square(c_v2), axis=0) 
                    - np.sum(np.square(c_v3), axis=0)
                    ).reshape(-1, 1)

    # Get bound vectors for constraint distribution
    LB, UB = self.calc_constr_bounds_subop(XS, i)

    # Widen intervals with nu
    LB = LB - nu
    UB = UB + nu
    
    # Prior mean
    constraint = self.constraints[i]
    mu_xs = self._mean_function(XS)              # shape (n_XS, 1)
    Lmu = constraint.op.apply_mean(mu_xs)
    
    # Posterior mean
    mu = Lmu + c_A@(self.C_sim - Lmu_XV) + c_B@self.f_tilde
    
    # Posterior standard deviation
    std = np.array(np.sqrt(c_Sigma_diag)).flatten()

    # Calculate probability that the constraint holds at each XS individually 
    # for each sample C_j and take the average over C_j
    if XS.shape[0] == 1:
        
        # Faster for single input
        probs = norm_cdf_int_approx(np.array(mu)[0], std, LB, UB)
        probs = np.array([probs.mean()])
        
    else:
        probs = np.apply_along_axis(norm_cdf_int_approx, axis = 0, arr = np.array(mu), std = std, LB = LB, UB = UB)
        probs = probs.mean(axis = 1)
        
    # Return probability
    return probs

def _calc_FiL2T(self, XS: np.ndarray, i: int) -> np.ndarray:
    # done
    """
    Return block-row i of the constraint covariance cross-term:
    Cov[L_i f(XS), L_j f(Xv_j)] for all j (i.e. Fi L2^T K_XS_XV)
    """
    constraint_i = self.constraints[i]
    blocks = []

    for constraint_j in self.constraints:
        if constraint_j.Xv is not None:
            # Cov[L_i f(XS), L_j f(Xv_j)]
            K_block = constraint_i.op.K_with_L(self.kernel, XS, constraint_j.op, constraint_j.Xv)
            blocks.append(K_block)

    return np.block(blocks) if blocks else np.zeros((XS.shape[0], 0))


def _argmin_pc_subop_finite(self, 
                            i, 
                            nu, 
                            Omega, 
                            batch_size = None, 
                            sampling_alg = 'ess', 
                            num_samples = 1000):
    # unworked
    """
    Same as _armin_pc_subup but over a finite domain Omega
    """

    # Calculations only depending on (X, Y)
    if not self.cached:
        self._prepare()

    if batch_size is None: batch_size = Omega.shape[0]

    # Split Omega in batches
    assert batch_size <= Omega.shape[0], 'batch_size must be less than number of elements in Omega'

    num_intervals, rem = np.divmod(Omega.shape[0], batch_size)

    # Compute constraint probability for each element in Omega
    if not self._has_virtual_obs():
        
        p_c = []
        for j in range(num_intervals): 
            p_c += list(self._constrprob_xs_1(Omega[j*batch_size:(j+1)*batch_size], i, nu))

        if rem != 0:
            p_c += list(self._constrprob_xs_1(Omega[-rem:], i, nu))
    else:
        p_c = []
        for j in range(num_intervals): 
                p_c += list(self._constrprob_xs_2(Omega[j*batch_size:(j+1)*batch_size], i, nu, num_samples, sampling_alg))

        if rem != 0:
                p_c += list(self._constrprob_xs_2(Omega[-rem:], i, nu, num_samples, sampling_alg))

    # Find smallest element
    p_c = np.array(p_c)
    idx = p_c.argmin()
    prob = p_c[idx]
    argmin = Omega[idx]

    return prob, argmin

def _sample_constr_XV(self, m, mu, sigma, LB, UB, algorithm, resample = False):
    """ 
    Generate m samples from the constraint distribution
    
    Input: 
    m -- number of samples
    mu, sigma, LB, UB -- distribution parameters of truncated Gaussian
    algorithm -- name of sampling algorithm ('rejection', 'gibbs' or 'minimax_tilting')
    resample -- resample = False -> Use old samples if they exist
    
    """
    n = mu.size
    M = np.vstack([np.eye(n), -np.eye(n)])
    g = np.hstack([-LB, UB])

    # Check if we should just use the old samples
    if self.C_sim is None: 
        generate_samples = True
    else:
        if m == self.C_sim.shape[1]:
            generate_samples = resample
        else:
            generate_samples = True
                

    if generate_samples:
        # Generate samples        
        #self.C_sim = rtmvnorm(n = m, mu = mu, sigma = sigma, a = LB, b = UB, algorithm = algorithm).T
    
        self.C_sim = sample_tmvn_ess(
            mu=mu,
            cov=sigma,
            A=M,
            b=g,
            X=self.X_training,
            y=self.Y,
            n_samples=m,
            burn_in=100,
            eta_init=20.0,
            update_eta=False
        )


