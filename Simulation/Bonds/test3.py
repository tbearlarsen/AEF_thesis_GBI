import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt


class TwoFactorTermStructureModel(sm.tsa.statespace.MLEModel):
    def __init__(self, endog, maturities):
        """
        Two-Factor Term Structure Model in State-Space Form.

        Parameters:
          endog (ndarray): T x m array of observed yields (T time periods, m maturities).
          maturities (list or array): List of maturities corresponding to the columns (in years).
        """
        self.maturities = np.asarray(maturities)
        m = len(maturities)
        k_states = 2  # two latent factors
        super(TwoFactorTermStructureModel, self).__init__(endog, k_states=k_states, k_posdef=k_states)

        # Initialize the state: diffuse initialization
        self.initialize_approximate_diffuse()

        # Set up state-space matrices that will be updated with parameters:
        # Transition matrix F (2x2) -- here we assume diagonal.
        self['transition'] = np.eye(2)  # will be updated in update()

        # Design (observation) matrix: yields are modeled as
        # y_t = D - B * state_t + error, so design = -B (m x 2)
        m = endog.shape[1]
        self['design'] = np.zeros((m, 2))  # will update in update()

        # Observation intercept (D) -- an m-dimensional vector (stored separately)
        self.obs_intercept = np.zeros(m)  # will update in update()

        # Selection matrix (we use identity here)
        self['selection'] = np.eye(2)

        # The state and observation covariance matrices will be updated in update().
        # Initialize _start_params as None; we'll set it externally.
        self._start_params = None

    @property
    def start_params(self):
        """
        Return the initial parameters for the optimization.
        This property must be implemented for MLEModel.
        """
        if self._start_params is None:
            # If not set, return a default vector of zeros.
            return np.zeros(self.k_params)
        else:
            return self._start_params

    def update(self, params, **kwargs):
        """
        Update the state-space matrices given the parameter vector.

        The parameter vector is structured as follows:
          params = [phi1, phi2, log(sigma_v1), log(sigma_v2),
                    D_1, ..., D_m,
                    beta_11, ..., beta_m1,
                    beta_12, ..., beta_m2,
                    log(sigma_e1), ..., log(sigma_e_m)]

        Total number of parameters = 4 + 4*m, where m = number of maturities.
        """
        m = self.endog.shape[1]
        # Unpack parameters:
        phi1, phi2, log_sigma_v1, log_sigma_v2 = params[:4]
        D = params[4:4 + m]
        beta1 = params[4 + m:4 + 2 * m]
        beta2 = params[4 + 2 * m:4 + 3 * m]
        log_sigma_e = params[4 + 3 * m:4 + 4 * m]

        sigma_v1 = np.exp(log_sigma_v1)
        sigma_v2 = np.exp(log_sigma_v2)
        sigma_e = np.exp(log_sigma_e)

        # Update state transition matrix (assumed diagonal):
        self['transition'] = np.array([[phi1, 0],
                                       [0, phi2]])

        # Update state noise covariance Q:
        self['state_cov'] = np.diag([sigma_v1 ** 2, sigma_v2 ** 2])

        # Update design (observation) matrix:
        # Our measurement equation: y_t = D - B * state_t + error,
        # so design = -B, where B is m x 2.
        B = np.column_stack((beta1, beta2))
        self['design'] = -B

        # Update the observation intercept:
        self.obs_intercept = D

        # Update observation noise covariance R (diagonal):
        self['obs_cov'] = np.diag(sigma_e ** 2)

    def transform_params(self, unconstrained):
        # Identity transform; using logs in update ensures positivity.
        return unconstrained

    def untransform_params(self, constrained):
        return constrained


# ---------------------------
# Usage Example
# ---------------------------
if __name__ == '__main__':
    # Load your Excel file containing yield data.
    # Assume the Excel file has an index column (dates) and columns: '1Y', '5Y', '10Y', '20Y', '30Y'
    data_file = r"/Users/osito/Repositories/AEF_thesis_GBI/Simulation/Bonds/Data/monthly_rates.xlsx"  # adjust path if needed
    data = pd.read_excel(data_file, index_col=0)

    # Extract observed yields (ensure they are numeric and drop any rows with missing data)
    observed_yields = data[['1Y', '5Y', '10Y', '20Y', '30Y']]
    observed_yields = observed_yields.apply(pd.to_numeric, errors='coerce').dropna()

    # Maturities (in years) corresponding to the columns:
    maturities = [1, 5, 10, 20, 30]

    # Endogenous variable for the state-space model: a T x m numpy array
    endog = observed_yields.values

    # Instantiate the two-factor term structure model
    model2f = TwoFactorTermStructureModel(endog, maturities)

    # Set up initial parameter guesses.
    # Total parameters = 4 + 4*m, with m = number of maturities.
    m = len(maturities)
    # Initial guesses for phi's (assume 0.9)
    phi1 = 0.9
    phi2 = 0.9
    # Log sigma_v's (small values, e.g., log(0.01))
    log_sigma_v1 = np.log(0.01)
    log_sigma_v2 = np.log(0.01)
    # D: initial guess for intercepts, e.g., sample means of yields.
    D_init = np.mean(endog, axis=0)
    # Beta's: initial guess, say 0.1 for each loading.
    beta1_init = 0.1 * np.ones(m)
    beta2_init = 0.1 * np.ones(m)
    # Log sigma_e's: initial guess, e.g., log(0.005)
    log_sigma_e_init = np.log(0.005) * np.ones(m)

    # Combine into an initial parameter vector:
    init_params = np.r_[phi1, phi2, log_sigma_v1, log_sigma_v2,
    D_init, beta1_init, beta2_init, log_sigma_e_init]

    # Set the initial parameters in the model (this satisfies the start_params property)
    model2f._start_params = init_params

    # Fit the model via maximum likelihood
    res = model2f.fit(disp=False)
    print(res.summary())

    # Filter the latent states using the fitted model
    filtered_states = res.filtered_state
    print("Filtered states (last observation):", filtered_states[:, -1])

    # Compute model-implied yields at the last observation.
    # Our measurement equation is: y = D - B * state.
    D_est = res.params[4:4 + m]
    beta1_est = res.params[4 + m:4 + 2 * m]
    beta2_est = res.params[4 + 2 * m:4 + 3 * m]
    last_state = filtered_states[:, -1]  # shape (2,)
    model_implied_yields = D_est - beta1_est * last_state[0] - beta2_est * last_state[1]

    print("\nModel-implied yields at last observation:")
    for T, y in zip(maturities, model_implied_yields):
        print(f"  {T}Y yield: {y:.4%}")

    # Plot the yield curves: observed (last observation) vs. model-implied.
    observed_curve = endog[-1, :]
    plt.figure(figsize=(8, 5))
    plt.plot(maturities, model_implied_yields, label='Model-Implied Yields', marker='o')
    plt.plot(maturities, observed_curve, label='Observed Yields', marker='x')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield')
    plt.title('Model-Implied vs. Observed Yield Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
