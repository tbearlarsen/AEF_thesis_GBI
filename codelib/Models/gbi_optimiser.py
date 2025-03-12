import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize
import matplotlib.pyplot as plt


###############################################################################
# Data Handling Class
###############################################################################
class PortfolioData:
    def __init__(self, goal_data_path, capmkt_path, correlations_path):
        """
        Initialise the data handler with file paths.
        """
        self.goal_data_path = goal_data_path
        self.capmkt_path = capmkt_path
        self.correlations_path = correlations_path

        self._load_data()
        self._initialize_parameters()

    def _load_data(self):
        """Load CSV data from file paths."""
        self.goal_data_raw = pd.read_csv(self.goal_data_path)
        self.capmkt_raw = pd.read_csv(self.capmkt_path)
        self.correlations_raw = pd.read_csv(self.correlations_path)

    def _initialize_parameters(self):
        """
        Set key parameters including number of assets, goals, expected returns,
        asset names, the covariance matrix and goal vectors.
        """
        self.num_assets = len(self.capmkt_raw.iloc[:, 1])
        self.num_goals = self.goal_data_raw.shape[1] - 1  # assume first column is not a goal

        self.return_vector = self.capmkt_raw.iloc[:, 1].to_numpy()
        self.asset_names = self.capmkt_raw.iloc[:, 0].astype(str).tolist()

        # Get correlations and build the covariance matrix.
        self.correlations = self.correlations_raw.iloc[:15, 1:16].astype(float)
        self.covariances = np.zeros((self.num_assets, self.num_assets))
        for i in range(self.num_assets):
            for j in range(self.num_assets):
                self.covariances[j, i] = (
                        self.capmkt_raw.iloc[i, 2] *
                        self.capmkt_raw.iloc[j, 2] *
                        self.correlations.iloc[j, i]
                )

        # Parse goal data.
        # We assume four goals (A, B, C, D) with each goal vector of the form:
        # [value ratio, funding requirement, time horizon]
        self.goal_vectors = {}
        goal_labels = ["A", "B", "C", "D"]
        for idx, label in enumerate(goal_labels):
            self.goal_vectors[label] = [
                self.goal_data_raw.iloc[0, idx + 1],
                self.goal_data_raw.iloc[1, idx + 1],
                self.goal_data_raw.iloc[2, idx + 1]
            ]
        self.goal_labels = goal_labels

        # Starting weights: random initialisation normalised to sum to 1.
        self.starting_weights = np.random.uniform(0, 1, self.num_assets)
        self.starting_weights /= np.sum(self.starting_weights)


###############################################################################
# Optimiser Class
###############################################################################
class PortfolioOptimizer:
    def __init__(self, data: PortfolioData, pool, n_trials=10 ** 5):
        """
        Initialise the optimiser with a PortfolioData instance, pool of wealth,
        and number of Monte Carlo trials.
        """
        self.data = data
        self.pool = pool
        self.n_trials = n_trials

        # Set attributes from the data object.
        self.num_assets = data.num_assets
        self.num_goals = data.num_goals
        self.return_vector = data.return_vector
        self.asset_names = data.asset_names
        self.covariances = data.covariances
        self.goal_vectors = data.goal_vectors
        self.goal_labels = data.goal_labels
        self.starting_weights = data.starting_weights

    # Static methods for basic functions.
    @staticmethod
    def sd_f(weight_vector, covar_table):
        """Calculate portfolio volatility."""
        covar_vector = np.zeros(len(weight_vector))
        for z in range(len(weight_vector)):
            covar_vector[z] = np.sum(weight_vector * covar_table[:, z])
        return np.sqrt(np.sum(weight_vector * covar_vector))

    @staticmethod
    def mean_f(weight_vector, return_vector):
        """Calculate expected portfolio return."""
        return np.sum(weight_vector * return_vector)

    @staticmethod
    def phi_f(goal_vector, goal_allocation, pool, mean, sd):
        """
        Calculate goal achievement probability.
        goal_vector is [value ratio, funding requirement, time horizon].
        """
        required_return = (goal_vector[1] / (pool * goal_allocation)) ** (1 / goal_vector[2]) - 1
        if goal_allocation * pool >= goal_vector[1]:
            return 1
        else:
            return 1 - norm.cdf(required_return, loc=mean, scale=sd)

    def _optim_function(self, weights, goal_vector, allocation):
        """Objective function for optimising subportfolio weights."""
        return 1 - self.phi_f(
            goal_vector,
            allocation,
            self.pool,
            self.mean_f(weights, self.return_vector),
            self.sd_f(weights, self.covariances)
        )

    @staticmethod
    def _constraint_function(weights):
        """Equality constraint: weights must sum to 1."""
        return np.sum(weights) - 1

    def optimize_within_goal_allocation(self):
        """
        Step 1:
        Enumerate possible across-goal allocations (from 0.01 to 1)
        and optimise the subportfolio weights for each goal.
        """
        self.goal_allocation = np.arange(0.01, 1.01, 0.01)
        # Store optimal weights and phi values for each goal.
        self.optimal_weights = {label: np.zeros((len(self.goal_allocation), self.num_assets))
                                for label in self.goal_labels}
        self.phi = {label: np.zeros(len(self.goal_allocation))
                    for label in self.goal_labels}

        for i, alloc in enumerate(self.goal_allocation):
            for label in self.goal_labels:
                goal_vector = self.goal_vectors[label]
                # If funding requirement is met, allocate entirely to the last asset.
                if goal_vector[1] <= self.pool * alloc:
                    optimal_weights_vec = [0] * (self.num_assets - 1) + [1]
                    self.optimal_weights[label][i, :] = optimal_weights_vec
                else:
                    res = minimize(
                        self._optim_function,
                        self.starting_weights,
                        args=(goal_vector, alloc),
                        constraints=[{'type': 'eq', 'fun': self._constraint_function}],
                        bounds=[(0, 1)] * self.num_assets,
                        method='SLSQP'
                    )
                    self.optimal_weights[label][i, :] = res.x

                # Calculate phi for this goal and allocation.
                self.phi[label][i] = self.phi_f(
                    goal_vector, alloc, self.pool,
                    self.mean_f(self.optimal_weights[label][i, :], self.return_vector),
                    self.sd_f(self.optimal_weights[label][i, :], self.covariances)
                )

    def simulate_across_goal_allocation(self):
        """
        Step 2:
        Simulate goal weights across trials and compute the utility for each simulated portfolio.
        """
        self.sim_goal_weights = np.zeros((self.n_trials, self.num_goals))
        for i in range(self.n_trials):
            rand_vector = np.random.uniform(0, 1, self.num_goals)
            normalizer = np.sum(rand_vector)
            self.sim_goal_weights[i, :] = np.where(
                np.round((rand_vector / normalizer) * 100, 0) < 1,
                1,
                np.round((rand_vector / normalizer) * 100)
            )
        self.sim_goal_weights = self.sim_goal_weights.astype(int)

        # Calculate utility for each simulated portfolio.
        # Note: subtract 1 from simulated weights for 0-indexing.
        utility = (
                self.goal_vectors["A"][0] * self.phi["A"][self.sim_goal_weights[:, 0] - 1] +
                self.goal_vectors["B"][0] * self.phi["B"][self.sim_goal_weights[:, 1] - 1] +
                self.goal_vectors["C"][0] * self.phi["C"][self.sim_goal_weights[:, 2] - 1] +
                self.goal_vectors["D"][0] * self.phi["D"][self.sim_goal_weights[:, 3] - 1]
        )
        self.optimal_goal_weights = self.sim_goal_weights[np.argmax(utility), :]

    def compute_aggregate_portfolio(self):
        """
        Step 3:
        Retrieve the optimal subportfolio allocations and compute the aggregate portfolio.
        """
        self.optimal_subportfolios = np.zeros((self.num_goals, self.num_assets))
        for i, label in enumerate(self.goal_labels):
            allocation_index = int(self.optimal_goal_weights[i]) - 1  # adjust for zero-indexing
            self.optimal_subportfolios[i, :] = self.optimal_weights[label][allocation_index, :]

        # Normalise simulated goal weights so they sum to 1, then compute the aggregate portfolio.
        optimal_goal_weights_norm = self.optimal_goal_weights / np.sum(self.optimal_goal_weights)
        self.optimal_aggregate_portfolio = optimal_goal_weights_norm @ self.optimal_subportfolios

    def plot_goal_allocation(self, goal="A"):
        """
        Visualise the subportfolio allocation as a function of the across-goal allocation.

        Parameters:
            goal (str or list): A single goal label (e.g. "A"), a list of goal labels,
                                or "all" to plot every goal.
        """
        if isinstance(goal, str):
            if goal.lower() == "all":
                goal_list = self.goal_labels
            else:
                goal_list = [goal]
        elif isinstance(goal, list):
            goal_list = goal
        else:
            raise ValueError("The 'goal' parameter must be a string or list of strings.")

        for g in goal_list:
            if g not in self.optimal_weights:
                print(f"Goal {g} not found in optimal weights.")
                continue
            plt.figure(figsize=(10, 6))
            plt.stackplot(self.goal_allocation * 100,
                          self.optimal_weights[g].T,
                          labels=self.asset_names,
                          alpha=0.7)
            plt.xlabel("Goal Allocation (%)", fontsize=14, fontweight='bold')
            plt.ylabel("Investment Weight", fontsize=14, fontweight='bold')
            plt.title(f"Goal {g} Subportfolio Allocation", fontsize=16, fontweight='bold')
            plt.legend(title="Asset", fontsize=12, title_fontsize=14)
            plt.grid(alpha=0.3)
            plt.show()

    def print_results(self):
        """Print the optimal across-goal allocation and the aggregate portfolio."""
        print("Optimal Across-Goal Allocation:")
        print(self.optimal_goal_weights)
        print("\nOptimal Aggregate Investment Allocation:")
        print(self.optimal_aggregate_portfolio)

    def run(self):
        """
        Run the complete optimisation process.
        Returns a dictionary with key output variables.
        """
        self.optimize_within_goal_allocation()
        self.simulate_across_goal_allocation()
        self.compute_aggregate_portfolio()
        self.plot_goal_allocation("all")
        self.print_results()
        return {
            "optimal_goal_weights": self.optimal_goal_weights,
            "optimal_aggregate_portfolio": self.optimal_aggregate_portfolio,
            "optimal_subportfolios": self.optimal_subportfolios
        }


###############################################################################
# Example Usage
###############################################################################
if __name__ == "__main__":
    # Adjust file paths as needed.
    goal_data_path = r"C:\Users\admin\Desktop\Thesis\AEF_msc_thesis_GBI\Example Goal Details.csv"
    capmkt_path = r"C:\Users\admin\Desktop\Thesis\AEF_msc_thesis_GBI\Capital Market Expectations.csv"
    correlations_path = r"C:\Users\admin\Desktop\Thesis\AEF_msc_thesis_GBI\Correlations - Kitchen Sink.csv"
    pool = 4654000

    # Create a data handler instance.
    data = PortfolioData(goal_data_path, capmkt_path, correlations_path)
    # Create an optimiser instance using the data.
    optimizer = PortfolioOptimizer(data, pool)
    results = optimizer.run()

    # Print returned variables.
    print("\nReturned Variables:")
    for key, value in results.items():
        print(f"{key}:")
        print(value)
