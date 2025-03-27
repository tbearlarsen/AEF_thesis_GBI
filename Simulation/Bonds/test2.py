from codelib.Models.vasicek_model import VasicekEstimator
from codelib.file_management.dynamic_file_pathing import get_root
import os

def main():
    root = get_root()
    data = os.path.join(root, "Simulation", "Bonds", "Data", "monthly_rate_and_yields.xlsx")

    estimator = VasicekEstimator(data, 12)
    kappa, theta, beta = estimator.estimate_params()
    rp = estimator.estimate_risk_premium()

    return kappa, theta, beta, rp

if __name__ == "__main__":
    kappa, theta, beta, rp = main()


