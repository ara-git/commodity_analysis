import numpy as np
import pandas as pd


class KalmanFilter:
    def __init__(
        self,
        params,
        rate,
        T=1,
        dt=1 / 252,
        inital_beta=[1, 1],
        initial_beta_cov=np.eye(2),
    ):
        self.sigma_1 = params["sigma_1"]
        self.sigma_2 = params["sigma_2"]
        self.rho = params["rho"]
        self.kappa = params["kappa"]
        self.alpha = params["alpha"]
        self.lambda_ = params["lambda"]
        self.mu = params["mu"]
        self.epsilon_e = params["epsilon_e"]

        self.rate = rate
        self.T = T  # maturity of forward contract
        self.dt = dt  # Daily time step
        self.inital_beta = np.array(inital_beta).reshape(2, 1)  # [2x1]
        self.initial_beta_cov = initial_beta_cov  # [2x2]

        self.alpha_hat = self.alpha - self.lambda_ / self.kappa

        # Prepare measurement and state equations
        self._prepare_measurement_equation()
        self._prepare_state_equation()

    def _prepare_measurement_equation(self):
        """Prepares the measurement equation components."""
        # intercept
        self.c = (
            self.rate
            - self.alpha_hat
            + 0.5 * self.sigma_2**2 / self.kappa**2
            - self.sigma_1 * self.sigma_2 * self.rho / self.kappa
        ) * self.T
        +0.25 * self.sigma_2**2 * (1 - np.exp(-2 * self.kappa * self.T)) / self.kappa**3
        +(
            self.alpha_hat * self.kappa
            + self.sigma_1 * self.sigma_2 * self.rho
            - self.sigma_2**2 / self.kappa
        ) * (1 - np.exp(-self.kappa * self.T)) / self.kappa**2

        # slope(Exogenous variable)
        self.X = np.array(
            [[1, -(1 - np.exp(-self.kappa * self.T)) / self.kappa]]
        )  # [1x2]

    def _prepare_state_equation(self):
        """Prepares the state equation components."""
        self.D = np.array(
            [
                (self.mu - 0.5 * self.sigma_1**2) * self.dt,
                self.kappa * self.alpha * self.dt,
            ]
        ).reshape(2, 1)
        self.T = np.array([[1, -self.dt], [0, 1 - self.kappa * self.dt]])  # [2x2]
        self.Q = np.array(
            [
                [
                    self.sigma_1**2 * self.dt,
                    self.rho * self.sigma_1 * self.sigma_2 * self.dt,
                ],
                [
                    self.rho * self.sigma_1 * self.sigma_2 * self.dt,
                    self.sigma_2**2 * self.dt,
                ],
            ]
        )  # [2x2]

    def read_csv(self, file_path="./data/WTI_Combined.csv"):
        """Reads a CSV file and returns spot and forward prices."""
        df = pd.read_csv(file_path)
        self.y_observed_list = np.log(df["Forward_Price"].values)

    def run_kalman_filter(self):
        """Runs the Kalman filter on the spot and forward prices."""
        num_of_observation = len(self.y_observed_list)
        ###
        # Initialize arrays to store results
        self.Beta_filtered_list = np.zeros((num_of_observation, 2, 1))
        self.Beta_cov_filtered_list = np.zeros((num_of_observation, 2, 2))
        self.Beta_pred_list = np.zeros((num_of_observation, 2, 1))
        self.Beta_cov_pred_list = np.zeros((num_of_observation, 2, 2))
        self.y_pred_list = np.zeros((num_of_observation, 1))

        # Initial state
        Beta_current = self.inital_beta  # [2x1]
        Beta_cov_current = self.initial_beta_cov  # [2x2]

        for t in range(num_of_observation):  # t = 0, 1, 2, ..., num_of_observation-1
            # Prediction step
            Beta_pred, Beta_cov_pred, y_pred = self._predict_next_step(
                Beta_current, Beta_cov_current
            )

            # Update step
            y_observed = self.y_observed_list[t]
            Beta_filtered, Beta_cov_filtered = self._filter_current_step(
                Beta_pred, Beta_cov_pred, y_pred, y_observed
            )

            # Store results
            self.Beta_filtered_list[t] = Beta_filtered
            self.Beta_cov_filtered_list[t] = Beta_cov_filtered
            self.Beta_pred_list[t] = Beta_pred
            self.Beta_cov_pred_list[t] = Beta_cov_pred
            self.y_pred_list[t] = y_pred

            # Update current state for next iteration
            Beta_current = Beta_filtered
            Beta_cov_current = Beta_cov_filtered
        return None

    def _predict_next_step(self, Beta_current, Beta_cov_current):
        """Predicts the next state and covariance."""
        Beta_pred = self.D + self.T @ Beta_current  # [2x1]
        Beta_cov_pred = self.T @ Beta_cov_current @ self.T.T + self.Q  # [2x2]
        y_pred = float(self.c + self.X @ Beta_pred)  # [1x1]
        return Beta_pred, Beta_cov_pred, y_pred

    def _filter_current_step(self, Beta_pred, Beta_cov_pred, y_pred, y_observed):
        """Applies the Kalman filter to the given spot and forward prices."""
        Kalman_gain = (
            Beta_cov_pred
            @ self.X.T
            / (self.X @ Beta_cov_pred @ self.X.T + self.epsilon_e**2)
        )  # [2x1]
        Beta_filtered = Beta_pred + Kalman_gain * (y_observed - y_pred)  # [2x1]
        Beta_cov_filtered = (np.eye(2) - Kalman_gain @ self.X) @ Beta_cov_pred  # [2x2]

        return Beta_filtered, Beta_cov_filtered

    def plot_results_of_prices(self):
        """Plots the observed and predicted forward prices."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(self.y_observed_list, label="Observed Forward Price", color="blue")
        plt.plot(
            self.y_pred_list,
            label="Predicted Forward Price",
            color="red",
            linestyle="--",
        )
        plt.xlabel("Time")
        plt.ylabel("Forward Price")
        plt.title("Kalman Filter: Observed vs Predicted Forward Prices")
        plt.legend()
        plt.show()

    def plot_results_of_beta(self):
        """Plots the filtered state variables over time."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))
        plt.plot(
            self.Beta_filtered_list[:, 0],
            label="Filtered Beta 1",
            color="blue",
        )
        plt.plot(
            self.Beta_filtered_list[:, 1],
            label="Filtered Beta 2",
            color="red",
        )
        plt.xlabel("Time")
        plt.ylabel("Filtered State Variables")
        plt.title("Kalman Filter: Filtered State Variables Over Time")
        plt.legend()
        plt.show()

    def export_results_to_csv(self, file_path="./data/Kalman_Filter_Results.csv"):
        """Exports the Kalman filter results to a CSV file."""
        results_df = pd.DataFrame(
            {
                "Observed_Forward_Price": self.y_observed_list.flatten(),
                "Predicted_Forward_Price": self.y_pred_list.flatten(),
                "Filtered_Beta_1": self.Beta_filtered_list[:, 0].flatten(),
                "Filtered_Beta_2": self.Beta_filtered_list[:, 1].flatten(),
            }
        )
        results_df.to_csv(file_path, index=False)
        print(f"Kalman filter results exported to {file_path}")


if __name__ == "__main__":
    params = {
        "sigma_1": 0.393,
        "sigma_2": 0.527,
        "rho": 0.766,
        "kappa": 1.876,
        "alpha": 0.106,
        "lambda": 0.198,
        "mu": 0.142,
    }  # from schwartz (1997), table VI
    params.update({"epsilon_e": 0.5})  # set temporarily

    ins_Kalman_filter = KalmanFilter(params, rate=0.05)
    ins_Kalman_filter.read_csv()
    ins_Kalman_filter.run_kalman_filter()
    ins_Kalman_filter.plot_results_of_prices()
    ins_Kalman_filter.plot_results_of_beta()
    ins_Kalman_filter.export_results_to_csv()
