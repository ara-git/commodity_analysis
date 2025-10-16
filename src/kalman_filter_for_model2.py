import numpy as np
import pandas as pd
from scipy.optimize import minimize


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
        self.params = params
        self.rate = rate
        self.forward_maturity = T  # maturity of forward contract
        self.dt = dt  # Daily time step
        self.inital_beta = np.array(inital_beta).reshape(2, 1)  # [2x1]
        self.initial_beta_cov = initial_beta_cov  # [2x2]

        # Prepare measurement and state equations
        self._update_params(params)
        self._prepare_measurement_equation()
        self._prepare_state_equation()

    def _update_params(self, params):
        """更新されたパラメータをセット"""
        self.sigma_1 = params["sigma_1"]
        self.sigma_2 = params["sigma_2"]
        self.rho = params["rho"]
        self.kappa = params["kappa"]
        self.alpha = params["alpha"]
        self.lambda_ = params["lambda"]
        self.mu = params["mu"]
        self.sigma_e = params["sigma_e"]
        self.alpha_hat = self.alpha - self.lambda_ / self.kappa

    def _prepare_measurement_equation(self):
        """Prepares the measurement equation components."""
        # intercept
        self.c = (
            self.rate
            - self.alpha_hat
            + 0.5 * self.sigma_2**2 / self.kappa**2
            - self.sigma_1 * self.sigma_2 * self.rho / self.kappa
        ) * self.forward_maturity
        +0.25 * self.sigma_2**2 * (
            1 - np.exp(-2 * self.kappa * self.forward_maturity)
        ) / self.kappa**3
        +(
            self.alpha_hat * self.kappa
            + self.sigma_1 * self.sigma_2 * self.rho
            - self.sigma_2**2 / self.kappa
        ) * (1 - np.exp(-self.kappa * self.forward_maturity)) / self.kappa**2

        # slope(Exogenous variable)
        self.X = np.array(
            [[1, -(1 - np.exp(-self.kappa * self.forward_maturity)) / self.kappa]]
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

    def read_csv(self, file_path="./data/input/WTI_Combined.csv"):
        """Reads a CSV file and returns spot and forward prices."""
        df = pd.read_csv(file_path)
        self.y_observed_list = np.log(df["Forward_Price"].values)

    def run_kalman_filter(self):
        """Runs the Kalman filter on the spot and forward prices."""
        self.num_of_observation = len(self.y_observed_list)
        ###
        # Initialize arrays to store results
        self.Beta_filtered_list = np.zeros((self.num_of_observation, 2, 1))
        self.Beta_cov_filtered_list = np.zeros((self.num_of_observation, 2, 2))
        self.Beta_pred_list = np.zeros((self.num_of_observation, 2, 1))
        self.Beta_cov_pred_list = np.zeros((self.num_of_observation, 2, 2))
        self.y_pred_list = np.zeros((self.num_of_observation, 1))
        self.pred_error_list = np.zeros((self.num_of_observation, 1))
        self.pred_error_var_list = np.zeros((self.num_of_observation, 1))

        ###
        # Initial state
        Beta_current = self.inital_beta  # [2x1]
        Beta_cov_current = self.initial_beta_cov  # [2x2]

        for t in range(
            self.num_of_observation
        ):  # t = 0, 1, 2, ..., self.num_of_observation-1
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

            # Calculate prediction error and variance for estimating parameters by Maximum Likelihood
            self.pred_error_list[t] = y_observed - y_pred  # [1x1], prediction error
            self.pred_error_var_list[t] = (
                self.X @ Beta_cov_pred @ self.X.T + self.sigma_e**2
            )  # [1x1], variance of prediction error

            # Update current state for next iteration
            Beta_current = Beta_filtered
            Beta_cov_current = Beta_cov_filtered
        return None

    def _predict_next_step(self, Beta_current, Beta_cov_current):
        """Predicts the next state and covariance."""
        Beta_pred = self.D + self.T @ Beta_current  # [2x1]
        Beta_cov_pred = self.T @ Beta_cov_current @ self.T.T + self.Q  # [2x2]
        y_pred = self.c + self.X @ Beta_pred  # [1x1]
        return Beta_pred, Beta_cov_pred, y_pred

    def _filter_current_step(self, Beta_pred, Beta_cov_pred, y_pred, y_observed):
        """Applies the Kalman filter to the given spot and forward prices."""
        Kalman_gain = (
            Beta_cov_pred
            @ self.X.T
            / (self.X @ Beta_cov_pred @ self.X.T + self.sigma_e**2)
        )  # [2x1]
        Beta_filtered = Beta_pred + Kalman_gain * (y_observed - y_pred)  # [2x1]
        Beta_cov_filtered = (np.eye(2) - Kalman_gain @ self.X) @ Beta_cov_pred  # [2x2]

        return Beta_filtered, Beta_cov_filtered

    def log_likelihood_wrapper(self, param_vector):
        """最適化用ラッパー関数（負の対数尤度を返す）"""
        # param_vector = [sigma_1, sigma_2, rho, kappa, alpha, lambda_, mu, sigma_e]
        p = {
            "sigma_1": abs(param_vector[0]),
            "sigma_2": abs(param_vector[1]),
            "rho": np.tanh(param_vector[2]),  # ensure -1 < rho < 1
            "kappa": abs(param_vector[3]),
            "alpha": param_vector[4],
            "lambda": param_vector[5],
            "mu": param_vector[6],
            "sigma_e": abs(param_vector[7]),
        }

        self._update_params(p)
        self._prepare_measurement_equation()
        self._prepare_state_equation()
        self.run_kalman_filter()

        ll = self._calculate_log_likelihood()
        return -ll  # minimize negative log-likelihood

    def _calculate_log_likelihood(self):
        """Calculates the log-likelihood of the observed data given the model."""
        log_likelihood = -0.5 * sum(
            np.log(self.pred_error_var_list)
            + self.pred_error_list**2 / self.pred_error_var_list
        )

        return log_likelihood

    def estimate_parameters_mle(self, initial_guess):
        """最尤法でパラメータを推定"""
        bounds = [
            (1e-5, None),  # sigma_1
            (1e-5, None),  # sigma_2
            (-2, 2),  # rho
            (1e-5, None),  # kappa
            (None, None),  # alpha
            (None, None),  # lambda
            (None, None),  # mu
            (1e-5, None),  # sigma_e
        ]

        result = minimize(
            self.log_likelihood_wrapper,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"disp": True},  # ✅ 追加：最適化の進捗を表示
        )

        if result.success:
            print("✅ 最尤推定が収束しました！")
            print("推定パラメータ:")
            print(result.x)
        else:
            print("⚠️ 最尤推定が収束しませんでした。")

        return result

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

    def export_results_to_csv(
        self, file_path="./data/output/Kalman_Filter_Results.csv"
    ):
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
        "sigma_e": 0.5,
    }

    ins_Kalman_filter = KalmanFilter(params, rate=0.05)
    ins_Kalman_filter.read_csv()

    # ✅ initial_guessを明示的に順番指定して渡す
    initial_guess = np.array(
        [
            params["sigma_1"],
            params["sigma_2"],
            params["rho"],
            params["kappa"],
            params["alpha"],
            params["lambda"],
            params["mu"],
            params["sigma_e"],
        ]
    )

    # result = ins_Kalman_filter.estimate_parameters_mle(initial_guess)

    ins_Kalman_filter.run_kalman_filter()
    ins_Kalman_filter.plot_results_of_prices()
    ins_Kalman_filter.plot_results_of_beta()
    ins_Kalman_filter.export_results_to_csv()
