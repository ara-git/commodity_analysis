import numpy as np
import pandas as pd
from scipy.optimize import minimize


class KalmanFilter:
    def __init__(
        self,
        params,
        T=1,
        dt=1 / 252,
        initial_beta=[1, 1],
        initial_beta_cov=np.eye(2),
    ):
        self.params = params
        self.forward_maturity = T  # maturity of forward contract
        self.dt = dt  # Daily time step
        self.initial_beta = np.array(initial_beta).reshape(2, 1)  # [2x1]
        self.initial_beta_cov = initial_beta_cov  # [2x2]

        # Prepare measurement and state equations
        self._update_params(params)

    def _update_params(self, params):
        """更新されたパラメータをセット"""
        self.sigma_1 = params["sigma_1"]
        self.sigma_2 = params["sigma_2"]
        self.sigma_3 = params["sigma_3"]
        self.rho_1 = params["rho_1"]
        self.rho_2 = params["rho_2"]
        self.rho_3 = params["rho_3"]
        self.kappa = params["kappa"]
        self.alpha = params["alpha"]
        self.lambda_ = params["lambda"]
        self.mu = params["mu"]
        self.a = params["a"]
        self.m = params["m"]
        self.sigma_e = params["sigma_e"]
        self.alpha_hat = self.alpha - self.lambda_ / self.kappa

    def _prepare_measurement_equation(self, r_t):
        """Prepares the measurement equation components."""
        # intercept
        tmp = (
            (self.kappa * self.alpha_hat + self.sigma_1 * self.sigma_2 * self.rho_1)
            * (
                1
                - np.exp(-self.kappa * self.forward_maturity)
                - self.kappa * self.forward_maturity
            )
            / self.kappa**2
        )
        -(
            self.sigma_2**2
            * (
                4 * (1 - np.exp(-self.kappa * self.forward_maturity))
                - (1 - np.exp(-2 * self.kappa * self.forward_maturity))
                - 2 * self.kappa * self.forward_maturity
            )
        ) / (4 * self.kappa**3)
        -(
            (self.a * self.m + self.sigma_1 * self.sigma_3 * self.rho_3)
            * (
                1
                - np.exp(-self.a * self.forward_maturity)
                - self.a * self.forward_maturity
            )
            / self.a**2
        )
        -(
            self.sigma_3**2
            * (
                4 * (1 - np.exp(-self.a * self.forward_maturity))
                - (1 - np.exp(-2 * self.a * self.forward_maturity))
                - 2 * self.a * self.forward_maturity
            )
            / (4 * self.a**3)
        )
        +self.sigma_2 * self.sigma_3 * self.rho_2 * (
            (
                (1 - np.exp(-self.kappa * self.forward_maturity))
                + (1 - np.exp(-self.a * self.forward_maturity))
                - (1 - np.exp(-(self.kappa + self.a) * self.forward_maturity))
            )
            / (self.kappa * self.a * (self.kappa + self.a))
            + (
                self.kappa**2 * (1 - np.exp(-self.a * self.forward_maturity))
                + self.a**2 * (1 - np.exp(-self.kappa * self.forward_maturity))
                - self.kappa * self.a**2 * self.forward_maturity
                - self.a * self.kappa**2 * self.forward_maturity
            )
            / (self.kappa**2 * self.a**2 * (self.kappa + self.a))
        )
        self.c = (r_t * (1 - np.exp(-self.a * self.forward_maturity)) / self.a) + tmp

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
                    self.rho_1 * self.sigma_1 * self.sigma_2 * self.dt,
                ],
                [
                    self.rho_1 * self.sigma_1 * self.sigma_2 * self.dt,
                    self.sigma_2**2 * self.dt,
                ],
            ]
        )  # [2x2]

    def read_csv(
        self,
        com_file_path="./data/input/WTI_Combined.csv",
        rate_file_path="./data/input/US_3M_rates.csv",
    ):
        """Reads a CSV file and returns spot and forward prices."""
        com_df = pd.read_csv(com_file_path)
        rate_df = pd.read_csv(rate_file_path)

        merged_df = pd.merge(com_df, rate_df, on="Date", how="left").dropna()
        self.Date_list = pd.to_datetime(merged_df["Date"]).values  # 日付情報を保存
        self.y_observed_list = np.log(merged_df["Forward_Price"].values)
        self.r_list = merged_df["rate"].values

    def run_kalman_filter(self):
        """Runs the Kalman filter on the spot and forward prices."""
        # Update measurement equation with the latest interest rates

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

        # update state equation
        self._prepare_state_equation()

        ###
        # Initial state
        Beta_current = self.initial_beta  # [2x1]
        Beta_cov_current = self.initial_beta_cov  # [2x2]

        for t in range(
            self.num_of_observation
        ):  # t = 0, 1, 2, ..., self.num_of_observation-1
            # Prediction step
            self._prepare_measurement_equation(
                self.r_list[t]
            )  # Update measurement equation with the latest interest rates
            Beta_pred, Beta_cov_pred, y_pred = self._predict_next_step(
                Beta_current, Beta_cov_current
            )  # [2x1], [2x2], [1x1]

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
            "sigma_3": abs(param_vector[2]),
            "rho_1": param_vector[3],  # ensure -1 < rho < 1
            "rho_2": param_vector[4],  # ensure -1 < rho < 1
            "rho_3": param_vector[5],  # ensure -1 < rho < 1
            "kappa": abs(param_vector[6]),
            "alpha": param_vector[7],
            "lambda": param_vector[8],
            "mu": param_vector[9],
            "a": abs(param_vector[10]),
            "m": param_vector[11],
            "sigma_e": abs(param_vector[12]),
        }

        self._update_params(p)
        self.run_kalman_filter()

        ll = self._calculate_log_likelihood()
        # print("ll", ll)  # 対数尤度を表示
        return -ll  # minimize negative log-likelihood

    def _calculate_log_likelihood(self):
        """Calculates the log-likelihood of the observed data given the model."""
        log_likelihood = -0.5 * sum(
            np.log(self.pred_error_var_list)
            + self.pred_error_list**2 / self.pred_error_var_list
        )

        return log_likelihood

    def estimate_parameters_mle(self):
        """最尤法でパラメータを推定"""
        bounds = [
            (1e-5, None),  # sigma_1
            (1e-5, None),  # sigma_2
            (1e-5, None),  # sigma_3
            (-1, 1),  # rho1
            (-1, 1),  # rho2
            (-1, 1),  # rho3
            (1e-5, None),  # kappa
            (None, None),  # alpha
            (None, None),  # lambda
            (None, None),  # mu
            (None, None),  # a
            (None, None),  # m
            (1e-5, None),  # sigma_e
        ]

        initial_guess = np.array(
            [
                self.params["sigma_1"],
                self.params["sigma_2"],
                self.params["sigma_3"],
                self.params["rho_1"],
                self.params["rho_2"],
                self.params["rho_3"],
                self.params["kappa"],
                self.params["alpha"],
                self.params["lambda"],
                self.params["mu"],
                self.params["a"],
                self.params["m"],
                self.params["sigma_e"],
            ]
        )

        result = minimize(
            self.log_likelihood_wrapper,
            initial_guess,
            method="L-BFGS-B",
            bounds=bounds,
            options={"disp": True, "gtol": 1e-2, "ftol": 1e-9, "maxiter": 10},
        )

        # result.x を常に反映
        p_opt = {
            "sigma_1": abs(result.x[0]),
            "sigma_2": abs(result.x[1]),
            "sigma_3": abs(result.x[2]),
            "rho_1": result.x[3],
            "rho_2": result.x[4],
            "rho_3": result.x[5],
            "kappa": abs(result.x[6]),
            "alpha": result.x[7],
            "lambda": result.x[8],
            "mu": result.x[9],
            "a": abs(result.x[10]),
            "m": result.x[11],
            "sigma_e": abs(result.x[12]),
        }
        self._update_params(p_opt)

        if result.success:
            print("✅ 最尤推定が収束しました！")
        else:
            print("⚠️ 最尤推定が収束しませんでした。途中のパラメータを反映します。")

        print("推定パラメータ:")
        print(result.x)

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
        results_df.insert(0, "Date", self.Date_list)  # 左端に日付情報を追加
        results_df.to_csv(file_path, index=False)
        print(f"Kalman filter results exported to {file_path}")

    def compare_and_export_parameters(
        self,
        initial_params,
        estimated_params,
        file_path="./data/output/Parameter_Estimation_Results.xlsx",
    ):
        """
        初期パラメータと推定パラメータを比較し、Excelに出力
        """
        comparison_df = pd.DataFrame(
            {
                "Parameter": list(initial_params.keys()),
                "Initial_Value": [initial_params[k] for k in initial_params.keys()],
                "Estimated_Value": [
                    estimated_params[k] for k in estimated_params.keys()
                ],
            }
        )
        comparison_df["Difference"] = (
            comparison_df["Estimated_Value"] - comparison_df["Initial_Value"]
        )

        print("\n📊 パラメータ比較結果：")
        print(comparison_df.to_string(index=False))

        # Excel出力
        comparison_df.to_excel(file_path, index=False)
        print(f"\n✅ パラメータ比較結果を {file_path} に出力しました。")

        return comparison_df


if __name__ == "__main__":
    estimate = True  # 最尤推定を実行するかどうか
    # 初期パラメータ
    params = {
        "sigma_1": 0.344,
        "sigma_2": 0.372,
        "sigma_3": 0.0081,
        "rho_1": 0.915,
        "rho_2": -0.0039,
        "rho_3": -0.0293,
        "kappa": 1.314,
        "alpha": 0.249,
        "lambda": 0.353,
        "mu": 0.315,
        "a": 0.2,
        "m": 1.0,  # not described in the paper
        "sigma_e": 0.5,  # not described in the paper
    }  # Schwartz (1997), Table IXの推定値

    ins_Kalman_filter = KalmanFilter(params)
    ins_Kalman_filter.read_csv(
        com_file_path="./data/input/WTI_Combined.csv",
        rate_file_path="./data/input/US_3M_rates.csv",
    )

    if estimate:
        print("最尤推定を実行します...")
        ins_Kalman_filter.run_kalman_filter()
        result = ins_Kalman_filter.estimate_parameters_mle()

        # 推定後のパラメータを反映
        estimated_params = {
            "sigma_1": abs(result.x[0]),
            "sigma_2": abs(result.x[1]),
            "sigma_3": abs(result.x[2]),
            "rho_1": result.x[3],
            "rho_2": result.x[4],
            "rho_3": result.x[5],
            "kappa": abs(result.x[6]),
            "alpha": result.x[7],
            "lambda": result.x[8],
            "mu": result.x[9],
            "a": abs(result.x[10]),
            "m": result.x[11],
            "sigma_e": abs(result.x[12]),
        }

        # 比較とExcel出力
        ins_Kalman_filter.compare_and_export_parameters(params, estimated_params)

        # 最新のパラメータで再実行
        params.update(estimated_params)

    # カルマンフィルタ再実行
    ins_Kalman_filter = KalmanFilter(params)
    ins_Kalman_filter.read_csv(
        com_file_path="./data/input/WTI_Combined.csv",
        rate_file_path="./data/input/US_3M_rates.csv",
    )

    ins_Kalman_filter.run_kalman_filter()
    ins_Kalman_filter.plot_results_of_prices()
    ins_Kalman_filter.plot_results_of_beta()
    ins_Kalman_filter.export_results_to_csv()
