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