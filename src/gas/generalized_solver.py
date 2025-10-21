import torch
from typing import List, Tuple, Optional, Any

from src.gas.solver_utils import NoiseScheduleVP

class GeneralizedSolver:
    """Generalized solver for diffusion ODE using theoretical guidance from DPM-Solver.
    
    Implements framework for solving diffusion ODE with support for multiple orders 
    and methods. Provides functionality for initializing coefficients, performing solver updates, 
    and calculating standard theoretical coefficients.
    """
    
    def __init__(self, model_fn: Any, noise_schedule: NoiseScheduleVP, use_theory_coef=True):
        """Initialize the Generalized Solver for Generalised Solver wrapper.
        
        Args:
            model_fn (model_wrapper.model_fn): A noise prediction model function, 
                which accepts the continuous-time input.
            noise_schedule (NoiseScheduleVP): Noise scheduler.
            use_theory_coef (bool): If is set to True, 
                then theoretical guidance is used. Default is True. 
        """
        self.model_class = model_fn
        self.noise_schedule = noise_schedule

        self.use_theory_coef = use_theory_coef

    def model_fn(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert the model to the data prediction model.
        
        Args:
            x (torch.Tensor): Model input.
            t (torch.Tensor): Timestep the model is evaluated at.
        
        Returns:
            torch.Tensor: Output of data prediction.
        """
        noise = self.model_class(x, t + self.t_couple[self.params_step])
        alpha_t, sigma_t = self.noise_schedule.marginal_alpha(t), self.noise_schedule.marginal_std(t)
        x0 = (x - sigma_t * noise) / alpha_t

        return x0

    def calc_vals(
        self, 
        t_prev_list: List[torch.Tensor], 
        t: float
    ) -> Tuple[
        torch.Tensor, 
        torch.Tensor, 
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor]], 
        Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]
    ]:
        """Calculate standard coefficients for DPM-Solver++ similar to the original implementation.
        Computes various coefficients used in the solver steps, including exponential terms,
        ratio values, and phi functions for different orders.

        Args:
            t_prev_list (List[torch.Tensor]): List of previous timesteps, 
                where the last is the most recent.
            t (float): Current timestep for which to compute coefficients.

        Returns:
            Tuple containing:
                a_ii (torch.Tensor): Ratio of standard deviations `sigma_{t} / sigma_{t_prev}`.
                alpha_t (torch.Tensor): Alpha coefficient at time `t`.
                (r0, r1) (Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]): 
                    Ratio coefficients for first and second previous steps (None if not available).
                (phi_1, phi_2, phi_3) (Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]):
                    Phi functions for different orders (None if not computable).
                
        Note:
            The function computes different sets of coefficients based on the number of available
            previous timesteps in t_prev_list:
                1 previous step: only `a_ii`, `alpha_t`, and `phi_1`;
                2 previous steps: adds `r0` and `phi_2`;
                3+ previous steps: adds `r1` and `phi_3`.
        """

        ns = self.noise_schedule
        t_prev_0 = t_prev_list[-1]
        lambda_prev_0, lambda_t = ns.marginal_lambda(t_prev_0), ns.marginal_lambda(t)
        h = lambda_t - lambda_prev_0
        log_alpha_t = ns.marginal_log_mean_coeff(t)
        sigma_prev_0, sigma_t = ns.marginal_std(t_prev_0), ns.marginal_std(t)
        alpha_t = torch.exp(log_alpha_t)

        phi_1 = torch.expm1(-h)
        a_ii = sigma_t / sigma_prev_0

        r0, r1 = None, None
        phi_2, phi_3 = None, None
        if len(t_prev_list) >= 2:
            lambda_prev_1 = ns.marginal_lambda(t_prev_list[-2])
            h_0 = lambda_prev_0 - lambda_prev_1
            r0 = h_0 / h

        if len(t_prev_list) >= 3:
            lambda_prev_2 = ns.marginal_lambda(t_prev_list[-3])
            h_1 = lambda_prev_1 - lambda_prev_2
            r1 = h_1 / h

            phi_2 = phi_1 / h + 1.
            phi_3 = phi_2 / h - 0.5

        return a_ii, alpha_t, (r0, r1), (phi_1, phi_2, phi_3)

    @torch.no_grad()
    def init_coefs(self, steps: int, order: int, timesteps: List[float]) -> None:
        """Initialize coefficients for LMS solvers similar to DPM-Solver.
        Sets up the necessary coefficients for LMS solvers based on the specified order and number of steps.

        Args:
            steps (int): Number of sampling steps.
            order (int): Solver order (determines the number of previous steps used).
            timesteps (List[float]): List of timesteps for which to compute coefficients.
        """
        assert order >= 1, f"order  = {order}"
        t_prev_list = [timesteps[0]]
        model_2 = torch.eye(2).double()
        model_prev_2 = [model_2[1], model_2[0]]
        model_3 = torch.eye(3).double()
        model_prev_3 = [model_3[2], model_3[1], model_3[0]]
        self.params_step = 0
        for step in range(1, steps + 1):
            t = timesteps[step]
            cur_order = min(step, order)
            c1, c2, c3 = 1., 0., 0.
            _, alpha_t, (_, _), (phi_1, _, _) = self.calc_vals(t_prev_list, t)
            C = - alpha_t * phi_1
            if cur_order == 2:
                model_res = self.second_update(0., model_prev_2, t_prev_list, t, [0] * cur_order) / C
                c1, c2 = model_res
            elif cur_order >= 3:
                model_prev_list = [torch.zeros(3)] * (cur_order - 3) + model_prev_3
                model_res = self.unbound_update(0., model_prev_list, t_prev_list, t, [0] * cur_order) / C
                c1, c2, c3 = model_res
            self.c1_diff[step - 1] = c1
            self.c2_diff[step - 1] = c2
            self.c3_diff[step - 1] = c3
            self.params_step += 1

            t_prev_list.append(t)
            t_prev_list = t_prev_list[-order:]
        self.params_step = 0

    def not_theory_update(
        self, 
        x: torch.Tensor, 
        model_prev_list: List[torch.Tensor], 
        t_prev_list: List[float], 
        t: float, 
        x_prev_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Perform a solver update without theoretical guidance for arbitrary order.
        Use S4S LMS + PC solver type from paper https://arxiv.org/abs/2502.17423.
        Updates the solution using previous model outputs and states.

        Args:
            x (torch.Tensor): Current solution.
            model_prev_list (List[torch.Tensor]): List of previous model outputs 
                at corresponding times in `t_prev_list`.
            t_prev_list (List[float]): List of previous time steps.
            t (float): Next timestep for the update step.
            order (int): Solver order (number of previous steps to use).
            x_prev_list (List[torch.Tensor]): List of previous solution estimates 
                at corresponding times in `t_prev_list`.

        Returns:
            torch.Tensor: Updated solution estimate at time t.
        """
        # Use S4S parametrization by the data prediction model from Appendix C.1

        a_ii, alpha_t, (_, _), (phi_1, _, _) = self.calc_vals(t_prev_list, t)
        C = - phi_1 * alpha_t
        x_t = a_ii * x

        # LMS
        prev_num = len(model_prev_list)
        for i in range(1, prev_num + 1):
            ci = self.__getattribute__(f"c{i}_diff")[self.params_step]
            x_t = x_t + C * ci * model_prev_list[-i]

        # LMS + PC
        # "A PC solver further refines this initial prediction, by subsequently applying Eq. (5) from S4S paper".
        x_t = a_ii * x_t
        for i in range(1, prev_num + 1):
            ai = self.__getattribute__(f"a{i}_diff")[self.params_step]
            x_t = x_t + C * ai * model_prev_list[-i]

        return x_t

    def first_update(
        self, x: torch.Tensor, 
        model_prev_list: List[torch.Tensor], 
        t_prev_list: List[float], 
        t: float, 
        x_prev_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single first-order solver update for diffusion ODE.
        Updates the solution using previous model outputs and states.

        Args:
            x (torch.Tensor): Current solution.
            model_prev_list (List[torch.Tensor]): List of previous model outputs 
                at corresponding times in `t_prev_list`.
            t_prev_list (List[float]): List of previous time steps.
            t (float): Next time for the update step.
            order (int): Solver order (number of previous steps to use).
            x_prev_list (List[torch.Tensor]): List of previous solution estimates 
                at corresponding times in `t_prev_list`.

        Returns:
            torch.Tensor: Updated solution estimate at time t.
        """
        assert len(model_prev_list) == len(t_prev_list)
        assert len(x_prev_list) == len(t_prev_list)
        model_prev_0 = model_prev_list[-1]
        a_ii, alpha_t, (_, _), (phi_1, _, _) = self.calc_vals(t_prev_list, t)

        a1 = a_ii + self.a1_diff[self.params_step]
        c1 = alpha_t * phi_1 + self.c1_diff[self.params_step]

        x_t = a1 * x - c1 * model_prev_0
        return x_t

    def second_update(
        self, 
        x: torch.Tensor, 
        model_prev_list: List[torch.Tensor], 
        t_prev_list: List[float], 
        t: float, 
        x_prev_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single second-order solver update for diffusion ODE.
        Updates the solution using previous model outputs and states.

        Args:
            x (torch.Tensor): Current solution.
            model_prev_list (List[torch.Tensor]): List of previous model outputs 
                at corresponding times in `t_prev_list`.
            t_prev_list (List[float]): List of previous time steps.
            t (float): Next timestep for the update step.
            order (int): Solver order (number of previous steps to use).
            x_prev_list (List[torch.Tensor]): List of previous solution estimates 
                at corresponding times in `t_prev_list`.

        Returns:
            torch.Tensor: Updated solution estimate at time t.
        """
        assert len(model_prev_list) == len(t_prev_list)
        assert len(x_prev_list) == len(t_prev_list)
        model_prev_1, model_prev_0 = model_prev_list[-2], model_prev_list[-1]
        a_ii, alpha_t, (r0, _), (phi_1, _, _) = self.calc_vals(t_prev_list, t)

        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)

        a1 = a_ii + self.a1_diff[self.params_step]
        c1 = alpha_t * phi_1 + self.c1_diff[self.params_step]
        c2 = 0.5 * (alpha_t * phi_1) + self.c2_diff[self.params_step]

        x_t = a1 * x - c1 * model_prev_0 - c2 * D1_0 
        x_t = x_t + self.a2_diff[self.params_step] * x_prev_list[-2]
        return x_t

    def unbound_update(
        self, 
        x: torch.Tensor, 
        model_prev_list: List[torch.Tensor], 
        t_prev_list: List[float], 
        t: float, 
        x_prev_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single high-order (≥3) solver update for diffusion ODE.
        Updates the solution using previous model outputs and states.

        Args:
            x (torch.Tensor): Current solution.
            model_prev_list (List[torch.Tensor]): List of previous model outputs 
                at corresponding times in `t_prev_list`.
            t_prev_list (List[float]): List of previous time steps.
            t (float): Next timestep for the update step.
            order (int): Solver order (number of previous steps to use).
            x_prev_list (List[torch.Tensor]): List of previous solution estimates 
                at corresponding times in `t_prev_list`.

        Returns:
            torch.Tensor: Updated solution estimate at time t.
        """
        assert len(model_prev_list) == len(t_prev_list)
        assert len(x_prev_list) == len(t_prev_list)
        model_prev_2, model_prev_1, model_prev_0 = model_prev_list[-3], model_prev_list[-2], model_prev_list[-1]
        a_ii, alpha_t, (r0, r1), (phi_1, phi_2, phi_3) = self.calc_vals(t_prev_list, t)

        D1_0 = (1. / r0) * (model_prev_0 - model_prev_1)
        D1_1 = (1. / r1) * (model_prev_1 - model_prev_2)
        D1 = D1_0 + (r0 / (r0 + r1)) * (D1_0 - D1_1)
        D2 = (1. / (r0 + r1)) * (D1_0 - D1_1)

        a1 = a_ii + self.a1_diff[self.params_step]
        c1 = alpha_t * phi_1 + self.c1_diff[self.params_step]
        c2 = alpha_t * phi_2 + self.c2_diff[self.params_step]
        c3 = alpha_t * phi_3 + self.c3_diff[self.params_step]

        x_t = a1 * x - c1 * model_prev_0 + c2 * D1 - c3 * D2
        
        prev_num = len(model_prev_list)
        for i in range(4, prev_num + 1):
            x_t = x_t + self.__getattribute__(f"c{i}_diff")[self.params_step] * model_prev_list[-i]

        for i in range(2, prev_num + 1):
            x_t = x_t + self.__getattribute__(f"a{i}_diff")[self.params_step] * x_prev_list[-i]
        return x_t

    def solver_update(
        self, 
        x: torch.Tensor, 
        model_prev_list: List[torch.Tensor], 
        t_prev_list: List[float], 
        t: float, 
        order: int, 
        x_prev_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """Perform a single step of the solver update for diffusion ODE.
        Updates the solution using previous model outputs and states.

        Args:
            x (torch.Tensor): Current solution.
            model_prev_list (List[torch.Tensor]): List of previous model outputs 
                at corresponding times in `t_prev_list`.
            t_prev_list (List[float]): List of previous time steps.
            t (float): Next timestep for the update step.
            order (int): Solver order (number of previous steps to use).
            x_prev_list (List[torch.Tensor]): List of previous solution estimates 
                at corresponding times in `t_prev_list`.

        Returns:
            torch.Tensor: Updated solution estimate at time t.
        """
        
        assert order >= 1, f"Solver order must be >= 1, got {order}"
        if self.use_theory_coef:
            if order == 1:
                x_t = self.first_update(x, model_prev_list, t_prev_list, t, x_prev_list)
            elif order == 2:
                x_t = self.second_update(x, model_prev_list, t_prev_list, t, x_prev_list)
            else:
                x_t = self.unbound_update(x, model_prev_list, t_prev_list, t, x_prev_list)
        else:
            x_t = self.not_theory_update(x, model_prev_list, t_prev_list, t, x_prev_list)

        self.params_step = self.params_step + 1
        return x_t

    def sample(
        self, 
        x: torch.Tensor, 
        steps: int, 
        order: int, 
        **kwargs
    ) -> torch.Tensor:
        """Sample from diffuision ODE.
        Use x like initial value and NFE=steps.

        Args:
            x (torch.Tensor): The initial value for sampling.
            steps (int): Number of sampling steps.
            order (int): Solver order (determines the number of previous steps used).

        Returns:
            torch.Tensor: The approximated solution.        
        """
        assert steps >= order
        self.params_step = 0

        # Init the initial values.
        timesteps = self.get_time_steps()
        assert timesteps.shape[0] - 1 == steps, f"timestep.shape = {timesteps.shape}"
        t = timesteps[0]
        t_prev_list = [t]
        model_prev_list = [self.model_fn(x, t)]
        x_prev_list = [x]

        for step in range(1, steps + 1):
            t = timesteps[step]
            cur_order = min(step, order)
            x = self.solver_update(x, model_prev_list, t_prev_list, t, order=cur_order, x_prev_list=x_prev_list)

            # We do not need to evaluate the final model value.
            if step == steps:
                break

            x_prev_list.append(x)
            t_prev_list.append(t)
            model_prev_list.append(self.model_fn(x, t))

            x_prev_list = x_prev_list[-order:]
            t_prev_list = t_prev_list[-order:]
            model_prev_list = model_prev_list[-order:]

        return x
