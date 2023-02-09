import numpy as np
import torch
from operators.backwardStep import BackwardStep

class pFB_tich_prox_distr_algorithm:
    def __init__(self, game, x_0=None, dual_0=None, aux_0=None,
                 primal_stepsize=0.001, dual_stepsize=0.001, consensus_stepsize=0.001,
                 exponent_vanishing_precision=2, exponent_vanishing_selection=1, alpha_tich_regul = 1):
        self.game = game
        self.primal_stepsize = primal_stepsize
        self.dual_stepsize = dual_stepsize
        self.consensus_stepsize = consensus_stepsize
        N_agents = game.N_agents
        n = game.n_opt_variables
        m = game.n_shared_ineq_constr
        if x_0 is not None:
            self.x = x_0
        else:
            self.x = torch.zeros(N_agents, n, 1)
        if dual_0:
            self.dual = dual_0
        else:
            self.dual = torch.zeros(N_agents,m, 1)
        if aux_0:
            self.aux = aux_0
        else:
            self.aux = torch.zeros(N_agents,m, 1)
        self.x_pivot = self.x # used for tichonov regularization. It is updated at every outer iteration.
        Q = torch.zeros(N_agents, n, n) # Local cost is zero
        q = torch.zeros(N_agents, n, 1)
        self.projection = BackwardStep(Q, q, game.A_ineq_loc, game.b_ineq_loc, game.A_eq_loc, game.b_eq_loc,1)
        self.eps_tich = lambda t: t**(-1*exponent_vanishing_precision)
        self.weight_sel = lambda t: t**(-1*exponent_vanishing_selection)
        self.alpha_tich_regul = alpha_tich_regul
        self.outer_iter = 1

    def run_once(self):
        x = self.x
        dual = self.dual
        aux = self.aux
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        r = self.game.F(x)
        sel = self.weight_sel(self.outer_iter) * self.game.nabla_phi(x)
        tich = self.alpha_tich_regul * (x - self.x_pivot)
        x_new, status = self.projection(x - self.primal_stepsize * (r + sel + tich + torch.bmm(torch.transpose(A_i, 1, 2), self.dual ) ) )
        aux_new = aux - self.consensus_stepsize * self.game.K(dual)
        d = 2 * torch.bmm(A_i, x_new) - torch.bmm(A_i, x) - b_i
        # Dual update
        dual_new = torch.maximum(dual + self.dual_stepsize * ( d + self.game.K(2*aux_new - aux) + self.game.K(dual)), torch.zeros(dual.size()))
        if torch.norm(x-x_new) + torch.norm(dual-dual_new) + torch.norm(aux-aux_new) <= self.eps_tich(self.outer_iter):
            self.outer_iter = self.outer_iter + 1
            self.x_pivot = x_new
        self.x = x_new
        self.dual = dual_new
        self.aux = aux_new

    def get_state(self):
        residual = self.compute_residual()
        cost = self.game.J(self.x)
        sel = self.game.phi(self.x)
        return self.x, self.dual, self.aux, residual, cost, sel

    def compute_residual(self):
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        x = self.x
        x_transformed, status = self.projection(x-self.game.F(x) - torch.matmul(torch.transpose(A_i, 1, 2), self.dual ) )
        dual_transformed = torch.maximum(self.dual + torch.sum(torch.bmm(A_i, self.x) - b_i, 0), torch.zeros(self.dual.size()))
        residual = np.sqrt( ((x - x_transformed).norm())**2 + ((self.dual-dual_transformed).norm())**2 )
        return residual


class FBF_HSDM_distr_algorithm:
    def __init__(self, game, x_0=None, dual_0=None, aux_0=None,
                 primal_stepsize=0.001, dual_stepsize=0.001, consensus_stepsize=0.001,
                 exponent_vanishing_selection=1):
        self.game = game
        self.primal_stepsize = primal_stepsize
        self.dual_stepsize = dual_stepsize
        self.consensus_stepsize = consensus_stepsize
        N_agents = game.N_agents
        n = game.n_opt_variables
        m = game.n_shared_ineq_constr
        if x_0 is not None:
            self.x = x_0
        else:
            self.x = torch.zeros(N_agents, n, 1)
        if dual_0:
            self.dual = dual_0
        else:
            self.dual = torch.zeros(N_agents,m, 1)
        if aux_0:
            self.aux = aux_0
        else:
            self.aux = torch.zeros(N_agents,m, 1)
        Q = torch.zeros(N_agents, n, n) # Local cost is zero
        q = torch.zeros(N_agents, n, 1)
        self.projection = BackwardStep(Q, q, game.A_ineq_loc, game.b_ineq_loc, game.A_eq_loc, game.b_eq_loc,1)
        self.weight_sel = lambda t: t**(-1*exponent_vanishing_selection)
        self.iteration = 1

    def run_once(self):
        x = self.x
        dual = self.dual
        aux = self.aux
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        r = self.game.F(x)
        x_half_fbf, status = self.projection(x - self.primal_stepsize * (r + torch.bmm(torch.transpose(A_i, 1, 2), self.dual ) ) )
        d = torch.bmm(A_i, x) - b_i
        dual_half_fbf =torch.maximum(dual + self.dual_stepsize * (d + self.game.K(aux - dual)), torch.zeros(dual.size()))
        aux_half_fbf = aux - self.consensus_stepsize * self.game.K(dual)

        r = self.game.F(x_half_fbf) - self.game.F(x)
        x_fbf, status = self.projection(x_half_fbf - self.primal_stepsize * (r + torch.bmm(torch.transpose(A_i, 1, 2), dual_half_fbf - dual ) ) )
        d = torch.bmm(A_i, x_half_fbf - x)
        dual_fbf = dual_half_fbf + self.dual_stepsize * (d + self.game.K(aux_half_fbf - aux) - self.game.K(dual_half_fbf - dual))
        aux_half = aux_half_fbf - self.consensus_stepsize * self.game.K(dual_half_fbf - dual)

        # HSDM step
        x_new = x_fbf - self.weight_sel(self.iteration) * self.game.nabla_phi(x)
        self.iteration = self.iteration + 1
        dual_new = dual_fbf
        aux_new = aux_half

        self.x = x_new
        self.dual = dual_new
        self.aux = aux_new

    def get_state(self):
        residual = self.compute_residual()
        cost = self.game.J(self.x)
        sel = self.game.phi(self.x)
        return self.x, self.dual, self.aux, residual, cost, sel

    def compute_residual(self):
        A_i = self.game.A_ineq_shared
        b_i = self.game.b_ineq_shared
        x = self.x
        x_transformed, status = self.projection(x-self.game.F(x) - torch.matmul(torch.transpose(A_i, 1, 2), self.dual ) )
        dual_transformed = torch.maximum(self.dual + torch.sum(torch.bmm(A_i, self.x) - b_i, 0), torch.zeros(self.dual.size()))
        residual = np.sqrt( ((x - x_transformed).norm())**2 + ((self.dual-dual_transformed).norm())**2 )
        return residual