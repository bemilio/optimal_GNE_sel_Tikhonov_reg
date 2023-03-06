import torch
from scipy.signal import kaiserord, lfilter, firwin, freqz
from numpy import absolute
from numpy import random

class RandomLinearSetup:
    def __init__(self, N_agents, n_opt_var, x_max_shared, x_max_loc, comm_graph):
        self.n_opt_variables = n_opt_var
        self.Q,self.c = self.define_cost_functions(N_agents, n_opt_var, comm_graph)
        self.A_ineq_loc_const, self.b_ineq_loc_const,\
        self.A_eq_loc_const, self.b_eq_loc_const, \
        self.index_soft_constraints = self.define_local_constraints(N_agents, n_opt_var, x_max_loc)
        self.A_ineq_shared, self.b_ineq_shared = self.define_shared_constraints(N_agents, n_opt_var, x_max_shared)
        self.Q_sel_fun, self.c_sel_fun = self.define_selection_function(N_agents, n_opt_var, comm_graph)


    def define_selection_function(self, N_agents, n_opt_var, comm_graph):
        # Selection function:
        # Q is a blockmatrix N * N * n_x * n_x. Q_ij needs be = Q_ji' (but no checks are in place).
        # The gradient of the selection function is Qx + c (so the ith gradient is \sum_j Q_ij x_j + c_j).
        # The gradient can be computed with torch.sum(torch.matmul(Q, x),dim=1) + c
        # The selection function value is x'(Qx + c). Obtained via torch.bmm(x.transpose(1,2),torch.sum(torch.matmul(Q, x),dim=1)+  c).
        Q_sel_fun = torch.zeros(N_agents, N_agents, n_opt_var, n_opt_var)
        Q_rand = 0.1*torch.from_numpy(random.rand(N_agents*n_opt_var, N_agents*n_opt_var))
        Q_rand = (torch.transpose(Q_rand,0,1) + Q_rand)/2
        c_sel_fun = torch.from_numpy(10*random.randn(N_agents, n_opt_var, 1))
        for i in range(N_agents):
            for j in comm_graph.neighbors(i): # Allow cost only if agents can communicate
                Q_sel_fun[i,j,:,:] = Q_rand[i*n_opt_var:(i+1)*n_opt_var, j*n_opt_var:(j+1)*n_opt_var]
        min_eig = self.get_min_eig_game(Q_sel_fun)
        additive_constant = random.rand() # add an identity times this constant (to make the sel. fun. str. convex)
        for i in range(N_agents):
            Q_sel_fun[i,i,:,:] = Q_sel_fun[i,i,:,:] + (additive_constant - min_eig) * torch.eye(n_opt_var)
        min_eig_transformed = self.get_min_eig_game(Q_sel_fun)
        eps = 10 ** (-10)
        if min_eig_transformed < -1 * eps:
            raise Exception("The selection function is not monotone")
        return Q_sel_fun, c_sel_fun

    def define_cost_functions(self, N_agents, n_opt_var, comm_graph):
        Q = torch.zeros(N_agents, N_agents, n_opt_var, n_opt_var)
        # Take a row-stoch. matrix M and define Q = -M + I
        M = torch.zeros(N_agents * n_opt_var, N_agents * n_opt_var)
        for i in range(N_agents):
            for j in comm_graph.neighbors(i):
                M[i*n_opt_var:(i+1)*n_opt_var,j*n_opt_var:(j+1)*n_opt_var] = \
                    torch.from_numpy(random.rand(n_opt_var, n_opt_var))
        M = torch.div(M, torch.matmul(torch.sum(M, dim=1).unsqueeze(1), torch.ones(1, N_agents * n_opt_var)))
        M = torch.eye(N_agents * n_opt_var) - M
        c = torch.zeros(N_agents, n_opt_var, 1)
        for i in range(N_agents):
            Q[i, i, :, :] = M[i * n_opt_var:(i + 1) * n_opt_var, i * n_opt_var:(i + 1) * n_opt_var]
            for j in comm_graph.neighbors(i):  # Allow cost only if agents can communicate
                Q[i, j, :, :] = M[i * n_opt_var:(i + 1) * n_opt_var, j * n_opt_var:(j + 1) * n_opt_var]
        min_eig = self.get_min_eig_game(Q)
        eps = 10 ** (-10)
        if min_eig< -1*eps:
            raise Exception("The game is not monotone")
        if min_eig>eps:
            print("Warning! The game has only one solution")
        return Q,c

    def get_min_eig_game(self, M):
        N = M.size(0)
        n_x = M.size(2)
        M_full = torch.zeros(N*n_x, N*n_x)
        for i in range(N):
            for j in range(N):
                M_full[i*n_x:(i+1)*n_x, j*n_x:(j+1)*n_x] = M[i,j,:,:]
        eigval, eigvec = torch.linalg.eig(M_full)
        min_eig = torch.min(torch.real(eigval))
        return min_eig

    def define_local_constraints(self, N_agents, n_channels, P_max_local):
        # Sum power over channels = P_max
        # sum_k x_k = P_max
        # Num of local constr: Number of channels
        n_local_const_eq = 1
        A_eq_loc_const = torch.zeros(N_agents, n_local_const_eq, self.n_opt_variables)
        b_eq_loc_const = torch.zeros(N_agents, n_local_const_eq, 1)

        # Num of local inequality constraints: Power are positive (N_channels)
        n_local_const_ineq = 2 * n_channels
        A_ineq_loc_const = torch.zeros(N_agents, n_local_const_ineq, self.n_opt_variables)
        b_ineq_loc_const = torch.zeros(N_agents, n_local_const_ineq, 1)
        index_soft_constraints = torch.zeros(N_agents, 1)
        index_constr = 0
        # -x_max <=x <= x_max
        for i in range(N_agents):
            A_ineq_loc_const[i, 0:self.n_opt_variables, :] = torch.eye(n_channels)
            b_ineq_loc_const[i, 0:self.n_opt_variables, :] = P_max_local[i, :]
            A_ineq_loc_const[i, self.n_opt_variables:2*self.n_opt_variables, :] = -torch.eye(n_channels)
            b_ineq_loc_const[i, self.n_opt_variables:2*self.n_opt_variables, :] = P_max_local[i, :]
        return A_ineq_loc_const, b_ineq_loc_const, A_eq_loc_const, b_eq_loc_const, index_soft_constraints


    def define_shared_constraints(self, N_agents, n_channels, x_max_shared):
        n_shared_ineq_constr = n_channels
        A_ineq_shared = torch.zeros(N_agents, n_shared_ineq_constr, self.n_opt_variables)
        b_ineq_shared = torch.zeros(N_agents, n_shared_ineq_constr, 1)
        for i_agent in range(N_agents):
            A_ineq_shared[i_agent, :, :] = torch.eye(n_channels)
            b_ineq_shared[i_agent, :, 0] = x_max_shared/N_agents
        return A_ineq_shared, b_ineq_shared

