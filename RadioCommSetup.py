import torch
from scipy.signal import kaiserord, lfilter, firwin, freqz
from numpy import absolute

class RadioCommSetup:
    def __init__(self, N_agents, n_channels, P_max_shared, P_max_local, taps_filter, comm_graph):
        self.n_opt_variables = n_channels
        self.Q,self.c = self.define_cost_functions(N_agents,taps_filter, n_channels, comm_graph)
        self.A_ineq_loc_const, self.b_ineq_loc_const,\
        self.A_eq_loc_const, self.b_eq_loc_const, \
        self.index_soft_constraints = self.define_local_constraints(N_agents, n_channels, P_max_local)
        self.A_ineq_shared, self.b_ineq_shared = self.define_shared_constraints(N_agents, n_channels, P_max_shared)
        M_interference = self.Q
        self.Q_sel_fun, self.c_sel_fun = self.define_selection_function(N_agents, n_channels, M_interference)


    def define_selection_function(self, N_agents, n_channels, M_interference):
        # Selection function:
        # Q is a blockmatrix N * N * n_x * n_x. Q_ij needs be = Q_ji' (but no checks are in place).
        # The gradient of the selection function is Qx + c (so the ith gradient is \sum_j Q_ij x_j + c_j).
        # The gradient can be computed with torch.sum(torch.matmul(Q, x),dim=1) + c
        # The selection function value is x'(Qx + c). Obtained via torch.bmm(x.transpose(1,2),torch.sum(torch.matmul(Q, x),dim=1)+  c).
        Q_sel_fun = torch.zeros(N_agents, N_agents, n_channels, n_channels)
        c_sel_fun = torch.zeros(N_agents, n_channels, 1)
        for i in range(N_agents):
            # See scutari 2012, eq. (15)
            I_others = list(range(N_agents))
            I_others.remove(i)
            c_sel_fun[i,:,:] = torch.sum(torch.vstack([torch.sum(M_interference[j,i,:,:], dim=1) for j in I_others ]), dim=0).unsqueeze(1)
        return Q_sel_fun, c_sel_fun

    def define_cost_functions(self, N, taps_filter, n_channels, comm_graph):
        M = torch.zeros(N, N, n_channels, n_channels)
        for i in range(N):
            M[i, i, :, :] = torch.eye(n_channels)
            for j in comm_graph.neighbors(i): # Allow cross-interference only if agents can communicate
                w, h = freqz(taps_filter[i, j, :], worN=n_channels)
                M[i, j, :, :] = torch.diag(torch.from_numpy(absolute(h)))
        c = torch.zeros(N, n_channels, 1)
        for i in range(N):
            c[i, :, :] = torch.div(torch.ones(n_channels,1), torch.diag(M[i,i,:,:]).unsqueeze(1))   # c_i = sigma_i/|H_ii(k)|
        # Make matrix positive semidefinite
        min_eig = self.get_min_eig_game(M)
        M_trans = M
        for i in range(N):
            M_trans[i, i, :, :] = M[i, i, :, :]-min_eig * torch.eye(n_channels)
        M_trans = M_trans / (1-min_eig)
        min_eig_transformed = self.get_min_eig_game(M_trans)
        eps = 10 ** (-10)
        if min_eig_transformed< -1*eps:
            raise Exception("The game is not monotone")
        if min_eig_transformed>eps:
            print("Warning! The game has only one solution")
        return M_trans,c

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
        for i in range(N_agents):
            A_eq_loc_const[i,:,:] = torch.ones(1,n_channels)
            b_eq_loc_const[i, :, :] = P_max_local[i,:]

        # Num of local inequality constraints: Power are positive (N_channels)
        n_local_const_ineq = n_channels
        A_ineq_loc_const = torch.zeros(N_agents, n_local_const_ineq, self.n_opt_variables)
        b_ineq_loc_const = torch.zeros(N_agents, n_local_const_ineq, 1)
        index_soft_constraints = torch.zeros(N_agents, 1)
        for i in range(N_agents):
            A_ineq_loc_const[i,:,:] = -1*torch.eye(n_channels)
        return A_ineq_loc_const, b_ineq_loc_const, A_eq_loc_const, b_eq_loc_const, index_soft_constraints


    def define_shared_constraints(self, N_agents, n_channels, P_max_shared):
        n_shared_ineq_constr = n_channels
        A_ineq_shared = torch.zeros(N_agents, n_shared_ineq_constr, self.n_opt_variables)
        b_ineq_shared = torch.zeros(N_agents, n_shared_ineq_constr, 1)
        for i_agent in range(N_agents):
            A_ineq_shared[i_agent, :, :] = torch.eye(n_channels)
            b_ineq_shared[i_agent, :, 0] = P_max_shared/N_agents
        return A_ineq_shared, b_ineq_shared

