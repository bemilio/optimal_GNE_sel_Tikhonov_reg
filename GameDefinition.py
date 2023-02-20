import networkx
import torch
import numpy as np
import networkx as nx
from cmath import inf

torch.set_default_dtype(torch.float64)

class Game:
    # Q \in R^N*N*n_x*n_x, block matrix where Q[i,j,:,:]=Q_ij
    # Define distributed linearly coupled game where each agent has the same number of opt. variables
    # \sum x_i' Q_ij x_j + c_i'x_i
    # s.t. A_loc_i x_i <=  local_up_bound_i
    #      \sum_i A_shared_i x_i <=  b_shared
    #
    # (Facoltative): Selection function (cooperative)
    # Q_sel has the same structure as Q.
    # The selection function is x'Q_sel x + c_sel' x
    # Define distributed linearly coupled game where each agent has the same number of opt. variables
    def __init__(self, N, n_opt_var, communication_graph, Q, c, A_eq_loc, A_ineq_loc, A_shared, b_eq_loc, b_ineq_loc, b_shared,
                 Q_sel=None, c_sel=None, test=False):
        if test:
            N, n_opt_var, Q, c, Q_sel, c_sel, A_shared, b_shared, \
                A_eq_loc, A_ineq_loc, b_eq_loc, b_ineq_loc, communication_graph = self.setToTestGameSetup()
        self.N_agents = N
        index_x = 0
        self.n_opt_variables = n_opt_var
        # Local constraints
        self.A_eq_loc = A_eq_loc
        self.A_ineq_loc = A_ineq_loc
        self.b_eq_loc = b_eq_loc
        self.b_ineq_loc = b_ineq_loc
        # Shared constraints
        self.A_ineq_shared= A_shared
        self.b_ineq_shared = b_shared
        self.n_shared_ineq_constr = self.A_ineq_shared.size(1)
        # TODO: check if norm(Q_ij) = 0 if i,j are not connected

        # Define the (nonlinear) game mapping as a torch custom activation function
        self.F = self.GameMapping(Q, c)
        self.J = self.GameCost(Q, c)
        # Define the consensus operator
        self.K = self.Consensus(communication_graph, self.n_shared_ineq_constr)
        # Define the selection function gradient (can be zero)
        self.nabla_phi = self.SelFunGrad(Q_sel, c_sel)
        self.phi = self.SelFun(Q_sel, c_sel)

    class GameCost(torch.nn.Module):
        def __init__(self, Q, c):
            super().__init__()
            self.Q = Q
            self.c = c

        def forward(self, x):
            C = torch.matmul(self.Q, x) # C is a block matrix where C[i,j,:,:] = Q[i,j,:,:] x[j]
            cost = torch.bmm(x.transpose(1,2), torch.sum(C, 1) + self.c)
            return cost

    class GameMapping(torch.nn.Module):
        def __init__(self, Q, c, test=False):
            super().__init__()
            self.Q = Q
            self.c = c
            self.test=test

        def forward(self, x):
            C = torch.matmul(self.Q, x) # C is a block matrix where C[i,j,:,:] = Q[i,j,:,:] x[j]
            pgrad = torch.sum(C, 1) + self.c
            return pgrad

        def get_strMon_Lip_constants(self):
            # Return strong monotonicity and Lipschitz constant
            # Convert Q from block matrix to standard matrix #TODO: turn block matrix into separate class
            N = self.Q.size(0)
            n_x = N = self.Q.size(2)
            Q_mat = torch.zeros(N*n_x, N*n_x)
            for i in range(N):
                for j in range(N):
                    Q_mat[i*n_x:(i+1)*n_x, j*n_x:(j+1)*n_x] = self.Q[i,j,:,:]
            U,S,V = torch.linalg.svd(Q_mat)
            return torch.min(S).item(), torch.max(S).item()

    class Consensus(torch.nn.Module):
        def __init__(self, communication_graph, N_dual_variables):
            super().__init__()
            # Convert Laplacian matrix to sparse tensor
            L = networkx.laplacian_matrix(communication_graph).tocoo()
            values = L.data
            rows = L.row
            cols = L.col
            indices = np.vstack((rows, cols))
            L = L.tocsr()
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            L_torch = torch.zeros(L.shape[0],L.shape[1], N_dual_variables, N_dual_variables)
            for i in rows:
                for j in cols:
                    L_torch[i,j,:,:] = L[i,j] * torch.eye(N_dual_variables)
            # TODO: understand why sparse does not work
            # self.L = L_torch.to_sparse_coo()
            self.L = L_torch

        def forward(self, dual):
            return torch.sum(torch.matmul(self.L, dual), dim=1) # This applies the laplacian matrix to each of the dual variables



    class SelFunGrad(torch.nn.Module):
        def __init__(self, Q_sel, c_sel):
            super().__init__()
            if Q_sel is not None and c_sel is not None:
                self.Q = Q_sel
                self.c = c_sel
                self.is_active = True
            else:
                self.is_active = False

        def forward(self, x):
            if self.is_active:
                return torch.sum(torch.matmul(self.Q, x),dim=1) + self.c
            else:
                return 0*x

    class SelFun(torch.nn.Module):
        def __init__(self, Q_sel, c_sel):
            super().__init__()
            if Q_sel is not None and c_sel is not None:
                self.Q = Q_sel
                self.c = c_sel
                self.is_active = True
            else:
                self.is_active = False

        def forward(self, x):
            # Cost is .5*x'Qx + c'x,  where Q is a block-diagonal matrix stored in a tensor N*N*n*n. The i,j-th block is in Q[i,j,:,:].
            if self.is_active:
                return torch.sum(torch.bmm(x.transpose(1,2), .5*torch.sum(torch.matmul(self.Q, x),dim=1) + self.c))
            else:
                return torch.tensor(0)

        def get_strMon_Lip_constants(self):
            # Return strong monotonicity and Lipschitz constant
            # Convert Q from block matrix to standard matrix #TODO: turn block matrix into separate class
            N = self.Q.size(0)
            n_x = N = self.Q.size(2)
            Q_mat = torch.zeros(N*n_x, N*n_x)
            for i in range(N):
                for j in range(N):
                    Q_mat[i*n_x:(i+1)*n_x, j*n_x:(j+1)*n_x] = self.Q[i,j,:,:]
            U,S,V = torch.linalg.svd(Q_mat)
            return torch.min(S).item(), torch.max(S).item()

    def setToTestGameSetup(self):
        # Feasible points on x_1=x_2, South-West quadrant. Optimal point in -c_sel (that is, [-.5, -.5])
        Q = torch.zeros((2, 2, 1, 1))
        Q[0, 1, 0, 0] = -1
        Q[1, 0, 0, 0] = 1
        c = torch.zeros((2, 1, 1))
        Q_sel = torch.zeros((2, 2, 1, 1))
        Q_sel[0, 0, 0, 0] = 1
        Q_sel[1, 1, 0, 0] = 1
        c_sel = torch.zeros((2, 1, 1))
        c_sel[0, 0] = 0.5
        c_sel[1, 0] = 0.5
        A_shared = torch.zeros((2, 1, 1))
        A_shared[0, 0, 0] = -1
        A_shared[1, 0, 0] = 1
        b_shared = torch.zeros(1)
        A_eq_loc = torch.zeros(2,1,1)
        A_ineq_loc = torch.zeros(2,1,1)
        b_eq_loc = torch.zeros(2,1,1)
        b_ineq_loc = torch.zeros(2, 1, 1)
        n_opt_var = 1
        N=2
        communication_graph = nx.complete_graph(2)
        return N,n_opt_var,Q,c,Q_sel,c_sel,A_shared,b_shared, A_eq_loc, A_ineq_loc, b_eq_loc, b_ineq_loc, communication_graph


