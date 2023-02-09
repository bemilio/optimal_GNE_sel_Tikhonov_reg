import numpy as np
import networkx as nx
import torch
import pickle
from GNE_distrib_selection import pFB_tich_prox_distr_algorithm, FBF_HSDM_distr_algorithm
from GNE_distrib import FBF_distr_algorithm
from RadioCommSetup import RadioCommSetup
from GameDefinition import Game
import time
import logging


def set_stepsizes(N, road_graph, A_ineq_shared, xi, algorithm='FRB'):
    theta = 0
    c = road_graph.edges[(0, 1)]['capacity']
    tau = road_graph.edges[(0, 1)]['travel_time']
    zeta = road_graph.edges[(0, 1)]['uncontrolled_traffic']
    k = 0.15 * tau / (c**xi)
    L = (2*k/N)* ((N+1) + (1 + zeta)**(xi-1) + (xi-1 * (1+zeta)**(xi-2)) )
    if algorithm == 'FRB':
        delta = 2*L / (1-3*theta)
        eigval, eigvec = torch.linalg.eig(torch.bmm(A_ineq_shared, torch.transpose(A_ineq_shared, 1, 2)))
        eigval = torch.real(eigval)
        alpha = 0.5/((torch.max(torch.max(eigval, 1)[0])) + delta)
        beta = N * 0.5/(torch.sum(torch.max(eigval, 1)[0]) + delta)
    if algorithm == 'FBF':
        eigval, eigvec = torch.linalg.eig(torch.sum(torch.bmm(A_ineq_shared, torch.transpose(A_ineq_shared, 1, 2)), 0)  )
        eigval = torch.real(eigval)
        alpha = 0.5/(L+torch.max(eigval))
        beta = 0.5/(L+torch.max(eigval))
    return (alpha.item(), beta.item(), theta)



if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)

    np.random.seed(1)
    N_iter=100
    N_it_per_residual_computation = 100
    N_agents = 25
    n_channels = 10
    n_neighbors = 6 # for simplicity, each agent has the same number of neighbours. This is only used to create the communication graph (but i's not needed otherwise)
    P_max_shared = 10 # Max power that can go on a channel
    P_max_local = torch.ones(N_agents,1) # Max power that each agent can output (cumulative on all channels)
    N_random_tests = 1
    n_taps = 10 # order of the filters
    is_connected = False
    comm_graph = nx.random_regular_graph(n_neighbors, N_agents)
    for test in range(N_random_tests):
        ##########################################
        #        Test case creation              #
        ##########################################
        comm_graph = nx.random_regular_graph(n_neighbors, N_agents)
        while not nx.is_connected(comm_graph):
            n_neighbors = n_neighbors+1
            comm_graph = nx.random_regular_graph(n_neighbors, N_agents)

        taps_filter = (1/n_taps) * torch.from_numpy(np.random.rand(N_agents, N_agents, n_taps))
        # Make symmetric with respect to agents
        taps_filter = 0.5*(taps_filter + torch.transpose(taps_filter, 0,1))
        game_params = RadioCommSetup(N_agents, n_channels, P_max_shared, P_max_local, taps_filter, comm_graph)
        print("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        logging.info("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        ##########################################
        #             Game inizialization        #
        ##########################################
        game = Game(N_agents, n_channels, comm_graph, 0*game_params.Q , 0*game_params.c,
                    game_params.A_eq_loc_const, game_params.A_ineq_loc_const, game_params.A_ineq_shared,
                    game_params.b_eq_loc_const, game_params.b_ineq_loc_const, game_params.b_ineq_shared,
                    Q_sel=game_params.Q_sel_fun, c_sel=game_params.c_sel_fun)
        if test == 0:

            print("The game has " + str(N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_ineq_constr) + " shared ineq. constraints" )
            logging.info("The game has " + str(N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_ineq_constr) + " shared ineq. constraints" )
            ##########################################
            #   Variables storage inizialization     #
            ##########################################
            # pFB-Tichonov
            x_store_tich = torch.zeros(N_random_tests, N_agents, game.n_opt_variables)
            dual_store_tich = torch.zeros(N_random_tests, N_agents, game.n_shared_ineq_constr)
            aux_store_tich = torch.zeros(N_random_tests, N_agents, game.n_shared_ineq_constr)
            residual_store_tich = torch.zeros(N_random_tests, N_iter // N_it_per_residual_computation)
            local_cost_store_tich = torch.zeros(N_random_tests, N_agents, N_iter // N_it_per_residual_computation)
            sel_func_store_tich = torch.zeros(N_random_tests, N_iter // N_it_per_residual_computation)
            # FBF-HSDM
            x_store_hsdm = torch.zeros(N_random_tests, N_agents, game.n_opt_variables)
            dual_store_hsdm = torch.zeros(N_random_tests, N_agents, game.n_shared_ineq_constr)
            aux_store_hsdm = torch.zeros(N_random_tests, N_agents, game.n_shared_ineq_constr)
            residual_store_hsdm = torch.zeros(N_random_tests, N_iter // N_it_per_residual_computation)
            local_cost_store_hsdm = torch.zeros(N_random_tests, N_agents, N_iter // N_it_per_residual_computation)
            sel_func_store_hsdm = torch.zeros(N_random_tests, N_iter // N_it_per_residual_computation)
            # FBF
            x_store_std = torch.zeros(N_random_tests, N_agents, game.n_opt_variables)
            dual_store_std = torch.zeros(N_random_tests, N_agents, game.n_shared_ineq_constr)
            aux_store_std = torch.zeros(N_random_tests, N_agents, game.n_shared_ineq_constr)
            residual_store_std = torch.zeros(N_random_tests, N_iter // N_it_per_residual_computation)
            local_cost_store_std = torch.zeros(N_random_tests, N_agents, N_iter // N_it_per_residual_computation)
            sel_func_store_std = torch.zeros(N_random_tests, N_iter // N_it_per_residual_computation)
            # [alpha, beta, theta] = set_stepsizes(N_agents, game.A_ineq_shared, xi, algorithm='FRB')
            ##########################################
            #          Algorithm inizialization      #
            ##########################################
            alg_tich = pFB_tich_prox_distr_algorithm(game, primal_stepsize=0.1, dual_stepsize=0.1,
                                                     consensus_stepsize=0.1,
                                                     exponent_vanishing_precision=2, exponent_vanishing_selection=1,
                                                     alpha_tich_regul=1)
            alg_hsdm = pFB_tich_prox_distr_algorithm(game, primal_stepsize=0.1, dual_stepsize=0.1,
                                                     consensus_stepsize=0.1, exponent_vanishing_selection=1)
            alg_std = FBF_distr_algorithm(game, primal_stepsize=0.1, dual_stepsize=0.1, consensus_stepsize=0.1)
        index_store = 0
        avg_time_per_it_tich = 0
        avg_time_per_it_hsdm = 0
        avg_time_per_it_std = 0
        for k in range(N_iter):
            ##########################################
            #             Algorithm run              #
            ##########################################
            start_time = time.time()
            alg_tich.run_once()
            end_time = time.time()
            avg_time_per_it_tich = (avg_time_per_it_tich * k + (end_time - start_time)) / (k + 1)
            start_time = time.time()
            alg_hsdm.run_once()
            end_time = time.time()
            avg_time_per_it_hsdm = (avg_time_per_it_hsdm * k + (end_time - start_time)) / (k + 1)
            start_time = time.time()
            alg_std.run_once()
            end_time = time.time()
            avg_time_per_it_std = (avg_time_per_it_std * k + (end_time - start_time)) / (k + 1)
            if k % N_it_per_residual_computation == 0:
                ##########################################
                #          Save performance metrics      #
                ##########################################
                # Tichonov
                x, d, a, r, c, s  = alg_tich.get_state()
                residual_store_tich[test, index_store] = r
                sel_func_store_tich[test, index_store] = s
                print("Tichonov: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " +  str(s.item()) +  " Average time: " + str(avg_time_per_it_tich))
                logging.info("Tichonov: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " +  str(s.item()) +" Average time: " + str(avg_time_per_it_tich))
                # HSDM
                x, d, a, r, c, s = alg_hsdm.get_state()
                residual_store_hsdm[test, index_store] = r
                sel_func_store_hsdm[test, index_store] = s
                print("HSDM: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " + str(
                    s.item()) + " Average time: " + str(avg_time_per_it_hsdm))
                logging.info("HSDM: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " + str(
                    s.item()) + " Average time: " + str(avg_time_per_it_hsdm))
                # FBF
                x, d, a, r, c = alg_std.get_state()
                residual_store_std[test, index_store] = r
                sel_func_store_std[test, index_store] = game.phi(x)
                print("FBF: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " + str(
                    s.item()) + " Average time: " + str(avg_time_per_it_std))
                logging.info("FBF: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " + str(
                    s.item()) + " Average time: " + str(avg_time_per_it_std))
                index_store = index_store + 1

        ##########################################
        #          Store final results           #
        ##########################################
        x, d, a, r, c, s = alg_tich.get_state()
        x_store_tich[test, :, :] = x.flatten(1)
        dual_store_tich[test, :,:] = d.flatten(1)
        aux_store_tich[test, :, :] = a.flatten(1)
        local_cost_store_tich[test, :, :] = c.flatten(1)
        # HSDM
        x, d, a, r, c, s = alg_hsdm.get_state()
        x_store_hsdm[test, :, :] = x.flatten(1)
        dual_store_hsdm[test, :, :] = d.flatten(1)
        aux_store_hsdm[test, :, :] = a.flatten(1)
        local_cost_store_hsdm[test, :, :] = c.flatten(1)
        # FBF
        x, d, a, r, c = alg_std.get_state()
        x_store_std[test, :, :] = x.flatten(1)
        dual_store_std[test, :, :] = d.flatten(1)
        aux_store_std[test, :, :] = a.flatten(1)
        local_cost_store_std[test, :, :] = c.flatten(1)

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result.pkl', 'wb')
    pickle.dump([ x_store_tich, x_store_hsdm, x_store_std,
                 dual_store_tich, dual_store_hsdm, dual_store_std,
                 aux_store_tich, aux_store_hsdm, aux_store_std,
                 residual_store_tich, residual_store_hsdm, residual_store_std,
                 sel_func_store_tich, sel_func_store_hsdm, sel_func_store_std ], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")


