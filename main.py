import numpy as np
import networkx as nx
import torch
import pickle
from GNE_distrib_selection import pFB_tich_prox_distr_algorithm, FBF_HSDM_distr_algorithm
from GNE_distrib import FBF_distr_algorithm
from RadioCommSetup import RadioCommSetup
from RandomLinearSetup import RandomLinearSetup

from GameDefinition import Game
import time
import logging
import sys

if __name__ == '__main__':
    logging.basicConfig(filename='log.txt', filemode='w',level=logging.DEBUG)
    use_test_game = False  # trigger 2-players sample zero-sum monotone game
    if use_test_game:
        print("WARNING: test game will be used.")
        logging.info("WARNING: test game will be used.")
    if len(sys.argv) < 2:
        seed = 1
        job_id=0
    else:
        seed=int(sys.argv[1])
        job_id = int(sys.argv[2])
    print("Random seed set to  " + str(seed))
    logging.info("Random seed set to  " + str(seed))
    np.random.seed(seed)
    N_iter=1000000
    N_it_per_residual_computation = 10
    N_agents = 10
    n_channels = 5
    n_neighbors = 4 # for simplicity, each agent has the same number of neighbours. This is only used to create the communication graph (but i's not needed otherwise)
    P_max_local = torch.ones(N_agents,1) # Max power that each agent can output (cumulative on all channels).
                                         # By the reformulation of Scutari, this is used in en EQUALITY constraint
                                         # (so the power output is = P_max)
    P_max_shared = 1*torch.sum(P_max_local).item()/(n_channels) # Max power that can go on a channel
    if P_max_shared < torch.sum(P_max_local).item()/n_channels:
        raise ValueError("The shared constraint is infeasible: increase power allowed on each channel")
    N_random_tests = 1
    n_taps = 10 # order of the filters
    is_connected = False
    comm_graph = nx.random_regular_graph(n_neighbors, N_agents)

    ##########################################
    #  Define alg. parameters to test        #
    ##########################################

    exponent_sel_fun_to_test_tich = [0.3, .6, 1.] # The selection function weight decades as 1/k^exp
    exponent_sel_fun_to_test_hsdm = [0.7]  # The selection function weight decades as 1/k^exp
    tichonov_inertia_to_test = [.1, 1.] # weight that multiplies \|x-x_k\| in the tich. reg. problem
    tichonov_epsilon_wrt_sel_fun_to_test = [2., 4., 6.] # The approx. error in the tich.reg. problem evolves as 1/k^(p*exp), where p is this parameter and exp is the first parameter
    parameters_to_test_hsdm=[]
    parameters_to_test_tich=[]
    for par in exponent_sel_fun_to_test_hsdm:
        parameters_to_test_hsdm.append((par,))
    for par_1 in exponent_sel_fun_to_test_tich:
        for par_2 in tichonov_inertia_to_test:
            for par_3 in tichonov_epsilon_wrt_sel_fun_to_test:
                parameters_to_test_tich.append((par_1,par_2,par_3))
    N_parameters_to_test_tich = len(parameters_to_test_tich)
    N_parameters_to_test_hsdm = 1 # len(parameters_to_test_hsdm)

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
        # game_params = RadioCommSetup(N_agents, n_channels, P_max_shared, P_max_local, taps_filter, comm_graph)
        game_params = RandomLinearSetup(N_agents, n_channels,  P_max_shared, P_max_local, comm_graph)
        print("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        logging.info("Initializing game for test " + str(test) + " out of " +str(N_random_tests))
        ##########################################
        #             Game inizialization        #
        ##########################################
        game = Game(N_agents, n_channels, comm_graph, game_params.Q , game_params.c,
                    game_params.A_eq_loc_const, game_params.A_ineq_loc_const, game_params.A_ineq_shared,
                    game_params.b_eq_loc_const, game_params.b_ineq_loc_const, game_params.b_ineq_shared,
                    Q_sel=game_params.Q_sel_fun, c_sel=game_params.c_sel_fun, test=use_test_game)
        x_0 = torch.from_numpy(np.random.rand(game.N_agents, game.n_opt_variables,
                                              1))  # torch.zeros(game.N_agents, game.n_opt_variables, 1)
        if test == 0:
            print("The game has " + str(game.N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_ineq_constr) + " shared ineq. constraints" )
            logging.info("The game has " + str(game.N_agents) + " agents; " + str(game.n_opt_variables) + " opt. variables per agent; " \
                  + str(game.A_ineq_loc.size()[1]) + " Local ineq. constraints; " + str(game.A_eq_loc.size()[1]) + " local eq. constraints; " + str(game.n_shared_ineq_constr) + " shared ineq. constraints" )
            ##########################################
            #   Variables storage inizialization     #
            ##########################################
            # pFB-Tichonov
            x_store_tich = torch.zeros(N_random_tests, N_parameters_to_test_tich, game.N_agents, game.n_opt_variables)
            dual_store_tich = torch.zeros(N_random_tests, N_parameters_to_test_tich, game.N_agents, game.n_shared_ineq_constr)
            aux_store_tich = torch.zeros(N_random_tests, N_parameters_to_test_tich, game.N_agents, game.n_shared_ineq_constr)
            residual_store_tich = torch.zeros(N_random_tests, N_parameters_to_test_tich, (N_iter // N_it_per_residual_computation))
            local_cost_store_tich = torch.zeros(N_random_tests, N_parameters_to_test_tich, game.N_agents, 1 + (N_iter // N_it_per_residual_computation))
            sel_func_store_tich = torch.zeros(N_random_tests, N_parameters_to_test_tich, (N_iter // N_it_per_residual_computation))
            # FBF-HSDM
            x_store_hsdm = torch.zeros(N_random_tests, N_parameters_to_test_hsdm, game.N_agents, game.n_opt_variables)
            dual_store_hsdm = torch.zeros(N_random_tests, N_parameters_to_test_hsdm, game.N_agents, game.n_shared_ineq_constr)
            aux_store_hsdm = torch.zeros(N_random_tests, N_parameters_to_test_hsdm, game.N_agents, game.n_shared_ineq_constr)
            residual_store_hsdm = torch.zeros(N_random_tests, N_parameters_to_test_hsdm, (N_iter // N_it_per_residual_computation))
            local_cost_store_hsdm = torch.zeros(N_random_tests, N_parameters_to_test_hsdm, game.N_agents, (N_iter // N_it_per_residual_computation))
            sel_func_store_hsdm = torch.zeros(N_random_tests, N_parameters_to_test_hsdm, (N_iter // N_it_per_residual_computation))
            # FBF
            x_store_std = torch.zeros(N_random_tests, 1, game.N_agents, game.n_opt_variables)
            dual_store_std = torch.zeros(N_random_tests, 1, game.N_agents, game.n_shared_ineq_constr)
            aux_store_std = torch.zeros(N_random_tests, 1, game.N_agents, game.n_shared_ineq_constr)
            residual_store_std = torch.zeros(N_random_tests, 1, (N_iter // N_it_per_residual_computation))
            local_cost_store_std = torch.zeros(N_random_tests, 1, game.N_agents, (N_iter // N_it_per_residual_computation))
            sel_func_store_std = torch.zeros(N_random_tests, 1, (N_iter // N_it_per_residual_computation))
            # Ideal values
            x_store_opt = torch.zeros(N_random_tests, 1, game.N_agents, game.n_opt_variables)
            phi_store_opt = torch.zeros(N_random_tests, 1)

        #######################################
        #          Run Tichonov               #
        #######################################
        # alg. initialization
        for index_parameter_set in range(N_parameters_to_test_tich):
            parameter_set_tich = parameters_to_test_tich[index_parameter_set]
            alg_tich = pFB_tich_prox_distr_algorithm(game, x_0=x_0,
                                                     exponent_vanishing_precision=parameter_set_tich[2]*parameter_set_tich[0],
                                                     exponent_vanishing_selection=parameter_set_tich[0],
                                                     alpha_tich_regul=parameter_set_tich[1])
            alg_tich.set_stepsize_using_Lip_const(safety_margin=.5)
            index_storage = 0
            avg_time_per_it_tich = 0
            for k in range(N_iter):
                if k % N_it_per_residual_computation == 0:
                    # Save performance metrics
                    x, d, a, r, c, s  = alg_tich.get_state()
                    residual_store_tich[test, index_parameter_set, index_storage] = r
                    sel_func_store_tich[test, index_parameter_set, index_storage] = s
                    print("Tichonov: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " +  str(s.item()) +  " Average time: " + str(avg_time_per_it_tich))
                    logging.info("Tichonov: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " +  str(s.item()) +" Average time: " + str(avg_time_per_it_tich))
                    index_storage = index_storage + 1
                #  Algorithm run
                start_time = time.time()
                alg_tich.run_once()
                end_time = time.time()
                avg_time_per_it_tich = (avg_time_per_it_tich * k + (end_time - start_time)) / (k + 1)

            # Store final variables
            x, d, a, r, c, s = alg_tich.get_state()
            x_store_tich[test, index_parameter_set, :, :] = x.flatten(1)
            dual_store_tich[test, index_parameter_set, :, :] = d.flatten(1)
            aux_store_tich[test, index_parameter_set, :, :] = a.flatten(1)
            local_cost_store_tich[test, index_parameter_set, :, :] = c.flatten(1)

        ######################################
        #          Run HSDM                  #
        ######################################
        # alg. initialization
        for index_parameter_set in range(N_parameters_to_test_hsdm):
            parameter_set_hsdm = parameters_to_test_hsdm[index_parameter_set]
            alg_hsdm = FBF_HSDM_distr_algorithm(game, x_0=x_0, exponent_vanishing_selection=parameter_set_hsdm[0])
            alg_hsdm.set_stepsize_using_Lip_const(safety_margin=.5)
            index_storage = 0
            avg_time_per_it_hsdm = 0
            for k in range(N_iter):
                if k % N_it_per_residual_computation == 0:
                    #          Save performance metrics      #
                    x, d, a, r, c, s = alg_hsdm.get_state()
                    residual_store_hsdm[test, index_parameter_set, index_storage] = r
                    sel_func_store_hsdm[test, index_parameter_set, index_storage] = s
                    print("HSDM: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " + str(
                        s.item()) + " Average time: " + str(avg_time_per_it_hsdm))
                    logging.info("HSDM: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " + str(
                        s.item()) + " Average time: " + str(avg_time_per_it_hsdm))
                    index_storage = index_storage + 1
                # Algorithm run
                start_time = time.time()
                alg_hsdm.run_once()
                end_time = time.time()
                avg_time_per_it_hsdm = (avg_time_per_it_hsdm * k + (end_time - start_time)) / (k + 1)

            # Store final variables
            x, d, a, r, c, s = alg_hsdm.get_state()
            x_store_hsdm[test, index_parameter_set, :, :] = x.flatten(1)
            dual_store_hsdm[test, index_parameter_set, :, :] = d.flatten(1)
            aux_store_hsdm[test, index_parameter_set, :, :] = a.flatten(1)
            local_cost_store_hsdm[test, index_parameter_set, :, :] = c.flatten(1)

        #####################################
        #          Run FBF                  #
        #####################################
        # alg. initialization
        alg_std = FBF_distr_algorithm(game, x_0=x_0)
        alg_std.set_stepsize_using_Lip_const(safety_margin=.5)
        index_storage = 0
        avg_time_per_it_tich = 0
        avg_time_per_it_hsdm = 0
        avg_time_per_it_std = 0
        for k in range(N_iter):
            if k % N_it_per_residual_computation == 0:
                # Save performance metrics
                # FBF
                x, d, a, r, c = alg_std.get_state()
                s = game.phi(x, d, a)
                residual_store_std[test, 0, index_storage] = r
                sel_func_store_std[test, 0, index_storage] = s
                print("FBF: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " + str(
                    s.item()) + " Average time: " + str(avg_time_per_it_std))
                logging.info("FBF: Iteration " + str(k) + " Residual: " + str(r.item()) + " Sel function: " + str(
                    s.item()) + " Average time: " + str(avg_time_per_it_std))
                index_storage = index_storage + 1
            # Algorithm run
            start_time = time.time()
            alg_std.run_once()
            end_time = time.time()
            avg_time_per_it_std = (avg_time_per_it_std * k + (end_time - start_time)) / (k + 1)
        # Store final variables
        x, d, a, r, c = alg_std.get_state()
        x_store_std[test, 0, :, :] = x.flatten(1)
        dual_store_std[test, 0, :, :] = d.flatten(1)
        aux_store_std[test, 0, :, :] = a.flatten(1)
        local_cost_store_std[test, 0, :, :] = c.flatten(1)
        ### Compute ideal solution -- WARNING: only useful for strictly feasible solution
        # x_opt, phi_opt = game.computeOptimalSelection()
        # x_store_opt[test,0,:,:] = x_opt.flatten(1)
        # phi_store_opt[test,:] = phi_opt

    print("Saving results...")
    logging.info("Saving results...")
    f = open('saved_test_result_'+ str(job_id) + ".pkl", 'wb')
    pickle.dump([ x_store_tich, x_store_hsdm, x_store_std,
                 dual_store_tich, dual_store_hsdm, dual_store_std,
                 aux_store_tich, aux_store_hsdm, aux_store_std,
                 residual_store_tich, residual_store_hsdm, residual_store_std,
                 sel_func_store_tich, sel_func_store_hsdm, sel_func_store_std,
                 parameters_to_test_tich, parameters_to_test_hsdm], f)
    f.close()
    print("Saved")
    logging.info("Saved, job done")


