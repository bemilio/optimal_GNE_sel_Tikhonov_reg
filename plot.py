import matplotlib as mpl
import seaborn as sns
import pandas as pd
mpl.interactive(True)
import matplotlib.pyplot as plt

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "STIXGeneral",
    "font.serif": ["Computer Modern Roman"],
})
import networkx as nx
import numpy as np
import pickle
import torch
import os

TICH = 0
HSDM = 1
FBF=2

method_name= { TICH: "Tich", HSDM:"HSDM", FBF:"FBF"}
load_files_from_current_dir = False
create_new_dataframe_file = True
if load_files_from_current_dir:
    directory = "."
else:
    directory = r"/Users/ebenenati/surfdrive/TUDelft/Simulations/Cross_interference/Tichonov_vs_HSDm/24_02_23"
if not os.path.exists(directory + "/Figures"):
    os.makedirs(directory + r"/Figures")

if create_new_dataframe_file or not os.path.exists('saved_dataframe.pkl'):
    if  create_new_dataframe_file:
        print("DataFrame file will be recreated as requested. This will take a while.")
    else:
        print("File containing the DataFrame not found, it will be created. This will take a while.")
    #### Toggle between loading saved file in this directory or all files in a specific directory
    if load_files_from_current_dir:
        f = open('saved_test_result.pkl', 'rb')
        x_store_tich, x_store_hsdm, x_store_std, \
            dual_store_tich, dual_store_hsdm, dual_store_std,  \
            aux_store_tich, aux_store_hsdm, aux_store_std, \
            residual_store_tich, residual_store_hsdm, residual_store_std, \
            sel_func_store_tich, sel_func_store_hsdm, sel_func_store_std, \
            parameters_tested= pickle.load(f)
        f.close()
    else:
        #########
        # Load all files in a directory and stack them in single tensors
        #########
        N_files = 0
        for filename in os.listdir(directory):
            if filename.find('.pkl')>=0:
                N_files=N_files+1 #count all files

        dummy = False
        for filename in os.listdir(directory):
            if filename.find('.pkl')>=0:
                f=open(directory+"/"+filename, 'rb')
                x_store_tich_single_file, x_store_hsdm_single_file, x_store_std_single_file, \
                    dual_store_tich_single_file, dual_store_hsdm_single_file, dual_store_std_single_file, \
                    aux_store_tich_single_file, aux_store_hsdm_single_file, aux_store_std_single_file, \
                    residual_store_tich_single_file, residual_store_hsdm_single_file, residual_store_std_single_file, \
                    sel_func_store_tich_single_file, sel_func_store_hsdm_single_file, sel_func_store_std_single_file, \
                    parameters_tested = pickle.load(f)
                if not dummy:
                    x_store_tich= x_store_tich_single_file
                    x_store_hsdm = x_store_hsdm_single_file
                    x_store_std = x_store_std_single_file
                    residual_store_tich = residual_store_tich_single_file
                    residual_store_hsdm= residual_store_hsdm_single_file
                    residual_store_std = residual_store_std_single_file
                    sel_func_store_tich = sel_func_store_tich_single_file
                    sel_func_store_hsdm = sel_func_store_hsdm_single_file
                    sel_func_store_std = sel_func_store_std_single_file
                else:
                    x_store_tich = torch.cat((x_store_tich, x_store_tich_single_file),dim=0)
                    x_store_hsdm = torch.cat((x_store_hsdm, x_store_hsdm_single_file),dim=0)
                    x_store_std = torch.cat((x_store_std, x_store_std_single_file),dim=0)
                    residual_store_tich = torch.cat((residual_store_tich, residual_store_tich_single_file),dim=0)
                    residual_store_hsdm = torch.cat((residual_store_hsdm, residual_store_hsdm_single_file),dim=0)
                    residual_store_std = torch.cat((residual_store_std, residual_store_std_single_file),dim=0)
                    sel_func_store_tich = torch.cat((sel_func_store_tich, sel_func_store_tich_single_file),dim=0)
                    sel_func_store_hsdm = torch.cat((sel_func_store_hsdm, sel_func_store_hsdm_single_file),dim=0)
                    sel_func_store_std = torch.cat((sel_func_store_std, sel_func_store_std_single_file),dim=0)
                dummy = True
    print("Files loaded...")
    N_tests = x_store_tich.size(0)
    N_tested_sets_of_params = x_store_tich.size(1)
    N_parameters = 3 # parameters_to_test is a list of tuples, each tuple contains N_parameters
    N_agents = x_store_tich.size(2)
    N_opt_var = x_store_tich.size(3)
    Steps_between_iterations = 10
    N_iterations = residual_store_tich.size(2)
    N_methods = 3 # tichonov, hsdm, standard


    fig, ax = plt.subplots(2, 1, figsize=(4 * 1, 3.1 * 1), layout='constrained', sharex='col')
    x = range(1,Steps_between_iterations*N_iterations, Steps_between_iterations)
    ax[0].plot(x,torch.mean(residual_store_std[:,4,:], dim=0), color='k', label="FBF")
    ax[0].fill_between(x,torch.min(residual_store_std[:,4,:], dim=0)[0].numpy(), \
                     y2=torch.max(residual_store_std[:,4,:], dim=0)[0].numpy(), alpha=0.2, color='k')
    ax[0].plot(x, torch.mean(residual_store_hsdm[:,4,:], dim=0), color='g', label="HSDM")
    ax[0].fill_between(x, torch.min(residual_store_hsdm[:,4,:], dim=0)[0].numpy(),\
                     y2=torch.max(residual_store_hsdm[:,4,:], dim=0)[0].numpy(), alpha=0.2, color='g')
    ax[0].plot(x, torch.mean(residual_store_tich[:,4,:], dim=0), color='m', label="Tichonov")
    ax[0].fill_between(x, torch.min(residual_store_tich[:,4,:], dim=0)[0].numpy(), \
                     y2=torch.max(residual_store_tich[:,4,:], dim=0)[0].numpy(), alpha=0.2, color='m')
    ax[0].grid(True)
    ax[0].set_yscale('log')
    ax[0].set_xscale('log')
    ax[0].set_ylabel("Residual", fontsize=9)
    ax[0].set_xlim(1, N_iterations * 10)
    plt.legend()

    ax[1].plot(x, torch.mean(sel_func_store_std[:,4,:], dim=0), color='k', label="FBF")
    ax[1].fill_between(x, torch.min(sel_func_store_std[:,4,:], dim=0)[0].numpy(), \
                     y2=torch.max(sel_func_store_std[:,4,:], dim=0)[0].numpy(), alpha=0.2, color='k')
    relative_advantage_hsdm = sel_func_store_hsdm[:,4,:] - sel_func_store_std[:,4,:]
    # ax[1].plot(x, torch.mean(relative_advantage_hsdm, dim=0) , color='g', label="HSDM")
    # ax[1].fill_between(x, torch.min(relative_advantage_hsdm, dim=0)[0].numpy(),\
    #                  y2=torch.max(relative_advantage_hsdm, dim=0)[0].numpy(), alpha=0.2, color='g')
    ax[1].plot(x, torch.mean(sel_func_store_hsdm[:,4,:], dim=0), color='g', label="HSDM")
    ax[1].fill_between(x, torch.min(sel_func_store_hsdm[:,4,:], dim=0)[0].numpy(), \
                     y2=torch.max(sel_func_store_hsdm[:,4,:], dim=0)[0].numpy(), alpha=0.2, color='g')
    # relative_advantage_tich = torch.div(sel_func_store_tich[:,4,:] - sel_func_store_std[:,4,:], torch.abs(sel_func_store_std[:,4,:]) )
    relative_advantage_tich = sel_func_store_tich[:,4,:] - sel_func_store_std[:,4,:]
    # ax[1].plot(x, torch.mean(relative_advantage_tich, dim=0), color='m', label="Tichonov")
    # ax[1].fill_between(x, torch.min(relative_advantage_tich, dim=0)[0].numpy(), \
    #                  y2=torch.max(relative_advantage_tich, dim=0)[0].numpy(), alpha=0.2, color='m')
    ax[1].plot(x, torch.mean(sel_func_store_tich[:,4,:], dim=0), color='m', label="Tichonov")
    ax[1].fill_between(x, torch.min(sel_func_store_tich[:,4,:], dim=0)[0].numpy(), \
                     y2=torch.max(sel_func_store_tich[:,4,:], dim=0)[0].numpy(), alpha=0.2, color='m')
    ax[1].grid(True)
    ax[1].set_xscale('log')
    ax[1].set_ylabel("Sel. Fun.", fontsize=9)
    ax[1].set_xlabel('Iteration', fontsize=9)
    ax[1].set_xlim(1, N_iterations * 10)
    plt.legend()
    plt.show(block=False)

    fig.savefig(directory + '/Figures/Method_comparison_HSDM_tich_FBF.png')
    fig.savefig(directory + '/Figures/Method_comparison_HSDM_tich_FBF.pdf')

    torch.Tensor.ndim = property(lambda self: len(self.shape))  # Necessary to use matplotlib with tensors

    parameters_set_to_plot_hsdm = []
    for i in range(len(parameters_tested)): # Maps the set of parameters used for tichonov to the set of parameters used for HSDM
        is_already_in_list = False
        for p_hsdm in parameters_set_to_plot_hsdm:
            if parameters_tested[i][0] == parameters_tested[p_hsdm][0]:
                is_already_in_list = True
        if not is_already_in_list:
            parameters_set_to_plot_hsdm.append(i)
    N_tested_sets_of_params_hsdm = len (parameters_set_to_plot_hsdm)

    N_datapoints = N_tests * (N_tested_sets_of_params + N_tested_sets_of_params_hsdm + 1)  * N_iterations
    list_parameter_set = np.zeros((N_datapoints,1))
    list_residual = np.zeros((N_datapoints,1))
    list_iteration = np.zeros((N_datapoints,1))
    list_method = np.zeros((N_datapoints,1))
    list_sel_fun = np.zeros((N_datapoints,1))

    index_datapoint=0
    for test in range(N_tests):
        for par_index in range(N_tested_sets_of_params):
            for iteration in range(N_iterations):
                # Tichonov
                list_residual[index_datapoint]=residual_store_tich[test,par_index,iteration].item()
                list_parameter_set[index_datapoint] = par_index
                list_iteration[index_datapoint] = iteration * 10 # residual is sampled every 10 iterations
                list_method[index_datapoint] = TICH
                list_sel_fun[index_datapoint] = sel_func_store_tich[test,par_index,iteration].item()
                index_datapoint = index_datapoint +1
                # HSDM
                if par_index in parameters_set_to_plot_hsdm:
                    list_residual[index_datapoint] = residual_store_hsdm[test, par_index, iteration]
                    list_parameter_set[index_datapoint] = par_index
                    list_iteration[index_datapoint] = iteration * 10 # residual is sampled every 10 iterations
                    list_method[index_datapoint] = HSDM
                    list_sel_fun[index_datapoint] = sel_func_store_hsdm[test, par_index, iteration].item()
                    index_datapoint = index_datapoint + 1
                # FBF
                if par_index == 0:
                    list_residual[index_datapoint] = residual_store_std[test, par_index, iteration]
                    list_parameter_set[index_datapoint] = 0
                    list_iteration[index_datapoint] = iteration * 10 # residual is sampled every 10 iterations
                    list_method[index_datapoint] = FBF
                    list_sel_fun[index_datapoint] = sel_func_store_std[test, par_index, iteration].item()
                    index_datapoint = index_datapoint +1
    df = pd.DataFrame({'Parameters set': list_parameter_set[:,0],\
                                    'Residual': list_residual[:,0], 'Iteration': list_iteration[:,0],'Method': list_method[:,0], 'Sel. Fun.': list_sel_fun[:,0]})
    # df['Parameters set'] = df['Parameters set'].map(parameters_labels)

    # Save dataframe
    f = open('saved_dataframe.pkl', 'wb')
    pickle.dump([df, parameters_tested, N_iterations, parameters_set_to_plot_hsdm], f)
    f.close()
    print("DataFrame file created.")


else:
    f = open('saved_dataframe.pkl', 'rb')
    df, parameters_tested, N_iterations, parameters_set_to_plot_hsdm = pickle.load(f)
    f.close()
##########
## PLOT ##
##########
# create dictionary that maps from parameter set to label
print("DataFrame acquired. Plotting...")
parameters_labels_tich = []
for i in range(len(parameters_tested)):
    parameters_labels_tich.append(r"$\gamma$ = " + str(parameters_tested[i][0]) +  # r"; $\alpha$ = " + str(parameters_tested[i][1]) + \
            r"; $\varepsilon$ = " + str(parameters_tested[i][2]))

parameters_labels_hsdm = []
for i in parameters_set_to_plot_hsdm:
    parameters_labels_hsdm.append(r"$\gamma$ = " + str(parameters_tested[i][0]))


fig, ax = plt.subplots(2,2, figsize=(4 * 2, 3.1 * 2), layout='constrained', sharex='col')
g = sns.lineplot(data=df.loc[ df['Method']==TICH ], drawstyle='steps-pre', errorbar=None, estimator='mean', x='Iteration', palette='bright',
             y='Residual', hue='Parameters set', linewidth = 2.0, ax=ax[0,0])
ax[0,0].grid(True)
ax[0,0].set_yscale('log')
ax[0,0].set_xscale('log')
ax[0,0].set_title("Tichonov")
ax[0,0].set_ylabel("Residual", fontsize = 9)
ax[0,0].set_xlabel('Iteration', fontsize = 9)
ax[0,0].set_xlim(1, N_iterations * 10 )
ax[0,0].set_ylim(ax[0,0].get_ylim()[0]/2, ax[0,0].get_ylim()[1]*2 )
g.legend(labels = parameters_labels_tich)

g= sns.lineplot(data=df.loc[ df['Method']==TICH ], drawstyle='steps-pre', errorbar=None, estimator='mean',  x='Iteration', palette='bright',
             y='Sel. Fun.', hue='Parameters set', linewidth = 2.0, ax=ax[1,0])
ax[1,0].grid(True)
ax[1,0].set_xscale('log')
# ax[1,0].set_yscale('log')
ax[1,0].set_ylabel("Sel. Fun.", fontsize = 9)
ax[1,0].set_xlabel('Iteration', fontsize = 9)
ax[1,0].set_xlim(1, N_iterations * 10 )
ax[1,0].set_ylim(ax[1,0].get_ylim()[0]/2, ax[1,0].get_ylim()[1]*2 )
g.legend(labels = parameters_labels_tich)

g = sns.lineplot(data=df.loc[ df['Method']==HSDM ], drawstyle='steps-pre', errorbar=None, estimator='mean', x='Iteration', palette='bright',
             y='Residual', hue='Parameters set', linewidth = 2.0, ax=ax[0,1])
ax[0,1].grid(True)
ax[0,1].set_xscale('log')
ax[0,1].set_yscale('log')
ax[0,1].set_title("HSDM")
ax[0,1].set_ylabel("Residual", fontsize = 9)
ax[0,1].set_xlabel('Iteration', fontsize = 9)
ax[0,1].set_ylim(ax[0,0].get_ylim())
g.legend(labels = parameters_labels_hsdm)

g = sns.lineplot(data=df.loc[ df['Method']==HSDM ], drawstyle='steps-pre', errorbar=None, estimator='mean', x='Iteration', palette='bright',
             y='Sel. Fun.', hue='Parameters set', linewidth = 2.0, ax=ax[1,1])
ax[1,1].grid(True)
ax[1,1].set_xscale('log')
# ax[1,1].set_yscale('log')
ax[1,1].set_ylabel("Sel. Fun.", fontsize = 9)
ax[1,1].set_xlabel('Iteration', fontsize = 9)
ax[1,1].set_ylim(ax[1,0].get_ylim())
g.legend(labels = parameters_labels_hsdm)

plt.show(block=False)

fig.savefig(directory + '/Figures/Parameters_comparison_tichonov_hsdm.png')
fig.savefig(directory + '/Figures/Parameters_comparison_tichonov_hsdm.pdf')


# Plot Tichonov with maximum exponent and minimum epsilon against HSDM with max. exponent and FBF
index_best_tich = 0
for i in range(len(parameters_tested)):
    if parameters_tested[i][0]>=parameters_tested[index_best_tich][0] and parameters_tested[i][2]<=parameters_tested[index_best_tich][2]:
        index_best_tich = i

index_best_hsdm = 0
for i in parameters_set_to_plot_hsdm:
    if parameters_tested[i][0]>=parameters_tested[index_best_hsdm][0]:
        index_best_hsdm = i

# fig, ax = plt.subplots(2,1, figsize=(4 * 2, 3.1 * 2), layout='constrained', sharex='col')
# g= sns.lineplot(data=pd.concat([df.loc[(df['Method']==TICH) & (df['Parameters set']==index_best_tich) ], \
#              df.loc[(df['Method']==HSDM) & (df['Parameters set']==index_best_hsdm)] , \
#              df.loc[(df['Method'] == FBF)] ]) , x='Iteration', palette='bright',
#              y='Residual', hue='Method', linewidth = 2.0, ax=ax[0], errorbar=("se", 2), estimator='mean', n_boot=0)
#
# ax[0].grid(True)
# ax[0].set_yscale('log')
# ax[0].set_xscale('log')
# ax[0].set_title("Tichonov")
# ax[0].set_ylabel("Residual", fontsize = 9)
# ax[0].set_xlabel('Iteration', fontsize = 9)
# ax[0].set_xlim(1, N_iterations * 10 )
# ax[0].set_ylim(10**(-3), 100)
# g.legend(['Tich', 'Conf. interval', 'HSDM', 'Conf. interval', 'FBF', 'Conf. interval'])
#
# g = plt.plot()
#
# g=sns.lineplot(data=pd.concat([df.loc[(df['Method']==TICH) & (df['Parameters set']==index_best_tich) ], \
#              df.loc[(df['Method']==HSDM) & (df['Parameters set']==index_best_hsdm)] , \
#              df.loc[(df['Method'] == FBF) & (df['Parameters set']==0)] ]) ,\
#              x='Iteration', palette='bright',
#              y='Sel. Fun.', hue='Method', linewidth = 2.0, ax=ax[1], errorbar=("se", 2), estimator='mean', n_boot=0)
#
# ax[1].grid(True)
# ax[1].set_yscale('log')
# ax[1].set_xscale('log')
# ax[1].set_title("Tichonov")
# ax[1].set_ylabel("Sel. Fun.", fontsize = 9)
# ax[1].set_xlabel('Iteration', fontsize = 9)
# ax[1].set_xlim(1, N_iterations * 10 )
# g.legend(['Tich', 'Conf. interval', 'HSDM', 'Conf. interval', 'FBF', 'Conf. interval'])
#
# plt.show(block=False)

# fig.savefig(directory + '/Figures/Method_comparison_HSDM_tich_FBF.png')
# fig.savefig(directory + '/Figures/Method_comparison_HSDM_tich_FBF.pdf')


print("Done")