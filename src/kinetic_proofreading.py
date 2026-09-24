import numpy as np
from numpy import random
import matplotlib.pylab as plt
import time
import logging
from datetime import datetime
import seaborn as sns

# import csv  # if csv is used in this program
np.seterr(divide='ignore')

''' choices of parameters '''
N_k = 2  # number of kinetic proofreading steps
# e_a = 1.5*N_k  # epsilon_a, activation energy
delta_e = 2*N_k  # energy decreases by delta_epsilon after binding
j_b = 0.55  # coupling strengh between b states, it's 2.00723 in Onsager‘s sol. to 2d Ising model
c = 0.32  # concentration of self
e_b_s = -1  # epsilon_b, binding energy of self
e_b_a = -6  # epsilon_b, binding energy of agonist
denk = delta_e/N_k  # for faster calculation
four_J_b = 4 * j_b  # for faster calculation

''' save log file '''
current_time = datetime.now().strftime("%Y%m%d")
logfile_name = f"date_{current_time}_kp_case_jb_{j_b}_c_{c}_simulation.log"
logging.basicConfig(filename = logfile_name, level = logging.INFO, format = '%(asctime)s - %(message)s')

''' system size and number of steps '''
L = 5  # length of the receptor lattice
Lsq = L**2  # total number of receptors
L_s = 21  # length of the whole lattice system
n_l = int(L_s*L_s*c)  # number of total ligands
left_r = int((L_s-L)/2)  # leftmost receptor: 8
right_r = int(left_r + L)  # (rightmost + 1) receptor: 13
middle_site = int(L_s//2)  # middle site of agonist: 10
rec = [8, 9, 10, 11, 12]  # possible positions of receptors
par = [k for k in range(21)]  # possible positions of particles
dt = 2.5*(10**(-3))  # iteration time interval
tsteps = 9*(10**5)  # thermalization steps
ssteps = 9*(10**5)  # number of MC steps when only self are present
asteps = 6*(10**4)
n_runs = 100  # number of trials
s = 3000  # store time evolution process every s data points
sa = 3000  # after fsteps, store the prob of activation every sa data points
thermal_steps = 10*s  # store time evolution of thermalization every 10*s data points


''' initial positions of all floating ligands '''


def init_float(n_ligands):
    floating = []
    for i in range(n_ligands):
        x_l = random.choice(par)
        y_l = random.choice(par)
        floating.append([x_l, y_l])
    return floating


''' rate constants '''
p_move = 0.25  # prob of ligand move
r_on = 1  # binding rate, indep. of a or b
# unbinding rates of self without coupling
r_off_zero_self = r_on * np.exp(e_b_s)  # when a = 0
r_off_one_self = r_on * np.exp(e_b_s - delta_e)  # when a = 1
# r_a_one < r_off((N_k-1)/N_k)
r_off_compare = r_on * np.exp(e_b_s - (N_k-1)*denk)
beta_one = 0.85
r_a_one = beta_one * r_off_compare
r_d_zero = 0.4


''' Get the rates '''
ron_dt = r_on * dt
roffzero_selfdt = r_off_zero_self * dt
roffone_selfdt = r_off_one_self * dt
roff_comparedt = r_off_compare * dt
raonedt = r_a_one * dt
rdzerodt = r_d_zero * dt


''' 
Monte Carlo simulation: 
To avoid a cannot be updated when it's non-integer,
a = 0 represents a = 0
a = 1 represents a = 1/N_k
a = 2 represents a = 2/N_k
.
.
.
a = N_k represents a = 1
'''


def MC_step(config_a, config_b, F):

    # update positions of floating ligands
    for each_l in range(len(F)):
        Rs = random.rand()
        if 0 < Rs <= 0.0625:
            F[each_l][0] = (F[each_l][0] - 1) % L_s
        elif 0.0625 < Rs <= 0.125:
            F[each_l][0] = (F[each_l][0] + 1) % L_s
        elif 0.125 < Rs <= 0.1875:
            F[each_l][1] = (F[each_l][1] - 1) % L_s
        elif 0.1875 < Rs <= 0.25:
            F[each_l][1] = (F[each_l][1] + 1) % L_s
        else:
            pass

    # update state of all the receptors
    for i in range(L):  # L = 5: length of receptor
        for j in range(L):
            # loop over i & j so use x & y here to randomly choose a site (x,y) in the receptor lattice
            x = random.choice(rec)
            y = random.choice(rec)
            r = random.rand()  # choose a random number in [0,1)

            # What to do when reach free receptor
            if config_b[x, y] == 0:
                # calculate the number of float ligands
                n_of_xy = F.count([x, y])
                if n_of_xy == 0:  # no free ligand above this site
                    if config_a[x, y] == 0:
                        pass
                    else:
                        current_a = config_a[x, y]
                        if r < rdzerodt:
                            config_a[x, y] = int(current_a - 1)
                else:  # n_of_xy >= 1, at least one free ligand above this site
                    ron_n_dt = ron_dt * n_of_xy
                    if config_a[x, y] == 0:
                        if r < ron_n_dt:
                            config_b[x, y] = 1
                            F.remove([x, y])
                    else:
                        current_a = config_a[x, y]
                        if r > (ron_n_dt + rdzerodt):
                            pass
                        elif r < ron_n_dt:
                            config_b[x, y] = 1
                            F.remove([x, y])
                        else:
                            config_a[x, y] = int(current_a - 1)

            # What to do when reach bound receptor
            else:
                nbs = config_b[x - 1, y]**2 + config_b[x + 1, y]**2 + \
                    config_b[x, y - 1]**2 + config_b[x, y + 1]**2 - 2
                neighbor_effect = np.exp(- four_J_b * nbs)

                if config_a[x, y] == 0:
                    # unbinding rate with coupling when a = 0
                    roffzero_sdt = roffzero_selfdt * neighbor_effect
                    if r > (roffzero_sdt + raonedt):
                        pass
                    elif r < roffzero_sdt:
                        config_b[x, y] = 0
                        F.append([x, y])
                    else:
                        config_a[x, y] = 1
                elif config_a[x, y] == N_k:
                    # unbinding rate with coupling when a = N_k (a = 1)
                    roffone_sdt = roffone_selfdt * neighbor_effect
                    if r < roffone_sdt:
                        config_b[x, y] = 0
                        F.append([x, y])
                else:
                    current_a = config_a[x, y]
                    # unbinding rate with coupling when a = current_a (a = current_a/N_k)
                    roff_sdt = ron_dt * \
                        np.exp(e_b_s - current_a * denk) * neighbor_effect
                    if r > (roff_sdt + raonedt):
                        pass
                    elif r < roff_sdt:
                        config_b[x, y] = 0
                        F.append([x, y])
                    else:
                        config_a[x, y] = int(current_a + 1)

    return config_a, config_b


''' statistics of receptors in each state '''


def dist(a, b):
    prob_dist = np.zeros((N_k+1, 2))
    for i in range(left_r, right_r):
        for j in range(left_r, right_r):
            aa = int(a[i][j])
            bb = int(b[i][j])
            prob_dist[aa][bb] += 1
    prob_dist = prob_dist/Lsq
    return prob_dist


''' monitor the states of all the receptors '''


def rec_stat(a, b):
    final_a = np.array([a[i][left_r:right_r] for i in range(left_r, right_r)])
    final_b = np.array([b[i][left_r:right_r] for i in range(left_r, right_r)])
    stat = [[[0, 0] for j in range(L)] for i in range(L)]
    for i in range(L):
        for j in range(L):
            aa = final_a[i][j]
            bb = final_b[i][j]
            stat[i][j][0] = round(aa/N_k, 3)
            stat[i][j][1] = bb
    return stat


''' plot the binding state of the receptors '''


def plot_b_lattice(b_grid, Step, Run):
    plt.figure(figsize=(4, 4))
    sns.heatmap(b_grid, cmap='gray_r', cbar=False, square=True,
                linewidths=1, linecolor='gray',
                xticklabels=False, yticklabels=False, vmin=0, vmax=1)
    plt.title(f"Run: {Run} | Step: {Step}")
    plt.tight_layout()
    plt.savefig(f"Run_{Run}_Step_{Step}.png")
    plt.close()


#################################################################################################
''' main program '''
#################################################################################################
# store time evolution process of a and b every s data points
monitor_A, monitor_B = np.zeros(
    (n_runs, int(ssteps/s))), np.zeros((n_runs, int(ssteps/s)))
samp_acti = np.zeros((n_runs, int(asteps/sa)))

start = time.time()  # calculate execution time

''' 1. thermalization and evolution of self '''

for each_run in range(n_runs):
    
    logging.info(f"thermalisation: jb = {j_b}, c = {c}, run = {each_run}")

    # set the initial conditions of new trial
    config_a = np.zeros((L_s, L_s))
    config_b = np.zeros((L_s, L_s))

    # set the initial positions of all floating ligands in new trial
    floating_ligands = init_float(n_l)

    # changing variables are reset in new trial
    monitor_stats = np.zeros((N_k+1, 2))
    mean_stats = np.zeros((N_k+1, 2))

    # thermalization
    step, m = 0, 0
    while step < tsteps:
        for each_s in range(s):
            MC_step(config_a, config_b, floating_ligands)
        monitor_stats = monitor_stats + dist(config_a, config_b)
        m += 1
        if step % (thermal_steps) == 0:
            logging.info(f"thermalisation step number = {step}")
            logging.info(f"prob of each state (simulation) = {monitor_stats/m}")
        step += s

    # changing variables are reset in new trial
    monitor_stats = np.zeros((N_k+1, 2))
    mean_stats = np.zeros((N_k+1, 2))

    # self form clusters
    step, m = 0, 0
    while step < ssteps:
        for each_s in range(s):
            MC_step(config_a, config_b, floating_ligands)
        monitor_A[each_run, m] = np.sum(config_a == N_k)
        monitor_B[each_run, m] = np.sum(config_b == 1)
        monitor_stats = monitor_stats + dist(config_a, config_b)
        m += 1
        if step % (thermal_steps) == 0:
            logging.info(f"self only step number = {step}")
            logging.info(f"prob of each state (simulation) = {monitor_stats/m}")
        step += s

    final_state = rec_stat(config_a, config_b)
    logging.info("self only: final states of receptors = ")
    for i in range(L):
        logging.info(f"{final_state[i]}")
    
    if each_run < 3:
        # save the final state of only the receptors
        rece_config_a = np.array([config_a[i][left_r:right_r]
                                 for i in range(left_r, right_r)])
        rece_config_b = np.array([config_b[i][left_r:right_r]
                                 for i in range(left_r, right_r)])
        np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_receptor_config_a_run_' +
                   str(each_run) + '.csv', rece_config_a, delimiter=",", fmt='%.1f')
        np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_receptor_config_b_run_' +
                   str(each_run) + '.csv', rece_config_b, delimiter=",", fmt='%.1f')
    
        # store the final state of the whole system
        whole_config_a, whole_config_b = config_a, config_b
        np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_whole_config_a_run_' +
                   str(each_run) + '.csv', whole_config_a, delimiter=",", fmt='%.1f')
        np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_whole_config_b_run_' +
                   str(each_run) + '.csv', whole_config_b, delimiter=",", fmt='%.3f')
    
        # save the final positions of the remaining float ligands
        np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_run_' + str(each_run) +
                   '_final_positions_of_remaining_floating_ligands.csv', floating_ligands, delimiter=",", fmt='%.1f')

    # after ssteps, sampling the prob of fully activation every sa data points
    monitor_stats = np.zeros((N_k+1, 2))
    step, ma = 0, 0
    while step < asteps:
        for each_sa in range(sa):
            MC_step(config_a, config_b, floating_ligands)
        step += sa
        ma += 1
        new_stat = dist(config_a, config_b)
        logging.info(f"thermalisation: time average step number = {ssteps + step}")
        logging.info(f"prob of each state (sampling) = {new_stat}")
        # p(a = N_k, b = 0) + p(a = N_k, b = 1)
        samp_acti[each_run, (ma-1)] = np.sum(new_stat[N_k])

end = time.time()
logging.info(f"jb = {j_b}, c = {c}, only self run time: {end - start}")

# average sampling data and save in a file (processed data)
aver_prob = np.array([np.mean(samp_acti[each_run])
                     for each_run in range(n_runs)])
all_aver_prob = aver_prob
np.savetxt('jb_' + str(j_b) + '_c_' + str(c) +
           '_only_self_ensemble_average_prob.csv', all_aver_prob, delimiter=",", fmt='%.8f')

# store time evolution of each trial
evolution_a, evolution_b = monitor_A, monitor_B
np.savetxt('jb_' + str(j_b) + '_c_' + str(c) +
           '_only_self_evolution_activity.csv', evolution_a, delimiter=",", fmt='%.1f')
np.savetxt('jb_' + str(j_b) + '_c_' + str(c) +
           '_only_self_evolution_bindings.csv', evolution_b, delimiter=",", fmt='%.1f')


''' record simulation time by saving another log file '''
current_time = datetime.now().strftime("%Y%m%d")
logfile_name = f"date_{current_time}_kp_case_jb_{j_b}_c_{c}_simul_hours.log"
logging.basicConfig(filename = logfile_name, level = logging.INFO, format = '%(asctime)s - %(message)s')
tot_hr = round((end-start)/3600, 1)
logging.info(f"jb = {j_b}, c = {c}, only self ")
logging.info(f"run time: {tot_hr}")
