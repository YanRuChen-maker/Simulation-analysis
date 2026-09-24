import numpy as np
from numpy import random
import matplotlib.pylab as plt
import time
# import csv  # if csv is used in this program
np.seterr(divide='ignore')  # ???
import logging
from datetime import datetime

date = '0425'  # date of running program

''' choices of parameters '''
N_k = 2  # number of kinetic proofreading steps
delta_e = 2*N_k  # energy decreases by delta_epsilon after binding
j_b = 0.55  # coupling strengh between b states
c = 0.32  # concentration of self
e_b_s =-1  # binding energy of self
e_b_a = -6  # binding energy of agonist
denk = delta_e/N_k  # for faster calculation
four_J_b = 4 * j_b  # for faster calculation

''' save log file '''
current_time = datetime.now().strftime("%Y%m%d")
logfile_name = f"date_{current_time}_rebind_case_jb_{j_b}_c_{c}_simulation.log"
logging.basicConfig(filename = logfile_name, level = logging.INFO, format = '%(asctime)s - %(message)s')

''' system size and number of steps '''
L = 5  # length of the receptor lattice
Lsq = L**2  # total number of receptors
L_s = 21  # length of the whole lattice system
n_l = int(L_s*L_s*c)  # number of total ligands
left_r = int((L_s-L)/2)  # leftmost receptor
right_r = int(left_r + L)  # (rightmost + 1) receptor
middle_site = int(L_s//2)  # middle site of agonist
rec = [8, 9, 10, 11, 12]  # possible positions of receptors
dt = 2.5*(10**(-3))  # iteration time interval
fsteps = 9*(10**5)  # number of MC steps
# after fsteps, number of MC steps for sampling the activation prob
asteps = 6*(10**4)
n_runs = 100  # number of trials
s = 3000  # save time evolution process every s data points
sa = 3000  # after fsteps, save the prob of activation every sa data points
thermal_steps = 10*s  # store time evolution of thermalization every some data points


''' rate constants '''
p_move = 0.25  # prob of ligand move
# rate constants
r_on = 1  # binding rate, indep. of a or b
# unbinding rates of self without coupling
r_off_zero_self = r_on * np.exp(e_b_s)  # when a = 0
r_off_one_self = r_on * np.exp(e_b_s - delta_e)  # when a = 1
r_off_compare = r_on * np.exp(e_b_s - (N_k-1)*denk)
# unbinding rates of agonist without coupling
r_off_zero_agonist = r_on * np.exp(e_b_a)  # when a = 0
r_off_one_agonist = r_on * np.exp(e_b_a - delta_e)  # when a = 1
beta_one = 0.85
r_a_one = beta_one * r_off_compare
r_d_zero = 0.4


''' address of reading files: a, b, float ligands '''

pc_address = 'C:/Users/a44a4/OneDrive/桌面/PHY_YRC/'
folder = date + '_v5_rebind_agonist/' + 'jb_' + \
    str(j_b) + '_c_' + str(c) + '_read_files/'
config_a_file = 'jb_' + str(j_b) + '_c_' + str(c) + '_whole_config_a_run_0.csv'
config_b_file = 'jb_' + str(j_b) + '_c_' + str(c) + '_whole_config_b_run_0.csv'
ligand_file = 'jb_' + str(j_b) + '_c_' + str(c) + \
    '_run_0_final_positions_of_remaining_floating_ligands.csv'


a_address = pc_address + folder + config_a_file
b_address = pc_address + folder + config_b_file
ligand_address = pc_address + folder + ligand_file

''' Get the rates '''
ron_dt = r_on * dt
roffzero_selfdt = r_off_zero_self * dt
roffone_selfdt = r_off_one_self * dt
roff_comparedt = r_off_compare * dt
roffzero_agonistdt = r_off_zero_agonist * dt
roffone_agonistdt = r_off_one_agonist * dt
raonedt = r_a_one * dt
rdzerodt = r_d_zero * dt


''' Calculate the following to speed up calculation '''
ron_dt_plus_rdzerodt = ron_dt + rdzerodt
#print('ron_dt_plus_rd_zero_dt', ron_dt_plus_rdzerodt)


''' 
Monte Carlo simulation after putting an agonist in the middle site: 
To avoid a cannot be updated when it's non-integer,
a = 0 represents a = 0 (inactive), a = 1 represents a = 1/N_k, a = 2 represents a = 2/N_k, ..., a = N_k represents a = 1 (fully-active)
'''


def f_MC_step(config_a, config_b, F, M):

    # update the position of the floating agonist
    if len(M) == 1:  # agonist floats above the lattice
        Ra = random.rand()
        if 0 < Ra <= 0.0625:
            M[0][0] = (M[0][0] - 1) % L_s
        elif 0.0625 < Ra <= 0.125:
            M[0][0] = (M[0][0] + 1) % L_s
        elif 0.125 < Ra <= 0.1875:
            M[0][1] = (M[0][1] - 1) % L_s
        elif 0.1875 < Ra <= 0.25:
            M[0][1] = (M[0][1] + 1) % L_s
        else:
            pass

    # update positions of floating self
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

    # update state of receptors
    for i in range(L):  # L = 5: length of receptor
        for j in range(L):
            x = random.choice(rec)
            y = random.choice(rec)
            r = random.rand()

            # What to do when reach empty receptor (b = 0)
            if config_b[x, y] == 0:
                # calculate the number of total floating ligands above this site
                n_of_xy = F.count([x, y]) + M.count([x, y])
                if n_of_xy == 0:  # if no ligand is above on this iste
                    if config_a[x, y] == 0:
                        pass
                    else:
                        current_a = config_a[x, y]
                        if r < rdzerodt:
                            config_a[x, y] = int(current_a - 1)

                else:  # n_of_xy >= 1: if ligand(s) float above this site
                    # if only an agonist floats above this site
                    if n_of_xy == 1 and M.count([x, y]) == 1:
                        if config_a[x, y] == 0:
                            if r < ron_dt:
                                config_b[x, y] = -1
                                M.remove([x, y])
                        else:
                            current_a = config_a[x, y]
                            if r > ron_dt_plus_rdzerodt:
                                pass
                            elif r < ron_dt:
                                config_b[x, y] = -1
                                M.remove([x, y])
                            else:
                                config_a[x, y] = int(current_a - 1)

                    # if only self float above this site
                    elif n_of_xy == F.count([x, y]):
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

                    else:  # if both agonist and selfs folat above this site
                        ron_n_dt = ron_dt * n_of_xy
                        # ron_dt for the rate of "an" agonist
                        if config_a[x, y] == 0:
                            if r > ron_n_dt:
                                pass
                            elif r < ron_dt:
                                config_b[x, y] = -1  # choose agonist to bind
                                M.remove([x, y])
                            else:
                                config_b[x, y] = 1  # choose self to bind
                                F.remove([x, y])
                        else:
                            current_a = config_a[x, y]
                            rdzerodt_plus_ron_n_dt = rdzerodt + ron_n_dt
                            if r > rdzerodt_plus_ron_n_dt:
                                pass
                            elif r < rdzerodt:
                                config_a[x, y] = int(current_a - 1)
                            elif ron_dt_plus_rdzerodt < r < rdzerodt_plus_ron_n_dt:
                                config_b[x, y] = 1  # choose self
                                F.remove([x, y])
                            else:
                                config_b[x, y] = -1  # choose agonist
                                M.remove([x, y])

            # What to do when reach bound receptor (b = 1)
            elif config_b[x, y] == 1:
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
                    # unbinding rate with coupling when a = current_a (a = current_a /N_k)
                    current_a = config_a[x, y]
                    roff_sdt = ron_dt * \
                        np.exp(e_b_s - current_a * denk) * neighbor_effect
                    if r > (roff_sdt + raonedt):
                        pass
                    elif r < roff_sdt:
                        config_b[x, y] = 0
                        F.append([x, y])
                    else:
                        config_a[x, y] = int(current_a + 1)

            # What to do when reach bound agonist (b = -1)
            else:
                nbs = config_b[x - 1, y]**2 + config_b[x + 1, y]**2 + \
                    config_b[x, y - 1]**2 + config_b[x, y + 1]**2 - 2
                neighbor_effect = np.exp(- four_J_b * nbs)

                if config_a[x, y] == 0:
                    # unbinding rate with coupling when a = 0
                    roffzero_adt = roffzero_agonistdt * neighbor_effect
                    if r > (roffzero_adt + raonedt):
                        pass
                    elif r < roffzero_adt:
                        config_b[x, y] = 0
                        M.append([x, y])
                    else:
                        config_a[x, y] = 1
                elif config_a[x, y] == N_k:
                    # unbinding rate with coupling when a = N_k (a = 1)
                    roffone_adt = roffone_agonistdt * neighbor_effect
                    if r < roffone_adt:
                        config_b[x, y] = 0
                        M.append([x, y])
                else:
                    # unbinding rate with coupling when a = current_a (a = current_a /N_k)
                    current_a = config_a[x, y]
                    roff_adt = ron_dt * \
                        np.exp(e_b_a - current_a * denk) * neighbor_effect
                    if r > (roff_adt + raonedt):
                        pass
                    elif r < roff_adt:
                        config_b[x, y] = 0
                        M.append([x, y])
                    else:
                        config_a[x, y] = int(current_a + 1)

    return config_a, config_b, F, M


''' statistics of receptors in each state '''


def dist(a, b):
    prob_dist = np.zeros((N_k+1, 2))
    for i in range(left_r, right_r):
        for j in range(left_r, right_r):
            aa = int(a[i][j])
            bb = int(b[i][j]**2)
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


''' read data as initial conditions　'''
# configurations of a and b, save in array


def read_data(original_file, save_data):
    with open(original_file) as file:
        initial_data = np.loadtxt(file, delimiter=',')
    for each_line in initial_data:
        save_data.append(each_line)

# ligands, save in list, since F.count([x,y]) only works for lists


def read_ligand_data(original_file, save_data):
    with open(original_file) as file:
        initial_data = np.loadtxt(file, delimiter=',')
    for each_line in initial_data:
        each_data = each_line.tolist()
        save_data.append(each_data)


#################################################################################################
''' main program : to see the effect after putting an agonist '''
############################################################################################
monitor_A, monitor_B = np.zeros(
    (n_runs, int(fsteps/s))), np.zeros((n_runs, int(fsteps/s)))
samp_acti = np.zeros((n_runs, int(asteps/sa)))
start = time.time()  # calculate execution time

'''
2. put an agonist in the middle site, and the lattice keeps evolving 
'''

for each_run in range(n_runs):

    logging.info(f"after putting an agonist: jb = {j_b}, c = {c}, run = {each_run}")

    # I test the effect of kinetic proofreading in another program, and record the final states. Then set the initial conditions of new trial here.

    # read data and save the initial condition
    con_a_data, con_b_data, ligand_data, agonist_data = [], [], [], []
    read_data(a_address, con_a_data)
    read_data(b_address, con_b_data)
    read_ligand_data(ligand_address, ligand_data)
    con_a, con_b, liga, agon = np.array(con_a_data), np.array(
        con_b_data), ligand_data, agonist_data

    # initial conditions
    config_a, config_b, floating_ligands, mobile_agonist = con_a, con_b, liga, agon
    config_a[middle_site, middle_site], config_b[middle_site,
                                                 middle_site] = 0, -1  # put the agonist which can rebind

    # changing variables are reset in new trial
    monitor_stats = np.zeros((N_k+1, 2))
    step, m = 0, 0

    # evolution after putting agonist on the site
    while step < fsteps:
        for each_s in range(s):
            f_MC_step(config_a, config_b, floating_ligands, mobile_agonist)
        monitor_A[each_run, m] = np.sum(config_a == N_k)
        monitor_B[each_run, m] = np.sum(config_b == 1) + np.sum(config_b == -1)
        monitor_stats = monitor_stats + dist(config_a, config_b)
        m += 1
        if step % (thermal_steps) == 0:
            logging.info(f"after put agonist step number = {step}")
            logging.info(f"prob of each state (simulation) = {monitor_stats/m}")
        step += s

    final_stats = rec_stat(config_a, config_b)
    logging.info(f"rebind agonist: jb = {j_b}, c = {c}, Run = {each_run}: final states of lattice = ")
    for i in range(L):
        logging.info(f"final_stats[i]")


    # after fsteps, sampling the prob of fully activation every sa data points
    monitor_stats = np.zeros((N_k+1, 2))
    step, ma = 0, 0
    while step < asteps:
        for each_sa in range(sa):
            f_MC_step(config_a, config_b, floating_ligands, mobile_agonist)
        step += sa
        ma += 1
        new_stat = dist(config_a, config_b)
        logging.info(f"rebind agonist: time average step number = {fsteps + step}")
        logging.info(f"prob of each state (sampling) = {new_stat}")
        # p(a = N_k, b = 0) + p(a = N_k, b = 1)
        samp_acti[each_run, (ma-1)] = np.sum(new_stat[N_k])

# average sampling data and save in a file (processed data)
aver_prob = np.array([np.mean(samp_acti[each_run])
                     for each_run in range(n_runs)])
all_aver_prob = aver_prob
np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_rebind_case_ensemble_average_prob.csv',
           all_aver_prob, delimiter=",", fmt='%.8f')

# save time evolution of each trial
evolution_a, evolution_b = monitor_A, monitor_B
np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_rebind_case_evolution_activity.csv',
           evolution_a, delimiter=",", fmt='%.1f')
np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_rebind_case_evolution_bindings.csv',
           evolution_b, delimiter=",", fmt='%.1f')

end = time.time()
logging.info(f"jb = {j_b}, c = {c} rebind case run time after putting an agonist: {end - start}")

