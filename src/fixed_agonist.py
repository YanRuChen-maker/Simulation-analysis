import numpy as np
from numpy import random
import matplotlib.pylab as plt
import time
# import csv  # if csv is used in this program
np.seterr(divide='ignore')

date = '0723'  # date of running program

''' choices of parameters '''
N_k = 2  # number of kinetic proofreading steps
delta_e = 2*N_k  # energy decreases by delta_epsilon after binding
j_b = 0.55  # coupling strengh between b states, it's 2.00723 in Onsager‘s sol. to 2d Ising model
c = 0.32  # concentration of self
e_b_s = -1  # epsilon_b, binding energy of self
e_b_a = -6  # epsilon_b, binding energy of agonist
denk = delta_e/N_k  # for faster calculation
four_J_b = 4 * j_b  # for faster calculation

''' system size and number of steps '''
L = 5  # length of the receptor lattice
Lsq = L**2  # total number of receptors
L_s = 21  # length of the whole lattice system
n_l = int(L_s*L_s*c)  # number of total ligands
left_r = int((L_s-L)/2)  # leftmost receptor
right_r = int(left_r + L)  # (rightmost + 1) receptor
middle_site = int(L_s//2)  # middle site of agonist
rec = [8, 9, 10, 11, 12]  # possible positions of receptors
dt = 2.5*(10**(-3))  # time interval
fsteps = 7.2*(10**5)  # number of MC steps
# after fsteps, number of MC steps for sampling the prob of activation
asteps = 6*(10**4)
n_runs = 100  # number of trials
s = 3000  # store time evolution process every s data points
sa = 3000  # after fsteps, store the prob of activation every sa data points
sn = 100  # store snapshot of receptors every sn data points
thermal_steps = 10*s  # store time evolution of thermalization every 10*s data points


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


''' address of reading files: a, b, float ligands '''

pc_address = 'C:/Users/rubby/OneDrive/桌面/PHY_YRC/'
folder = date + '_v5_fixed_agonist/' + 'jb_' + \
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
raonedt = r_a_one * dt
rdzerodt = r_d_zero * dt


''' 
Monte Carlo simulation after putting an agonist in the middle site: 
To avoid a cannot be updated when it's non-integer,
a = 0 represents a = 0 (inactive)
a = 1 represents a = 1/N_k
a = 2 represents a = 2/N_k
.
.
.
a = N_k represents a = 1 (fully-active)
'''


def f_MC_step(config_a, config_b, F):

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

            if x == middle_site and y == middle_site:  # when reach immobile agonist
                # only consider change of a since b = 1 never changes
                if config_a[x, y] == N_k:
                    pass
                else:
                    current_a = config_a[x, y]
                    if r < raonedt:
                        config_a[x, y] = int(current_a + 1)

            else:  # when reach other sites
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


''' save the snapshots of the receptors '''


def snapshot(a, b, snap_a, snap_b, MS):
    site = 0
    for i in range(left_r, right_r):
        for j in range(left_r, right_r):
            snap_a[MS, site] = a[i, j]
            snap_b[MS, site] = b[i, j]
            site += 1


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
''' main program : to see the effect after fixing an agonist '''
############################################################################################
monitor_A, monitor_B = np.zeros(
    (n_runs, int(fsteps/s))), np.zeros((n_runs, int(fsteps/s)))
snapshot_a, snapshot_b = np.zeros(
    (int(fsteps/sn), Lsq)), np.zeros((int(fsteps/sn), Lsq))
samp_acti = np.zeros((n_runs, int(asteps/sa)))


start = time.time()  # calculate execution time

'''
2. put an immobile agonist in the middle site, and the lattice keeps evolving 
'''

for each_run in range(n_runs):

    print('\n' + 'fixed agonist: jb = ' + str(j_b) +
          '_c = ' + str(c) + '_run = ' + str(each_run))

    # I test the effect of kinetic proofreading in another program, and record the final states. Then set the initial conditions of new trial here.

    # read data and save the initial condition
    con_a_data, con_b_data, ligand_data = [], [], []
    read_data(a_address, con_a_data)
    read_data(b_address, con_b_data)
    read_ligand_data(ligand_address, ligand_data)
    con_a, con_b, liga = np.array(
        con_a_data), np.array(con_b_data), ligand_data

    # initial conditions
    config_a, config_b, floating_ligands = con_a, con_b, liga
    config_a[middle_site, middle_site], config_b[middle_site,
                                                 middle_site] = 0, 1  # put the immobile agonist

    # changing variables are reset in new trial
    monitor_stats = np.zeros((N_k+1, 2))
    step, m = 0, 0

    # evolution after putting agonist on the site
    if each_run == 0:  # save snapshots of the 1st trial
        ms = 0
        while step < fsteps:
            for each_s in range(s):
                f_MC_step(config_a, config_b, floating_ligands)
                if each_s % sn == 0:
                    snapshot(config_a, config_b, snapshot_a, snapshot_b, ms)
                    ms += 1
            monitor_A[each_run, m] = np.sum(config_a == N_k)
            monitor_B[each_run, m] = np.sum(
                config_b == 1) + np.sum(config_b == -1)
            monitor_stats = monitor_stats + dist(config_a, config_b)
            m += 1
            if step % (thermal_steps) == 0:
                print("after put agonist step number = ", step)
                print("prob of each state (simulation) = ", monitor_stats/m)
            step += s

        # save the snapshots of the 1st trial
        snapshot_A, snapshot_B = snapshot_a, snapshot_b
        np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_Run_' + str(each_run) +
                   '_snapshot_a.csv', snapshot_A, delimiter=",", fmt='%.1f')
        np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_Run_' + str(each_run) +
                   '_snapshot_b.csv', snapshot_B, delimiter=",", fmt='%.1f')

    else:
        while step < fsteps:
            for each_s in range(s):
                f_MC_step(config_a, config_b, floating_ligands)
            monitor_A[each_run, m] = np.sum(config_a == N_k)
            monitor_B[each_run, m] = np.sum(
                config_b == 1) + np.sum(config_b == -1)
            monitor_stats = monitor_stats + dist(config_a, config_b)
            m += 1
            if step % (thermal_steps) == 0:
                print("after put agonist step number = ", step)
                print("prob of each state (simulation) = ", monitor_stats/m)
            step += s

    final_stats = rec_stat(config_a, config_b)
    print('fixed agonist: jb = ' + str(j_b) + '_c = ' + str(c) +
          '_Run = ' + str(each_run) + ': final states of lattice = ')
    for i in range(L):
        print(final_stats[i])


    # after fsteps, sampling the prob of fully activation every sa data points
    monitor_stats = np.zeros((N_k+1, 2))
    step, ma = 0, 0
    while step < asteps:
        for each_sa in range(sa):
            f_MC_step(config_a, config_b, floating_ligands)
        step += sa
        ma += 1
        new_stat = dist(config_a, config_b)
        print("fixed agonist: time average step number = ", fsteps + step)
        print("prob of each state (sampling) = ", new_stat)
        # p(a = N_k, b = 0) + p(a = N_k, b = 1)
        samp_acti[each_run, (ma-1)] = np.sum(new_stat[N_k])


# average sampling data and save in a file (processed data)
aver_prob = np.array([np.mean(samp_acti[each_run])
                     for each_run in range(n_runs)])
all_aver_prob = aver_prob
np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_fixed_case_ensemble_average_prob.csv',
           all_aver_prob, delimiter=",", fmt='%.8f')

# store time evolution of each trial
evolution_a, evolution_b = monitor_A, monitor_B
np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_fixed_case_evolution_activity.csv',
           evolution_a, delimiter=",", fmt='%.1f')
np.savetxt('jb_' + str(j_b) + '_c_' + str(c) + '_fixed_case_evolution_bindings.csv',
           evolution_b, delimiter=",", fmt='%.1f')

end = time.time()
print('jb = ' + str(j_b) + '_c = ' + str(c) +
      '_fixed case run time after　putting an agonist: ', (end - start))
