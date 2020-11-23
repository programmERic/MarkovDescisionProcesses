import hiive.mdptoolbox.example
import hiive.mdptoolbox.mdp


import gym
import numpy as np
import matplotlib.pyplot as plt

qlearn_iter = 10000

def obtain_forest():
    
    P, R = hiive.mdptoolbox.example.forest(S=1000, # [int, optional] The number of states, which should be an integer greater than 1. Default: 3
                                     r1=2000,# [float, optional] The reward when the forest is in its oldest state and action ‘Wait’ is performed. Default: 4.
                                     r2=1000, # [float, optional] The reward when the forest is in its oldest state and action ‘Cut’ is performed. Default: 2.
                                     p=0.002, # [float, optional] The probability of wild fire occurence, in the range ]0, 1[. Default: 0.1.
                                     is_sparse=False) # [bool, optional] If True, then the probability transition matrices will be returned in sparse format, otherwise they will be in dense format. Default: False.

    return P, R

def forest_value_iteration():
    P, R = obtain_forest()
    vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma=0.8, epsilon=1e-10, max_iter=100)
    forest_vi_stats = vi.run()
    return forest_vi_stats


def forest_policy_iteration():
    P, R = obtain_forest()
    pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=0.8, max_iter=100)
    forest_pi_stats = pi.run()
    return forest_pi_stats

def forrest_qlearning():
    P, R = obtain_forest()
    pi = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=0.8, n_iter=qlearn_iter,
                                        alpha=0.66,
                                        alpha_decay=0.99,
                                        alpha_min=0.001,
                                        epsilon=1.0,
                                        epsilon_min=0.1,
                                        epsilon_decay=0.99)
    forest_ql_stats = pi.run()
    return forest_ql_stats

def lake_value_iteration():
    P, R = convert_openai_env()
    vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma=0.99, epsilon=1e-50, max_iter=100)
    forest_vi_stats = vi.run()
    return forest_vi_stats

def lake_policy_iteration():
    P, R = convert_openai_env()
    pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=0.99, max_iter=100)
    lake_pi_stats = pi.run()
    return lake_pi_stats

def lake_qlearning():
    P, R = convert_openai_env()
    ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=0.01, n_iter=qlearn_iter,
                                        alpha=0.9,
                                        alpha_decay=0.99,
                                        alpha_min=0.001,
                                        epsilon=0.9,
                                        epsilon_min=0.1,
                                        epsilon_decay=0.99)
    lake_ql_stats = ql.run()
    return lake_ql_stats

def obtain_from_stats(stats, key):
    val = np.zeros(shape=(len(stats)))
    if key != "Policy":
        val[:] = [stat[key] for stat in stats]
    else:
        s_sum = len(stats)
        val[:] = [np.sum(stat[key]) for stat in stats]
    return val

def convergence_discount_rate(P, R, name):

    rates = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    # QLearning
    plt.figure()

    for r in rates:
        pi = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=r, n_iter=10000,
                                        alpha=0.25,
                                        alpha_decay=0.99,
                                        alpha_min=0.001,
                                        epsilon=0.25,
                                        epsilon_min=0.1,
                                        epsilon_decay=0.99)
        ql_stats = pi.run()
        mean_v = obtain_from_stats(ql_stats, "Mean V")
        plt.plot(mean_v, label=r)

    plt.title("Mean Value for varying Discount Rates ({}, QLearning)".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.legend(loc="best")
    plt.savefig("discount_rates_{}_qlearn.png".format(name))

    # Value Iteration
    plt.figure()

    for r in rates:
        vi = hiive.mdptoolbox.mdp.ValueIteration(P, R, gamma=r, epsilon=0.001, max_iter=300)
        vi_stats = vi.run()
        mean_v = obtain_from_stats(vi_stats, "Mean V")
        plt.plot(mean_v, label=r)

    plt.title("Mean Value for varying Discount Rates ({}, Value)".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.legend(loc="best")
    plt.savefig("discount_rates_{}_value.png".format(name))

    # Value Iteration
    plt.figure()

    for r in rates:
        pi = hiive.mdptoolbox.mdp.PolicyIteration(P, R, gamma=r, max_iter=30)
        pi_stats = pi.run()
        mean_v = obtain_from_stats(pi_stats, "Mean V")
        plt.plot(mean_v, label=r)

    plt.title("Mean Value for varying Discount Rates ({}, Policy)".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.legend(loc="best")
    plt.savefig("discount_rates_{}_policy.png".format(name))

def qlearning_alpha(P, R, name):

    alpha_rates = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    # QLearning
    plt.figure()
    
    for alpha in alpha_rates:
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=0.99, n_iter=qlearn_iter,
                                        alpha=alpha,
                                        alpha_decay=0.99,
                                        alpha_min=0.001,
                                        epsilon=0.33,
                                        epsilon_min=0.1,
                                        epsilon_decay=0.99)
        ql_stats = ql.run()
        mean_v = obtain_from_stats(ql_stats, "Mean V")
        plt.plot(mean_v, label=alpha)

    plt.title("Mean Value for varying Alpha Rates ({}, QLearning)".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.legend(loc="best")
    plt.savefig("alpha_rates_{}_qlearn.png".format(name))

def qlearning_epsilon(P, R, name):

    epsilon_rates = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]

    # QLearning
    plt.figure()

    if name=="Forest":
        alpha = 0.5
    else:
        alpha = 0.9
    
    for epsilon in epsilon_rates:
        ql = hiive.mdptoolbox.mdp.QLearning(P, R, gamma=0.99, n_iter=qlearn_iter,
                                        alpha=alpha,
                                        alpha_decay=0.99,
                                        alpha_min=0.001,
                                        epsilon=epsilon,
                                        epsilon_min=0.1,
                                        epsilon_decay=0.99)
        ql_stats = ql.run()
        mean_v = obtain_from_stats(ql_stats, "Mean V")
        plt.plot(mean_v, label=epsilon)

    plt.title("Mean Value for varying Epsilon Rates ({}, QLearning)".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.legend(loc="best")
    plt.savefig("epsilon_rates_{}_qlearn.png".format(name))

def plot_mean_v(vi_stats, pi_stats, name):

    mean_v_vi = obtain_from_stats(vi_stats, "Mean V")
    
    mean_v_pi = np.zeros(shape=(mean_v_vi.shape[0]))
    mean_v_pi = obtain_from_stats(pi_stats, "Mean V")
    #mean_v_pi[:mean_v.shape[0]] = mean_v
    #mean_v_pi[mean_v.shape[0]:] = mean_v_pi[mean_v.shape[0]-1]

    plt.figure()
    plt.title("{} MDP Mean Value versus iterations".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.plot(mean_v_vi, label="Value Iteration")
    plt.plot(mean_v_pi, label="Policy Iteration")
    plt.legend(loc="best")
    plt.savefig("value_{}_meanV.png".format(name))

def plot_policy_convergence(vi_stats, pi_stats, name):

    policy_vi = obtain_from_stats(vi_stats, "Policy")
    policy_pi = obtain_from_stats(pi_stats, "Policy")

    plt.figure()
    plt.title("{} MDP Policy versus iterations".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Policy Representation")
    plt.plot(policy_vi, label="Value Iteration")
    plt.plot(policy_pi, label="Policy Iteration")
    plt.legend(loc="best")
    plt.savefig("value_{}_policy.png".format(name))

def plot_qlearning_policy_convergence(ql_stats, name):

    policy_ql = obtain_from_stats(ql_stats, "Policy")

    plt.figure()
    plt.title("{} QLearning Policy versus iterations".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Policy Representation")
    plt.plot(policy_ql)
    plt.savefig("value_{}_policy.png".format(name))

def plot_mean_v_qlearning(q1_stats, name):

    mean_v_ql = obtain_from_stats(q1_stats, "Mean V")

    plt.figure()
    plt.title("{} MDP QLearning Mean Value versus iterations".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Mean V")
    plt.plot(mean_v_ql)
    plt.savefig("value_{}_meanV_qlearn.png".format(name))


def plot_time_per_iteration(vi_stats, pi_stats, name):

    time_v_vi = obtain_from_stats(vi_stats, "Time")
    time_v_pi = obtain_from_stats(pi_stats, "Time")

    plt.figure()
    plt.title("{} MDP Time versus iterations".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Time")
    plt.plot(time_v_vi, label="Value Iteration")
    plt.plot(time_v_pi, label="Policy Iteration")
    plt.legend(loc="best")
    plt.savefig("value_{}_time.png".format(name))

def plot_time_per_iteration_qlearning(q1_stats, name):

    time_v_ql = obtain_from_stats(q1_stats, "Time")

    plt.figure()
    plt.title("{} MDP QLearning Time versus iterations".format(name))
    plt.xlabel("Iteration")
    plt.ylabel("Time")
    plt.plot(time_v_ql)
    plt.savefig("value_{}_time_qlearn.png".format(name))

def convert_openai_env():
    '''
    Constructs the necessary T & R np arrays for mdptoolbox
    '''
    env = gym.make('FrozenLake8x8-v0')
    P = np.zeros(shape=(env.nA, env.nS, env.nS))
    R = np.zeros(shape=(env.nA, env.nS, env.nS))

    for action in range(env.nA):
        for state in range(env.nS):
            possibs = env.P[state][action]      
            for possibility in possibs:
                prob = possibility[0]          
                to_state = possibility[1]
                reward = possibility[2]
                done = possibility[3]

                P[action, state, to_state] += prob
                R[action, state, to_state] = reward
    return P, R   

    
  
if __name__ == "__main__":
    '''
    lake_pi_stats = lake_policy_iteration()
    lake_vi_stats = lake_value_iteration()
    plot_policy_convergence(lake_vi_stats, lake_pi_stats, name="Lake")
    
    forest_vi_stats = forest_value_iteration()
    forest_pi_stats = forest_policy_iteration()
    plot_policy_convergence(forest_vi_stats, forest_pi_stats, name="Forest")
    '''
    '''
    forest_ql_stats = forrest_qlearning()
    plot_qlearning_policy_convergence(forest_ql_stats, name="Forest")
    plot_mean_v_qlearning(forest_ql_stats, name="Forest")
    plot_time_per_iteration_qlearning(forest_ql_stats, name="Forest")
    '''
    
    lake_ql_stats = lake_qlearning()
    plot_qlearning_policy_convergence(lake_ql_stats, name="Lake")
    plot_mean_v_qlearning(lake_ql_stats, name="Lake")
    plot_time_per_iteration_qlearning(lake_ql_stats, name="Lake")
    

    '''
    P, R = obtain_forest()
    qlearning_alpha(P, R, "Forest")
    qlearning_epsilon(P, R, "Forest")
    
    P, R = convert_openai_env()
    qlearning_alpha(P, R, "Lake")
    qlearning_epsilon(P, R, "Lake")
    convergence_discount_rate(P, R, "Lake")
    '''
    '''
    P, R = convert_openai_env()
    convergence_discount_rate(P, R, "Frozen Lake")
    '''
    '''
    P, R = obtain_forest()
    convergence_discount_rate(P, R, "Forest")
    '''
    '''
    lake_vi_stats = lake_value_iteration()
    lake_pi_stats = lake_policy_iteration()
    
    

    plot_mean_v(lake_vi_stats, lake_pi_stats, "Frozen Lake")
    plot_time_per_iteration(lake_vi_stats, lake_pi_stats, "Frozen Lake")
    '''
    '''
    P, R = obtain_forest()
    convergence_discount_rate(P, R, "Forest")
    
    q1_stats = forrest_qlearning()
    plot_mean_v_qlearning(q1_stats)
    plot_time_per_iteration_qlearning(q1_stats)

    
    forest_vi_stats = forest_value_iteration()
    forest_pi_stats = forest_policy_iteration()

    plot_mean_v(forest_vi_stats, forest_pi_stats)
    plot_time_per_iteration(forest_vi_stats, forest_pi_stats)
    '''

    
    

