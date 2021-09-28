import matplotlib.pyplot as plt
from bandit import *


nb_arms = 10
nb_it = 10000
means = [np.random.random() for _ in range(nb_arms)]

avg_regret_eg = BernoulliBandit(means).run_epsilon_greedy(nb_runs=10, nb_it=nb_it, epsilon=[0, 0.01,0.05,0.1, 0.2, 0.5, 1])

for e in avg_regret_eg:
    plt.plot(range(nb_it), avg_regret_eg[e], label='EG epsilon={}'.format(e));
plt.legend()
plt.xlabel('iterations')
plt.ylabel('regret cumulé')
plt.title('Regret cumulé pour plusieurs valeurs de epsilon')
plt.show()


nb_arms = 10
nb_it = 10000
nb_runs = 10
K = 10
theta = np.random.uniform(-2/K, 2/K, (K))
context = np.random.uniform(-1, 1, (K, nb_arms))
means = theta @ context
bandit = GaussianBandit(means)

avg_regret_eg = bandit.run_epsilon_greedy(nb_runs, nb_it, [0.01])
avg_regret_ucb = bandit.run_ucb(nb_runs, nb_it)
avg_regret_lin_ucb, theta2 = bandit.run_linear_ucb(context, nb_runs, nb_it, [1.5,2], [1,2])

for e in avg_regret_eg:
    plt.plot(range(nb_it), avg_regret_eg[e], label='EG epsilon={}'.format(e));
plt.plot(range(nb_it), avg_regret_ucb, label='UCB');
for a in avg_regret_lin_ucb:
    plt.plot(range(nb_it), avg_regret_lin_ucb[a], label='LinUCB (lambda, beta) = {}'.format(a));
plt.legend()
plt.xlabel('Itérations')
plt.ylabel('Regret cumulé')
plt.title('Regret cumulé en fonction des différents algorithmes pour un problème à 10 bras suivant des lois gaussiennes')
plt.show()


nb_it = 10000
context = np.random.uniform(-1, 1, (K, nb_arms))

avg_regret_lin_ucb, theta2 = bandit.run_linear_ucb(context, nb_runs, nb_it, [0.5, 1,1.5,2,2.5,3], [4])
avg_regret_lin_ucb2, theta22 = bandit.run_linear_ucb(context, nb_runs, nb_it, [1.5], [1.1,2,4,8,10])

plt.figure(figsize=(15,7))
plt.subplot(121)
for a in avg_regret_lin_ucb:
    plt.plot(range(nb_it), avg_regret_lin_ucb[a], label='LinUCB (lambda, beta) = {}'.format(a));
plt.legend() 
plt.xlabel('Itérations')
plt.ylabel('Regret cumulé')
plt.title("Influence du paramètre lambda")
plt.subplot(122)
for a in avg_regret_lin_ucb2:
    plt.plot(range(nb_it), avg_regret_lin_ucb2[a], label='LinUCB (lambda, beta) = {}'.format(a));
plt.legend()
plt.xlabel('Itérations')
plt.ylabel('Regret cumulé')
plt.title("Influence du paramètre beta")
plt.show()