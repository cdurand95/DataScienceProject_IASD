import numpy as np
import math

class Bandit:
    def __init__(self, means):
        self.means = np.array(means)
        self.nb_pulls = np.zeros(len(means))
        self.rewards = np.zeros(len(means))
        self.regrets = []
    
    def reset(self):
        self.nb_pulls = np.zeros(len(self.means))
        self.rewards = np.zeros(len(self.means))
        self.regrets = []

    def K(self):
        return len(self.means)
    
    def pull(self, a):
        raise NotImplementedError

    def regret(self, a):
        raise NotImplementedError

    def get_choices():
        raise NotImplementedError
  

    def epsilon_greedy(self, n, epsilon):
        choices = self.get_choices()
    
        for i in range(n):
            if len(choices) == 0:
                return 

            if i == 0 or np.random.binomial(1, epsilon) == 1:
                a = np.random.choice(choices)
            else:  
                pulls = np.array(self.nb_pulls) 
                pulls[pulls == 0] = 1
                rewards = np.array(self.rewards)
                rewards[[not j in choices for j in range(self.K())]] = -math.inf
                    
                a = np.random.choice(np.where((rewards / pulls) == np.max(rewards / pulls))[0])
                     
            self.pull(a)
            self.regret(a)

  

    def ucb(self, n):

        upper_bounds = np.repeat(math.inf, self.K())
        choices = self.get_choices()
        upper_bounds[[not i in choices for i in range(self.K())]] = -math.inf
    
        for i in range(n):  
            if len(choices) == 0:
                return

            a = np.random.choice(np.where(upper_bounds == np.max(upper_bounds))[0])
            self.pull(a)
           
            self.regret(a)
            for k in choices:
                if(self.nb_pulls[k] > 0):
                    upper_bounds[k] = self.rewards[k] / self.nb_pulls[k] + np.sqrt(2 * np.log(i+1) / self.nb_pulls[k])

    def linear_ucb(self, n, context, l, beta):

        V = l * np.identity(context.shape[0])
        theta = [np.zeros(context.shape[0])] * n
        choices = self.get_choices()
        K = np.zeros((n, context.shape[0]))

        upper_bounds = np.sqrt(beta) * np.sqrt(np.diag(context.T @ np.linalg.inv(V) @ context))
        upper_bounds[[not i in choices for i in range(self.K())]] = -math.inf
    
       
        for i in range(n):
            if len(choices) == 0:
                return theta
                          
            a = np.random.choice(np.where(upper_bounds == np.max(upper_bounds))[0])
            reward = self.pull(a)
            
            self.regret(a)
            K[i] = reward * context[:,a]
            V = V + np.outer(context[:,a], context[:,a])
          
            theta[i] = np.linalg.inv(V) @ np.sum(K, axis=0)
            
            upper_bounds = theta[i].T @ context + np.sqrt(beta) * np.sqrt(np.diag(context.T @ np.linalg.inv(V) @ context))

        return theta

    def run_epsilon_greedy(self, nb_runs=1, nb_it=1000, epsilon=[0]):
        avg_regret = {}
        for e in epsilon:
            regret = []
            for run in range(nb_runs):
                self.epsilon_greedy(nb_it, e)
                regret.append(np.cumsum(self.regrets))
                self.reset()

            avg_regret[e] = np.mean(np.array(regret), axis=0)

        return avg_regret

    def run_ucb(self, nb_runs=1, nb_it=1000):
        regret = []
        for run in range(nb_runs):
            self.ucb(nb_it)
            regret.append(np.cumsum(self.regrets))
            self.reset()

        avg_regret = np.mean(np.array(regret), axis=0)

        return avg_regret

    def run_linear_ucb(self, context, nb_runs=1, nb_it=1000, lambdas=[1], beta=[1]):
        avg_regret = {}
        theta = {}
        for l in lambdas:
            for b in beta:
                regret = []
                for run in range(nb_runs):
                    theta[(l,b)] = self.linear_ucb(nb_it, context, l, b)
                    regret.append(np.cumsum(self.regrets))
                    self.reset()

                avg_regret[(l,b)] = np.mean(np.array(regret), axis=0)

        return avg_regret, theta



class RandomBandit(Bandit):
    def __init__(self, means):
        super(RandomBandit, self).__init__(means)

    def pull(self, a):
        raise NotImplementedError

    def get_choices(self):
        return np.arange(self.K())

    def regret(self, a):
        self.regrets += [max(self.means) - self.means[a]]

class BernoulliBandit(RandomBandit):
    
    def __init__(self, means):
        super(BernoulliBandit, self).__init__(means)
    
    def pull(self, a):
        reward = np.random.binomial(1, self.means[a])
        self.nb_pulls[a] += 1
        self.rewards[a] += reward

        return reward

class GaussianBandit(RandomBandit):
    
    def __init__(self, means):
        super(GaussianBandit, self).__init__(means)
    
    def pull(self, a):
        reward = np.random.normal(self.means[a])
        self.nb_pulls[a] += 1
        self.rewards[a] += reward
        return reward

class ExactBandit(Bandit):
    def __init__(self, values):
        super(ExactBandit, self).__init__(values)
    
    def pull(self, a):
        reward = self.means[a]
        self.nb_pulls[a] += 1
        self.rewards[a] += reward
        return reward

    def get_choices(self): # bras dont on connait la vraie valeur
        return np.arange(self.K())[self.means != 0] 

    def regret(self, a):
        self.regrets += [max(self.means) - self.means[a]]