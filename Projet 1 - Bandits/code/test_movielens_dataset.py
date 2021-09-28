import matplotlib.pyplot as plt
from scipy.sparse.linalg import svds
import pandas as pd
from bandit import *

# param dataframe ratings : Dataframe de notes avec les utilisateurs en ligne et les films en colonne
# param int N : Nombre de notes minimum par film
# return la matrice des notes sans les films notes moins de N fois

def get_rating_matrix(ratings, N=100): 
    ratings.columns = ['UserID', 'MovieID', 'Rating', 'Timestamp']
    ratings_count = ratings.groupby(by='MovieID', as_index=True).size()
    top_ratings = ratings_count[ratings_count>=N]
    ratings_topN = ratings[ratings.MovieID.isin(top_ratings.index)]
    n_users = ratings_topN.UserID.unique().shape[0]
    n_movies = ratings_topN.MovieID.unique().shape[0]
    R_df = ratings_topN.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)
    R = R_df.values

    return R


# param matrix R : Matrice de notes avec les utilisateurs en ligne et les films en colonne
# param float context_part : Part (dans ]0,1[) d'utilisateurs utilisés pour le context
# param float train_part : Part (dans ]0,1[) des notes utilisés pour l'entraînement
# return : La matrice à transformer en contexte et les matrices d'entraînement et de test

def train_test_split(R, context_part, train_part):
    M = R[:round(context_part*R.shape[0]), ] 
    T = R[round(context_part*R.shape[0]):, ]
    mask = np.random.binomial(1, train_part*(T!=0), T.shape)
    M1 = T * mask 
    M2 = T * (1 - mask) 

    return M, M1, M2


# param matrix R : Matrice de notes avec les utilisateurs en ligne et les films en colonne
# param int d : Dimension de factorisation
# param float context_part : Part (dans ]0,1[) d'utilisateurs utilisés pour le context
# param float train_part : Part (dans ]0,1[) des notes utilisés pour l'entraînement
# param int nb_runs : Nombre d'entraînement par utilisateur (pour normaliser les résultats)
# param int nb_it : Nombre d'itération LinUCB par utilisateur
# return les rmse par utilisateur et totales sur les donnnes d'entrainement et de test et les regrets par utilisateur

def run_test(R, d, context_part, train_part, nb_runs, nb_it, l, b):
    M, M1, M2 = train_test_split(R, context_part, train_part)
    U, s, Vt = svds(M, k = d)

    # rmse par utilisateur
    train_err = [np.zeros(nb_it)] * M1.shape[0] 
    test_err = [np.zeros(nb_it)] * M1.shape[0]

    # rmse totale
    total_train_err = np.zeros(nb_it)
    total_test_err = np.zeros(nb_it)

    # regrets
    regrets = []
    tmp = 0
    k = 100
    for u in range(k):
        bandit = ExactBandit(M1[u,:])
        avg_regret_lin_ucb, theta = bandit.run_linear_ucb(Vt, nb_runs, nb_it, [l], [b])
        regrets.append(avg_regret_lin_ucb[(l, b)])
        ratings = theta[(l, b)] @ Vt
        
        if np.sum(M1[u,:] != 0) > 0:
            err = np.sum((ratings[:,M1[u,:] != 0] - M1[u,:][M1[u,:] != 0])**2, axis=1)
            train_err[u] = np.sqrt(err / np.sum(M1[u,:] != 0))
            total_train_err += err
        if np.sum(M2[u,:] != 0) > 0:
            err = np.sum((ratings[:,M2[u,:] != 0] - M2[u,:][M2[u,:] != 0])**2, axis=1)
            test_err[u] = np.sqrt(err / np.sum(M2[u,:] != 0))
            total_test_err += err
    
    total_train_err = np.sqrt(total_train_err / np.sum(M1[:k,:] != 0))
    total_test_err = np.sqrt(total_test_err / np.sum(M2[:k,:] != 0))

    return total_train_err, total_test_err, train_err, test_err, regrets


def main(ratings):
    R = get_rating_matrix(ratings)
    total_train_err = {}
    total_test_err = {}
    train_err = {}
    test_err = {}
    regrets = {}
    for i in np.arange(0.6, 0.81, 0.2):
        total_train_err[i], total_test_err[i], train_err[i], test_err[i], regrets[i] = run_test(R, 20, 0.6, i, 1, 400, 0.001, 250)


    plt.figure(figsize=(6,4))
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w']
    c=0
    for i in total_train_err:
        plt.plot(total_train_err[i][:200], label='Train {}%'.format(round(i, 1)), c=colors[c]);
        plt.plot(total_test_err[i][:200], label='Test {}%'.format(round(1-i, 1)), c=colors[c], linestyle='dashed');
        c+=1

    plt.legend(loc="upper right")
    plt.xlabel('Itérations')
    plt.ylabel("RMSE")
    plt.show()

    plt.figure()
    for i in regrets:
        r = []
        for t in range(len(regrets[i])):
            if(len(regrets[i][t]) > 0): 
                r.append(regrets[i][t]) 
     
        plt.plot(np.mean(r, axis = 0), label='Train {}%'.format(round(i,1)))
    plt.legend()
    plt.xlabel('Itérations')
    plt.ylabel("Regret cumulé")
    plt.show()

ratings = pd.read_csv('ml-latest-small/ratings.csv')
main(ratings)