# pagerank.py
"""Volume 1: The Page Rank Algorithm.
Drew Henrichsen
Algorithm Design and Optimization
"""

import numpy as np
import scipy.sparse as spsp
from scipy import linalg as la
import scipy as sp

def to_matrix(filename, n):
    """Returns the nxn adjacency matrix described by the data from the file.

    Parameters:
        datafile (str): The name of a .txt file describing a directed graph.
        Lines describing edges should have the form '<from node>\t<to node>\n'.
        The file may also include comments.
    n (int): The number of nodes in the graph described by datafile

    Returns:
        A SciPy sparse dok_matrix.
    """
    preworkspace = np.zeros((n,n))
    workspace = spsp.dok_matrix(preworkspace)

    with open(filename) as f:
        for line in f:
            preresult = line.strip().split()
            try:
                result = map(int,preresult)
            except ValueError:
                continue
            workspace[result[0],result[1]] +=1
    return workspace


def calculateK(A,N):
    """Normallizes the columns of the adjacency matrix, calls the normallized matrix K

    Parameters:
        A (ndarray): adjacency matrix of an array
        N (int): the datasize of the array

    Returns:
        K (ndarray)
    """
    for n in A:
        if sum(n) == 0:
            for p in xrange(len(n)):
                n[p] = 1.
    Diagonal = np.sum(A,axis = 1)
    
    return A.T/Diagonal.astype(float)


def iter_solve(adj, N=None, d=.85, tol=1E-5):
    """Returns the page ranks of the network described by 'adj'.
    performs the PageRank algorithm repeatedly until the error is less than 'tol'.

    Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restricts the computation to the first 'N' nodes of the graph.
            If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stops iterating when the change in approximations to the
            solution is less than 'tol'.

    Returns:
        The approximation to the steady state.
    """
    p0 = []
    if N is None:
        length = len(adj)
        K = calculateK(adj,len(adj))
        p0 = [np.random.rand() for p in xrange(length)]
        p1 = d*np.dot(K,p0) + (1-d)/len(adj)
        while (la.norm(p1-p0) > tol):
            p0 = p1
            p1 = d*np.dot(K,p0) + (1-d)
        return p1/len(adj)
    else:
        adj = adj[:N,:N]
        length = len(adj)
        K = calculateK(adj,len(adj))
        p0 = [np.random.rand() for p in xrange(length)]
        p1 = d*np.dot(K,p0) + (1-d)/len(adj)
        while (la.norm(p1-p0) > tol):
            p0 = p1
            p1 = d*np.dot(K,p0) + (1-d)/len(adj) * np.ones(len(adj))
        return p1/len(adj)



def eig_solve(adj, N=None, d=.85):
    """Returns the page ranks of the network described by 'adj'. Use SciPy's
    eigenvalue solver to calculate the steady state of the PageRank algorithm


    Parameters:
        adj (ndarray): The adjacency matrix of a directed graph.
        N (int): Restrict the computation to the first 'N' nodes of the graph.
            If N is None (default), use the entire matrix.
        d (float): The damping factor, a float between 0 and 1.
        tol (float): Stop iterating when the change in approximations to the
            solution is less than 'tol'.

    Returns:
        The approximation to the steady state.
    """
    a,b = adj.shape
    if N == None:
        N = b
    E = np.ones(N)
    A = adj[:N,:N]
    K = calculateK(A,N)
    work = d*K + float(1.-d)/N*np.ones_like(A)
    vals, vects = la.eig(work)
    to_return = np.argmax(vals)
    return np.real(vects[:,to_return])/np.sum(vects[:,to_return])
    #determine how much of the adjacency matrix we're dealing with.


def team_rank(filename='ncaa2013.csv'):
    """Use iter_solve() to predict the rankings of the teams in a given
    dataset of games. The dataset should have two columns, representing
    winning and losing teams. Each row represents a game, with the winner on
    the left, loser on the right. Parses this data to create the adjacency
    matrix, and feeds this into the solver to predict the team ranks.

    Parameters:
        filename (str): The name of the data file.
    Returns:
        ranks (list): The ranks of the teams from best to worst.
        teams (list): The names of the teams, also from best to worst.
    """
    visited = set()
    workhorse = dict()
    games = dict()
    count = 0
    length = 347# This is the number of teams in the 5000 games
    workspace = np.zeros((length,length))
    with open('./ncaa2013.csv','r') as ncaafile:
        ncaafile.readline()
        for line in ncaafile:
            teams = line.strip().split(',')
	    if teams[0] in games.keys():
		games[teams[0]].append(teams[1])
	    else:
		games[teams[0]] = [teams[1]]
	    if teams[1] not in games.keys():
		games[teams[1]] = list()
    length = len(games.keys())
    workspace = np.zeros((length,length))
    count = 0
    num_team = dict()
    team_num = dict()
    for p in games.keys():
	team_num[p] = count
	num_team[count] = p
	count +=1
    
    for winner in games.keys():
	for loser in games[winner]:
	    workspace[team_num[loser],team_num[winner]] = 1
	
    result =iter_solve(workspace,d = .70)
    order = np.argsort(result)
    final = []
    ranks = [n+1 for n in xrange(length)]
    for i in order:
	final.append(num_team[i])
    return np.sort(result)[::-1], final[::-1]

if __name__ == "__main__":
    workspace = to_matrix("matrix.txt",10).toarray()
    #print workspace
    #print calculateK(workspace,len(workspace))
    #print iter_solve(workspace, N=None, d=.85, tol=1E-5)
    #print eig_solve(workspace, N = None, d=.85)
    print team_rank()
    
