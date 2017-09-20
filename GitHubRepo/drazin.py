# drazin.py
"""Volume 1: The Drazin Inverse.
Drew Henrichsen
Data and Algorithms
"""

import numpy as np
from scipy import linalg as la
import scipy as sp
import networkx


# Helper function for problems 1 and 2.
def index(A, tol=1e-5):
    """Computes the index of the matrix A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.allclose(la.det(A),0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k


def is_drazin(A, Ad, k):
    """Verifies that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        bool: True of Ad is the Drazin inverse of A, False otherwise.
    """
    check1 = np.allclose(np.dot(A,Ad),np.dot(Ad,A))
    
    
    A_power = np.linalg.matrix_power(A,k)
    A_power_plus = np.linalg.matrix_power(A,k+1)
    check2 = np.allclose(np.dot(A_power_plus,Ad),A_power)
    
    check3 = np.allclose(np.dot(Ad, np.dot(A,Ad)),Ad)
    return check1 and check2 and check3

def drazin_inverse(A, tol=1e-4):
    """Computes  the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        Ad ((n,n) ndarray): The Drazin inverse of A.
    """
    (n,n) = np.shape(A)
    f = lambda x: abs(x) > tol
    g = lambda x: abs(x) <= tol
    Q1,S,k1 = la.schur(A, sort=f)
    Q2,T,k2 = la.schur(A, sort=g)
    U = np.hstack((S[:,:k1],T[:,:n-k1]))
    U_neg_1 = la.inv(U)
    V = np.dot(np.dot(U_neg_1,A),U)
    Z = np.zeros((n,n),dtype = float)
    if k1 != 0:
        M_neg_1 = la.inv(V[:k1,:k1])
        Z[:k1,:k1] = M_neg_1
    return np.dot(U,np.dot(Z,U_neg_1))

def effective_res(A):
    """Computes the effective resistance for each node in a graph.

    Parameters:
        A ((n,n) ndarray): The adjacency matrix of an undirected graph.

    Returns:
        ER ((n,n) ndarray): A matrix of which the ijth entry is the effective
        resistance from node i to node j.
    """
    
    #instantiates the Laplacian
    D = np.diag(A.sum(axis = 1))
    L = D-A
    
    Final = np.zeros_like(A,dtype = float)
    I = np.eye(len(L))
    for j in xrange(len(L)):
        L_wkspce = np.copy(L)
        L_wkspce[j,:] = I[j,:]
        L_semiFinal = (drazin_inverse(L_wkspce))
        for n in xrange(len(L_semiFinal)):
            Final[j,n] = L_semiFinal[n,n]
        Final[j,j] = 0
    return Final

class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
        an adjacency matrix.
        
        Parameters:
            filename (str): The name of a file containing graph data.
        """
        
        workspace = np.zeros((33,33),dtype = float)
        workhorse = dict()
        visited = set()
        visited_ = []
        count = 0
        with open(filename,'r') as socnet:
            for line in socnet:
                friends = line.strip().split(',')
                for n in friends:
                    if n not in visited:
                        workhorse[n] = count 
                        count +=1
                        visited.add(n)
                        visited_.append(n)
                workspace[workhorse[friends[0]],workhorse[friends[1]]] +=1
                workspace[workhorse[friends[1]],workhorse[friends[0]]] +=1
        self.names = visited_
        self.friendships = workspace
        self.friend_distance = effective_res(workspace)
        self.index = workhorse


    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or for a
        particular node.
        
        Parameters:
            node (str): The name of a node in the network.
        
        Returns:
            node1, node2 (str): The names of the next nodes to be linked.
                Returned if node is None.
            node1 (str): The name of the next node to be linked to 'node'.
                Returned if node is not None.
        
        Raises:
            ValueError: If node is not in the graph.
        """

        if node is None:
            mask2 = self.friendships.astype(bool)
            mask1 = self.friend_distance > 0
            mask3 = mask1 ^ mask2
            minval = np.min(self.friend_distance[mask3])
            row,col = np.where(self.friend_distance==minval)
            return self.names[int(row)],self.names[int(col)]
        else:
            if node in self.names:# find a friend for our_friend
                our_friend = self.index[node]
                mask2 = self.friendships[:,our_friend].astype(bool)
                mask2[our_friend] = True
                workspace = np.copy(self.friend_distance[~mask2,our_friend])
                mask1 = workspace > 0
                result = np.min(workspace)
                to_return = np.where(self.friend_distance[:,our_friend] == result)             
                return self.names[int(to_return[0])]
            else:
                raise ValueError("That name is not in the network!")


    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
        adjacency matrix and the effective resistance matrix.

        Parameters:
            node1 (str): The name of a node in the network.
            node2 (str): The name of a node in the network.

        Raises:
            ValueError: If either node1 or node2 is not in the graph.
        """
        if node1 not in self.names or node2 not in self.names:
            raise ValueError("That name is not in the network!")
        else:
            # Find the indices corresponding to the desired names.
            i = self.index[node1]
            j = self.index[node2]
            self.friendships[i,j] = self.friendships[j,i] = 1
            self.friend_distances = effective_res(self.friendships)

