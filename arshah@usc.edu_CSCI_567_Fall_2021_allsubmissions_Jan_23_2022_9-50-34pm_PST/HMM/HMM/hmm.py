








from __future__ import print_function
import json
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(Z_t = s_j|Z_{t-1} = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: A dictionary mapping each observation symbol to its index 
        - state_dict: A dictionary mapping each state to its index
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array where alpha[i, t-1] = P(Z_t = s_i, X_{1:t}=x_{1:t})
                 (note that this is alpha[i, t-1] instead of alpha[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros([S, L])
        ######################################################
        # TODO: compute and return the forward messages alpha
        ######################################################
        for t in range(L):
            #for s in range(S):
            if t == 0:
                alpha[:, 0] = self.pi * self.B[:, O[0]]
                #break
            else:
                alpha[:, t] = self.B[:, O[t]] * (self.A.T @ alpha[:, t - 1])
                
        return alpha

        
    def backward(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array where beta[i, t-1] = P(X_{t+1:T}=x_{t+1:T} | Z_t = s_i)
                    (note that this is beta[i, t-1] instead of beta[i, t])
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros([S, L])
        #######################################################
        # TODO: compute and return the backward messages beta
        #######################################################
        for t in range(L - 1, -1, -1):
            for s in range(S):
                if t == L - 1:
                    beta[:, L-1] = 1
                    break
                else:
                    beta[s, t] = self.A[s, :] @ (self.B[:, O[t+1]] * beta[:, t + 1])
        
        return beta
                

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(X_{1:T}=x_{1:T})
        """
        
        #####################################################
        # TODO: compute and return prob = P(X_{1:T}=x_{1:T})
        #   using the forward/backward messages
        #####################################################
        prob = self.forward(Osequence)[:, 0].T @ self.backward(Osequence)[:, 0]
        return prob


    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - gamma: (num_state*L) A numpy array where gamma[i, t-1] = P(Z_t = s_i | X_{1:T}=x_{1:T})
		           (note that this is gamma[i, t-1] instead of gamma[i, t])
        """
        ######################################################################
        # TODO: compute and return gamma using the forward/backward messages
        ######################################################################
        gamma = self.forward(Osequence) * self.backward(Osequence) / self.sequence_prob(Osequence)
        return gamma

    
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array where prob[i, j, t-1] = 
                    P(Z_t = s_i, Z_{t+1} = s_j | X_{1:T}=x_{1:T})
        """
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        prob = np.zeros([S, S, L - 1])
        #####################################################################
        # TODO: compute and return prob using the forward/backward messages
        #####################################################################
        fm = self.forward(Osequence)
        bm = self.backward(Osequence)
        norm_cons = self.sequence_prob(Osequence)        
        for t in range(L-1):                        
            prob[:, :, t] =  fm[:, t].reshape(S, 1) * self.A * (self.B[:, O[t+1]] * bm[:, t+1]).reshape(1, S)/ norm_cons            
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden states (return actual states instead of their indices;
                    you might find the given function self.find_key useful)
        """
        path = []
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        d1 = np.zeros([S,L])
        d2 = np.zeros([S,L])
        path = [0] * L
        ################################################################################
        # TODO: implement the Viterbi algorithm and return the most likely state path
        ################################################################################
        for t in range(L):
            if t == 0:
                d1[:, 0] = self.pi * self.B[:, O[0]]
                d2[:, 0] = np.argmax(self.A * d1[:, 0].reshape(S, 1), axis=0)
            else:                
                d1[:, t] = self.B[:, O[t]] * np.max(self.A * d1[:, t-1].reshape(S, 1), axis=0)
                d2[:, t] = np.argmax(self.A * d1[:, t-1].reshape(S, 1), axis=0)
                        
        path[-1] = int(np.argmax(d1[:, -1]))
        for t in range(L-1, 0, -1):            
            path[t-1] = int(d2[path[t], t])        
        for i in range(L):
            path[i] = self.find_key(self.state_dict, path[i])

        return path


    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
