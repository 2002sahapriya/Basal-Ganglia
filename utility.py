
import numpy as np


class Utility:
    @staticmethod
    def to_bundle(distributions, domain_phis):
        '''
        Convert a list of distributions into 512-dimensional SSP bundles.
        Parameters:
            - bundles: List of vectors representing a distribution over a domain space
        Returns:
            - b: List of vectors representing a set of 512-dimensional SSP version of the distribution supplied. 
        '''
        bundles = []
        for i in range(len(distributions)):
            bundle = np.einsum('n,nd->d', distributions[i].squeeze(), domain_phis)
            bundles.append(bundle)
        return bundles
    
    @staticmethod
    def to_scale(salience, normalized_distributions):
        '''
        Scale normalized vectors by salience values.
        Parameters:
            - scales: list of floating point numbers 
            - vectors: list of vectors to be scaled 
        Returns:
            - scaled: a list of vectors scaled by a numeric scalar value
        '''
        assert len(salience) == len(normalized_distributions), "The `salience` argument has length: {len(salience)}, and `normalized_distributions` has length: {len(normalized_distributions)}. Make sure both arguments are the same length."
        scaled = []
        for i in range(len(salience)):
            scaled.append(salience[i] * normalized_distributions[i])
        return scaled

    @staticmethod
    def to_normalize(distributions):
        '''
        Normalize each distribution to unit peak (max = 1).

        Parameters: 
            - distributions (list[nd.narray]): List of distributions to be normalized.
        Returns:
            - nd.narray: List of normalized distributions with peaks at 1 
        '''
        normalized = distributions / np.max(distributions, axis=1, keepdims=True)
        return normalized
    
    @staticmethod
    def to_bimodal(distributions):
        '''
        Combine consecutive pairs of unimodal distributions into bimodal distributions.
        
        To create N beta distributions, the length of distributions should be 2N. 
        Bimodal distributions are created by combining consequetive pairs of beta distributions
        biomodal = distributions[0] + distribution[1]
        '''
        assert len(distributions) % 2 == 0, "The list `distributions` should have an even length."
        biomodals = []
        for index in range(0, len(distributions), 2):
            biomodals.append(distributions[index] + distributions[index + 1])
        return biomodals
    

class ActionIterator:
    '''
    The ActionIterator class implements a simple cyclic action scheduler and 
    feature bundler for spatial semantic pointers (SSPs).

    It is designed for scenarios where multiple actions are represented as high-dimensional vectors (SSPs), 
    and only one action is emphasized at a time while others are assigned lower salience. 
    
    The class scales SSPs by action salience and bundles them into continuous representations 
    for downstream processing (e.g., input to a basal ganglia model).

    This has been copied and modified TODO
    '''
    def __init__(self, n_actions, normalized_vectors, domain_phis):
        '''
        Parameters:
            - n_actions (int): Number of discrete actions
            - normalized_vectors (list): List of normalized SSP vectors, each of shape (512,)
            - domain_phis (nd.narray, nd.ndarray): Basis matrix used for bundling SSPs back into 512-dimensional space. 
                                                   Shape: (N, 512) where N = size of domain
        '''
        self.n_actions = n_actions
        self.actions = np.ones(n_actions) * 0.1      # initial saliences
        self.vectors = normalized_vectors            # list of 512-dim SSPs
        self.domain_phis = domain_phis               # shape (N,512) for bundling

    def step(self, t):
        '''
        Compute the bundled action vectors at time step t.
        Paramters:
            - t (float): Current time step. Used to cyclically activate actions.
        Returns:
            - np.ndarray: Concatenated vector of shape (n_actions * 512,).
                          Represents the bundled salience-weighted action SSPs for the current time step.
        '''
        # rotate which action is "on"
        idx = int(t % self.n_actions)
        self.actions[:] = 0.5
        self.actions[idx] = 0.9
        # scale each SSP by its salience
        scaled = [self.actions[i] * self.vectors[i]
                  for i in range(self.n_actions)]
        # bundle each scaled SSP back into a 512-d vector via einsum
        bundles = [
            np.einsum('n,nd->d', scaled[i].squeeze(), self.domain_phis)
            for i in range(self.n_actions)
        ]
        return np.concatenate(bundles)              # shape (n_actions * 512,)