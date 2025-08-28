# CLASS FOR BASAL GANGLIA HELPER FUNCTIONS
import numpy as np


class Utility:
    @staticmethod
    def to_bundle(distributions, domain_phis):
        '''
        Convert distributions to 512D bundles.
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
        Scales the vectors by a numeric value 
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
        Takes in a list of vectors and scales them to all have their peaks at 1. 
        '''
        normalized = distributions / np.max(distributions, axis=1, keepdims=True)
        return normalized
    
    @staticmethod
    def to_bimodal(distributions):
        '''
        Creates bimodal beta distributions.
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
    def __init__(self, n_actions, normalized_vectors, domain_phis):
        self.n_actions = n_actions
        self.actions = np.ones(n_actions) * 0.1      # initial saliences
        self.vectors = normalized_vectors            # list of 512-dim SSPs
        self.domain_phis = domain_phis               # shape (N,512) for bundling

    def step(self, t):
        # rotate which action is "on"
        idx = int(t % self.n_actions)
        self.actions[:] = 0.5
        self.actions[idx] = 0.9

        # print(self.actions, idx)

        # print(f'At time t={t}, salience: {self.actions}, winning channel: {np.argmax(self.actions)}')

        # scale each SSP by its salience
        scaled = [self.actions[i] * self.vectors[i]
                  for i in range(self.n_actions)]

        # bundle each scaled SSP back into a 512-d vector via einsum
        bundles = [
            np.einsum('n,nd->d', scaled[i].squeeze(), self.domain_phis)
            for i in range(self.n_actions)
        ]

        # stack and flatten to one long vector
        return np.concatenate(bundles)              # shape (n_actions * 512,)