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