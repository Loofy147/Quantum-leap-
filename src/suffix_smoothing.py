import numpy as np

class SuffixSmoothing:
    """
    Implements Suffix Smoothing for Quantum Suffix Trees.
    Used for searching Quantum Error Correction (QEC) codes.
    Formula: P(t|wn) = lambda * P_ML(t|wn) + (1 - lambda) * P(t|wn-1)
    """
    def __init__(self, lmbda=0.8):
        self.lmbda = lmbda
        self.probabilities = {} # Stores P_ML(target | context)

    def train(self, observations):
        """
        observations: list of (context_suffix, target_tag)
        """
        counts = {}
        context_counts = {}

        for context, target in observations:
            # Build counts for all suffix levels
            for i in range(len(context) + 1):
                suffix = context[i:]
                counts[(suffix, target)] = counts.get((suffix, target), 0) + 1
                context_counts[suffix] = context_counts.get(suffix, 0) + 1

        # Calculate ML probabilities
        for (suffix, target), count in counts.items():
            self.probabilities[(suffix, target)] = count / context_counts[suffix]

    def get_smoothed_probability(self, context, target):
        """
        Recursive implementation of the successive abstraction formula.
        """
        # Base case: empty suffix
        if not context:
            return self.probabilities.get(('', target), 0.0)

        p_ml = self.probabilities.get((context, target), 0.0)

        # Recursive step: lambda * P_ML + (1-lambda) * P_smoothed(shorter_suffix)
        return self.lmbda * p_ml + (1 - self.lmbda) * self.get_smoothed_probability(context[1:], target)

    def search_qec_code(self, syndrome_sequence, code_candidates):
        """
        Searches for the best QEC code based on smoothed probabilities of error suffixes.
        """
        scores = {}
        for code in code_candidates:
            # Probability that the given syndrome sequence leads to the code's error pattern
            prob = self.get_smoothed_probability(syndrome_sequence, code)
            scores[code] = prob

        return max(scores, key=scores.get) if scores else None
