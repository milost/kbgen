from typing import List, Tuple

import numpy as np

from .rule import Rule
from .rule_set import RuleSet


class RuleSetStats(RuleSet):
    def __init__(self, rules: List[Rule] = None):
        """
        Creates a set of rules given a list of rules. Also contains functionality to calculate statistics for the rule
        set.
        :param rules: the AMIE rules that were parsed from a file
        """
        super(RuleSetStats, self).__init__(rules)

    def avg(self, attribute_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the mean and the standard deviation for the confidence values of the rules in this rule set. The
        confidence for which mean and standard deviation is calcuated is specified via the "attribute_name" parameter.
        :param attribute_name: name of the instance variable (std_confidence or pca_confidence) for which the average
                               is calculated.
        :return: tuple of the mean and the standard deviation of the specified value of the rule
        """
        values = []
        for rule in self.rules:
            values.append(rule.__getattribute__(attribute_name))
        return np.mean(values), np.std(values)
