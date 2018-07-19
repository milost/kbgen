import numpy as np

from rules import RuleSet


class RuleSetStats(RuleSet):
    def __init__(self, rules=None):
        super(RuleSetStats, self).__init__(rules)

    def avg(self, attname):
        values = []
        for rule in self.rules:
            values.append(rule.__getattribute__(attname))
        return np.mean(values), np.std(values)
