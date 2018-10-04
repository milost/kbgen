import re
from math import nan
from typing import Dict

from rdflib import URIRef

from rules import Literal, Rule


class RuleStats(Rule):
    def __init__(self, antecedents=None, consequents=None, head_cov=nan, std_confidence=nan, pca_confidence=nan, pos_expl=nan,
                 std_body_sz=nan, pca_body_sz=nan, func_var=nan, std_low_bd=nan, pca_low_bd=nan, pca_conf_est=nan):
        super(RuleStats, self).__init__(antecedents, consequents, std_confidence, pca_confidence)
        self.head_cov = head_cov
        self.pos_expl = pos_expl
        self.std_body_sz = std_body_sz
        self.pca_body_sz = pca_body_sz
        self.func_var = func_var
        self.std_low_bd = std_low_bd
        self.pca_low_bd = pca_low_bd
        self.pca_conf_est = pca_conf_est

    @staticmethod
    def parse_amie(line: str, relation_to_id: Dict[URIRef, int]):
        cells = line.split("\t")
        rule_string = cells[0]
        head_cov = float(cells[1].strip())
        std_conf = float(cells[2].strip())
        pca_conf = float(cells[3].strip())
        pos_expl = float(cells[4].strip())
        std_body_sz = float(cells[5].strip())
        pca_body_sz = float(cells[6].strip())
        # func_var = float(cells[7].strip())
        func_var = 0.0
        std_low_bd = float(cells[8].strip())
        pca_low_bd = float(cells[9].strip())
        pca_conf_est = float(cells[10].strip())
        assert "=>" in rule_string
        ant_cons = rule_string.split("=>")
        ant_cons = list(filter(None, ant_cons))
        ant_string = ant_cons[0].strip()
        con_string = ant_cons[1].strip()

        ant_string = re.sub("(\?\w+)\s+\?", "\g<1>|?", ant_string)
        con_string = re.sub("(\?\w+)\s+\?", "\g<1>|?", con_string)

        antecedents = []
        for ant in ant_string.split("|"):
            lit = Literal.parse_amie(ant, relation_to_id)
            if lit is None:
                return None
            antecedents.append(lit)

        consequents = []
        for con in con_string.split("|"):
            lit = Literal.parse_amie(con, relation_to_id)
            if lit is None:
                return None
            consequents.append(lit)

        return RuleStats(antecedents, consequents, head_cov, std_conf, pca_conf, pos_expl,
                         std_body_sz, pca_body_sz, func_var, std_low_bd, pca_low_bd, pca_conf_est)

