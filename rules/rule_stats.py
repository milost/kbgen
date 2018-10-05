import re
from math import nan
from typing import Dict, List

from rdflib import URIRef

from rules import Literal, Rule


class RuleStats(Rule):
    def __init__(self,
                 antecedents: List[Literal] = None,
                 consequents: List[Literal] = None,
                 head_coverage: float = nan,
                 standard_confidence: float = nan,
                 pca_confidence: float = nan,
                 positive_examples: float = nan,
                 std_body_size: float = nan,
                 pca_body_size: float = nan,
                 functional_variable: float = nan,
                 std_lower_bound: float = nan,
                 pca_lower_bound: float = nan,
                 pca_confidence_estimation: float = nan):
        """
        Creates a rule object that also contains the statistical information created by AMIE when the rule was
        extracted.
        The measures are further explained in the AMIE paper(http://resources.mpi-inf.mpg.de/yago-naga/amie/amie.pdf)

        :param antecedents: list of literals that describe the premise
        :param consequents: list of literals that describe the conclusion (should only contain one literal in AMIE)
        :param head_coverage: the proportion of pairs from the head relation that are covered by the predictions of the
                              rule
        :param standard_confidence: takes all facts that are not in the KB as negative evidence. Thus it is the ratio
                                    of its predictions that are in the kB
        :param pca_confidence: the confidence of the partial completeness assumption (PCA). It identifies more
                               productive rules than the other measures
        :param positive_examples: TODO: meaning (should be an int)
        :param std_body_size: TODO: meaning (should be an int)
        :param pca_body_size: TODO: meaning (should be an int)
        :param functional_variable: TODO: meaning (should be "?a" or "?b")
        :param std_lower_bound: TODO: meaning
        :param pca_lower_bound: TODO: meaning
        :param pca_confidence_estimation: TODO: meaning
        """
        super(RuleStats, self).__init__(antecedents, consequents, standard_confidence, pca_confidence)
        self.head_coverage = head_coverage
        self.positive_examples = positive_examples
        self.std_body_size = std_body_size
        self.pca_body_size = pca_body_size
        self.functional_variable = functional_variable
        self.std_lower_bound = std_lower_bound
        self.pca_lower_bound = pca_lower_bound
        self.pca_confidence_estimation = pca_confidence_estimation

    @staticmethod
    def parse_amie(line: str, relation_to_id: Dict[URIRef, int]):
        """
        Parses an AMIE rule from a line in a file, translates the relation URI to an id and creates a rule object.
        Also parses all the statistical information regarding the rule and saves it.
        :param line: line of a file that contains an AMIE rule
        :param relation_to_id: dictionary pointing from relation URIs to the ids used in the models
        :return: rule object containing the parsed AMIE rule
        """
        # extract fields from tsv-formatted AMIE rule
        cells = line.split("\t")
        rule_string = cells[0]
        head_coverage = float(cells[1].strip())
        std_confidence = float(cells[2].strip())
        pca_confidence = float(cells[3].strip())
        positive_examples = float(cells[4].strip())
        std_body_size = float(cells[5].strip())
        pca_body_size = float(cells[6].strip())
        # functional_variable = float(cells[7].strip())
        functional_variable = 0.0
        std_lower_bound = float(cells[8].strip())
        pca_lower_bound = float(cells[9].strip())
        pca_confidence_estimation = float(cells[10].strip())

        # split rule into premise and conclusion
        assert "=>" in rule_string, "Rule string does not contain \"=>\" substring!"
        premise, conclusion = [rule_part.strip() for rule_part in rule_string.split("=>") if rule_part]

        # TODO: why this replacement (matches "?[a-zA-Z0-9_]+<whitespace>+?" (i.e., relation begins with ?)
        premise = re.sub("(\?\w+)\s+\?", "\g<1>|?", premise)
        conclusion = re.sub("(\?\w+)\s+\?", "\g<1>|?", conclusion)

        # split premise into single literals (i.e., triples)
        antecedents = []
        for antecedent in premise.split("|"):
            literal = Literal.parse_amie(antecedent, relation_to_id)
            if literal is None:
                return None
            antecedents.append(literal)

        # split conclusion into single literals (i.e., triples)
        consequents = []
        for consequent in conclusion.split("|"):
            literal = Literal.parse_amie(consequent, relation_to_id)
            if literal is None:
                return None
            consequents.append(literal)

        return RuleStats(antecedents,
                         consequents,
                         head_coverage,
                         std_confidence,
                         pca_confidence,
                         positive_examples,
                         std_body_size,
                         pca_body_size,
                         functional_variable,
                         std_lower_bound,
                         pca_lower_bound,
                         pca_confidence_estimation)

