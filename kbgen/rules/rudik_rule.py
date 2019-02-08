import json
from typing import Dict, List

from rdflib import URIRef

from .rule import Rule
from .literal import Literal


class RudikRule(Rule):
    def __init__(self,
                 premise: List[Literal] = None,
                 conclusion: List[Literal] = None,
                 rudik_premise: List[Dict[str, str]] = None,
                 rudik_conclusion: Dict[str, str] = None,
                 hashcode: int = None,
                 rule_type: bool = None):
        super(RudikRule, self).__init__(premise, conclusion)
        self.rudik_premise = rudik_premise
        self.rudik_conclusion = rudik_conclusion
        self.hashcode = hashcode
        self.rule_type = rule_type

    @classmethod
    def parse_rudik(cls, rule_dict: dict, relation_to_id: Dict[URIRef, int]) -> 'RudikRule':
        graph_iri = rule_dict["graph_iri"]
        rule_type = rule_dict["rule_type"]
        rudik_premise = rule_dict["premise_triples"]
        premise = [Literal.parse_rudik(triple, relation_to_id, graph_iri) for triple in rudik_premise]
        premise = [literal for literal in premise if literal is not None]
        rudik_conclusion = rule_dict["conclusion_triple"]
        conclusion = Literal.parse_rudik(rudik_conclusion, relation_to_id, graph_iri)
        assert conclusion is not None, f"Conlusion could not be parsed: {rule_dict}"
        hashcode = rule_dict["hashcode"]
        return cls(premise=premise,
                   conclusion=[conclusion],
                   rudik_premise=rudik_premise,
                   rudik_conclusion=rudik_conclusion,
                   hashcode=hashcode,
                   rule_type=rule_type)
