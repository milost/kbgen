from typing import Dict, List, Tuple, Optional

from rdflib import URIRef, Graph

from .rule import Rule
from .literal import Literal


class RudikRule(Rule):
    def __init__(self,
                 premise: List[Literal] = None,
                 conclusion: List[Literal] = None,
                 rudik_premise: List[Dict[str, str]] = None,
                 rudik_conclusion: Dict[str, str] = None,
                 hashcode: int = None,
                 rule_type: bool = None,
                 graph_iri: str = None):
        super(RudikRule, self).__init__(premise, conclusion)
        self.rudik_premise = rudik_premise
        self.rudik_conclusion = rudik_conclusion
        self.hashcode = hashcode
        self.rule_type = rule_type
        self.graph_iri = graph_iri

    def to_dict(self):
        return {
            "graph_iri": self.graph_iri,
            "rule_type": self.rule_type,
            "hashcode": self.hashcode,
            "premise_triples": self.rudik_premise,
            "conclusion_triple": self.rudik_conclusion
        }

    def to_rudik(self):
        return self

    def is_negative(self):
        return not self.rule_type

    def produce(self,
                graph: Graph,
                subject_uri: URIRef,
                relation_uri: URIRef,
                object_uri: URIRef) -> List[Tuple[URIRef, URIRef, URIRef]]:
        """
        If this is a negative rule don't return anything (since negative rules don't produce new facts). Otherwise
        just use the normal produce implementation.
        """
        if self.rule_type:
            return super(RudikRule, self).produce(graph, subject_uri, relation_uri, object_uri)
        else:
            return []

    @classmethod
    def parse_rudik(cls, rule_dict: dict, relation_to_id: Dict[URIRef, int]) -> 'RudikRule':
        graph_iri = rule_dict["graph_iri"]
        rule_type = rule_dict["rule_type"]
        rudik_premise = rule_dict["premise_triples"]
        rudik_conclusion = rule_dict["conclusion_triple"]
        premise = []
        errors = []
        for triple in rudik_premise:
            try:
                literal = Literal.parse_rudik(triple, relation_to_id, graph_iri)
                premise.append(literal)
            except RuntimeError as e:
                errors.append(e)
        try:
            conclusion = Literal.parse_rudik(rudik_conclusion, relation_to_id, graph_iri)
        except RuntimeError as e:
            errors.append(e)

        if errors:
            error_message = "\n".join([f"\t{error}" for error in errors])
            raise RuntimeError(f"Dropping rule due to unparseable literals\n{error_message}")

        hashcode = rule_dict["hashcode"]
        return cls(premise=premise,
                   conclusion=[conclusion],
                   rudik_premise=rudik_premise,
                   rudik_conclusion=rudik_conclusion,
                   hashcode=hashcode,
                   rule_type=rule_type,
                   graph_iri=graph_iri)
