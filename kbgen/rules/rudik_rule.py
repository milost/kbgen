from typing import Dict, List, Tuple, Optional

from rdflib import URIRef, Graph

from ..util_models import URIRelation
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

    def validate(self,
                 graph: Graph,
                 subject_uri: URIRef,
                 relation_uri: URIRef,
                 object_uri: URIRef) -> bool:
        """
        Produces new facts according to this rule given a new input fact.
        :param graph: the synthesized graph
        :param subject_uri: uri of the subject in the new fact
        :param relation_uri: uri of the relation in the new fact
        :param object_uri: uri of the object in the new fact
        :return: a list of facts produced by this rule
        """
        # if there is only one literal in the premise the new fact is invalid if the premise exists
        if len(self.antecedents) == 1:
            premise_relation_uri = URIRelation.get_uri(self.antecedents[0].relation)
            premise_fact = None

            # the subject and object of the premise and the conclusion are the same entities
            if (
                self.antecedents[0].literal_subject_id == self.consequents[0].literal_subject_id
                and self.antecedents[0].literal_object_id == self.consequents[0].literal_object_id
            ):
                premise_fact = (subject_uri, premise_relation_uri, object_uri)
            # the subject and object of the premise are swapped in the conclusion
            elif (
                self.antecedents[0].literal_subject_id == self.consequents[0].literal_object_id
                and self.antecedents[0].literal_object_id == self.consequents[0].literal_subject_id
            ):
                premise_fact = (object_uri, premise_relation_uri, subject_uri)

            # the new fact is valid if the premise fact does not exist in the graph
            return premise_fact and premise_fact not in graph

        else:
            # there are multiple literals in the premise
            # to check for triples matching every literal, a sparql query is built from them

            # build the where part of the sparql query and find the literal matching the relation type of the input fact
            # if such a literal exists
            query_patterns, new_literal = self.antecedents_patterns(graph, subject_uri, relation_uri, object_uri)

            # if the patterns of the sparql query do not contain either the subject or the object, only query for
            # possible solutions to the query
            # an ask query only queries if the pattern has a solution, i.e. do any nodes match the pattern
            # it will return a yes/no answer
            if "?b" not in query_patterns and "?a" not in query_patterns:
                query_projection = "ask "
            else:
                # insert the selectors for subject and object into the select query if they exist in the query pattern
                query_projection = "select where "

                # the resulting query would look like "select ?a ?b ..." if both cases are true
                if "?b" in query_patterns:
                    query_projection = query_projection.replace("select ", "select ?b ")
                if "?a" in query_patterns:
                    query_projection = query_projection.replace("select ", "select ?a ")

            # build remaining part of the query and execute it
            query_patterns = "{" + query_patterns + "}"
            sparql_query = query_projection + query_patterns
            query_result = graph.query(sparql_query)

            # if the query result is not empty the premise is true and the fact can't be added
            return not query_result

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
