import json
from typing import Optional, List, Tuple, Dict

from rdflib import URIRef, Graph
from tqdm import tqdm

from .realworld_literal import RealWorldLiteral
from ..util_models import URIRelation


class RealWorldRule(object):
    """
    These rules contain literals with the actual URIs of the real world knowledge base instead of the synthetic one.
    """
    def __init__(self,
                 premise: List[RealWorldLiteral] = None,
                 conclusion: RealWorldLiteral = None,
                 rudik_premise: List[Dict[str, str]] = None,
                 rudik_premise_str: str = None,
                 rudik_conclusion: Dict[str, str] = None,
                 rudik_conclusion_str: str = None,
                 hashcode: int = None,
                 rule_type: bool = None,
                 graph_iri: str = None):
        """
        Create a rule from a premise and a conclusion. The premise can contain multiple literals, while the conclusion
        is only one literal.

        :param premise: list of literals that describe the premise
        :param conclusion: conclusion literal
        """
        self.premise: List[RealWorldLiteral] = premise or []
        self.conclusion: RealWorldLiteral = conclusion
        self.rudik_premise: List[Dict[str, str]] = rudik_premise
        self.rudik_premise_str: str = rudik_premise_str
        self.rudik_conclusion: Dict[str, str] = rudik_conclusion
        self.rudik_conclusion_str: str = rudik_conclusion_str
        self.hashcode: int = hashcode
        self.rule_type: bool = rule_type
        self.graph_iri: str = graph_iri

    def _to_amie_str(self):
        rule_string = ""

        # add premise literals
        for literal in self.premise:
            rule_string += f"{literal} "

        # add implication arrow
        rule_string += "=> "

        # add conclusion literal
        rule_string += str(self.conclusion)

        return rule_string

    def _to_rudik_str(self):
        return f"{self.rudik_premise_str} => {self.rudik_conclusion_str}"

    def __str__(self):
        return self._to_rudik_str()

    def full_query_pattern(self) -> str:
        query_pattern = ""
        for literal in self.premise:
            query_pattern += literal.sparql_patterns()

        if "?b" not in query_pattern and "?a" not in query_pattern:
            query_projection = "ask "
        else:
            # insert the selectors for subject and object into the select query if they exist in the query pattern
            query_projection = "select where "

            # the resulting query would look like "select ?a ?b ..." if both cases are true
            if "?b" in query_pattern:
                query_projection = query_projection.replace("select ", "select ?b ")
            if "?a" in query_pattern:
                query_projection = query_projection.replace("select ", "select ?a ")

        # build remaining part of the query and execute it
        query_pattern = "{" + query_pattern + "}"
        return query_projection + query_pattern

    def premise_patterns(self,
                         graph: Graph,
                         subject_uri: URIRef,
                         relation_uri: URIRef,
                         object_uri: URIRef) -> Tuple[str, Optional[RealWorldLiteral]]:
        """
        Creates the SPARQL pattern to filter the graph according to the premise of this rule (i.e., all literals in the
        premise).
        :param graph: the synthesized graph
        :param subject_uri: uri of the subject in the new fact
        :param relation_uri: uri of the relation in the new fact
        :param object_uri: uri of the object in the new fact
        :return: tuple of the full SPARQL pattern of the premise and the literal of the premise with a matching relation
        type as the new fact, if such a literal exists
        """
        # contains the concatenated SPARQL patterns of the literals, i.e. the SPARQL filter to match nodes that conform
        # with all literals in the premise
        patterns = ""

        # subject of a matching literal
        matched_literal_subject = None

        # object of a matching literal
        matched_literal_object = None

        # the literal that matches the new fact
        matched_literal = None

        # test if a literal in the premise handles the same relation that is in the new fact
        # save the literal and its subject and object if such an literal exists
        for literal in self.premise:
            literal_predicate = literal.relation
            if literal_predicate == relation_uri:
                matched_literal_subject = literal.literal_subject(escape_literal=False)
                matched_literal_object = literal.literal_object(escape_literal=False)
                matched_literal = literal
                break

        # concatenate the SPARQL pattern fo every literal to query nodes matching all literals
        # exclude the literal with a matching relation type since it is already satisfied by the new fact that will be
        # added
        for literal in self.premise:
            if literal.relation != relation_uri:
                patterns += literal.sparql_patterns()

        subject_entity = f"<{subject_uri}>"
        object_entity = f"<{object_uri}>"

        if matched_literal_subject is not None:
            patterns = patterns.replace(matched_literal_subject, subject_entity)

        if matched_literal_object is not None:
            patterns = patterns.replace(matched_literal_object, object_entity)

        return patterns, matched_literal

    def to_dict(self) -> dict:
        return {
            "graph_iri": self.graph_iri,
            "rule_type": self.rule_type,
            "hashcode": self.hashcode,
            "premise_triples": self.rudik_premise,
            "premise": self.rudik_premise_str,
            "conclusion_triple": self.rudik_conclusion,
            "conclusion": self.rudik_conclusion_str,
            "query_patten": self.full_query_pattern()
        }

    def is_negative(self):
        return self.rule_type

    def _produce_fact(self, subject_uri: URIRef, object_uri: URIRef) -> Tuple[URIRef, URIRef, URIRef]:
        """
        Given the subject and object URI of the premise produce a new fact (i.e., in a positive rule). The purpose
        of this method is to find out the order of premise subject and object in the conclusion.
        :param subject_uri: the subject in the premise
        :param object_uri: the object in the premise
        :return: the new fact that is produced by this rule
        """
        assert len(self.premise) == 1, "Its currently only possilbe to produce facts for rules with a single literals" \
                                       " in the premise."
        predicate = self.conclusion.relation
        premise = self.premise[0]
        if premise.is_literal_subject or premise.is_literal_object:
            # TODO: handle subject or object literal
            raise RuntimeError("Subject or object literals can't be handled when producing facts.")

        swap = premise.subject_id == self.conclusion.object_id and premise.object_id == self.conclusion.subject_id
        if swap:
            return object_uri, predicate, subject_uri
        else:
            assert (premise.subject_id == self.conclusion.subject_id
                    and premise.object_id == self.conclusion.object_id), f"Subject and object ids don't match " \
                f"in rule {self}"
            return subject_uri, predicate, object_uri

    def enforce(self, graph: Graph):
        if self.rule_type:
            self._enforce_positive(graph)
        else:
            self._enforce_negative(graph)

    def _enforce_positive(self, graph: Graph):
        if len(self.premise) == 1:
            self._enforce_single_literal(graph)

    def _enforce_single_literal(self, graph: Graph) -> Graph:
        predicate = self.premise[0].relation
        new_triples = []
        print("Producing new triples")
        for subject, _, object in tqdm(graph.triples((None, predicate, None))):
            new_triples.append(self._produce_fact(subject, object))
        print(f"Produced {len(new_triples)} new facts for rule {self}")

        graph_size = len(graph)
        print("Adding new triples to graph")
        for triple in tqdm(new_triples):
            graph.add(triple)
        print(f"Added {len(graph) - graph_size} new facts")

        return graph

    def _enforce_negative(self, graph: Graph):
        raise NotImplementedError

    @staticmethod
    def parse_amie(line: str) -> 'RealWorldRule':
        raise NotImplementedError

    @classmethod
    def parse_rudik(cls, rule_dict: dict) -> 'RealWorldRule':
        graph_iri = rule_dict["graph_iri"]
        rule_type = rule_dict["rule_type"]
        rudik_premise = rule_dict["premise_triples"]
        rudik_conclusion = rule_dict["conclusion_triple"]
        premise = []
        errors = []
        for triple in rudik_premise:
            try:
                literal = RealWorldLiteral.parse_rudik(triple, graph_iri)
                premise.append(literal)
            except RuntimeError as e:
                errors.append(e)

        conclusion = None
        try:
            conclusion = RealWorldLiteral.parse_rudik(rudik_conclusion, graph_iri)
        except RuntimeError as e:
            errors.append(e)

        if errors:
            error_message = "\n".join([f"\t{error}" for error in errors])
            raise RuntimeError(f"Dropping rule due to unparseable literals\n{error_message}")

        hashcode = rule_dict["hashcode"]
        return cls(premise=premise,
                   conclusion=conclusion,
                   rudik_premise=rudik_premise,
                   rudik_conclusion=rudik_conclusion,
                   hashcode=hashcode,
                   rule_type=rule_type,
                   graph_iri=graph_iri)
