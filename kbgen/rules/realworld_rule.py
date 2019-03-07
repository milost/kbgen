import datetime
import random
from typing import Optional, List, Tuple, Dict

from rdflib import URIRef, Graph, Literal
from tqdm import tqdm

from .realworld_literal import RealWorldLiteral


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

    __repr__ = __str__

    def full_query_pattern(self, include_conclusion: bool = False) -> str:
        query_pattern = ""
        for literal in self.premise:
            query_pattern += literal.sparql_patterns()

        if include_conclusion:
            query_pattern += self.conclusion.sparql_patterns()

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
        if len(self.premise) == 1:
            return self._produce_single_literal_fact(subject_uri, object_uri)
        else:
            return subject_uri, self.conclusion.relation, object_uri

    def _produce_single_literal_fact(self, subject_uri: URIRef, object_uri: URIRef) -> Tuple[URIRef, URIRef, URIRef]:
        """
        Given the subject and object URI of the premise produce a new fact (i.e., in a positive rule). The purpose
        of this method is to find out the order of premise subject and object in the conclusion.
        :param subject_uri: the subject in the premise
        :param object_uri: the object in the premise
        :return: the new fact that is produced by this rule
        """
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

    def enforce(self, graph: Graph) -> Graph:
        if self.rule_type:
            return self._enforce_positive(graph)
        else:
            return self._enforce_negative(graph)

    def _enforce_positive(self, graph: Graph) -> Graph:
        if len(self.premise) == 1:
            return self._enforce_single_literal(graph)

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

    def _enforce_multi_literal(self, graph: Graph, reflexiveness: bool = False) -> Graph:
        query = self.full_query_pattern()
        print(f"Query for {self}: {query}")
        result = graph.query(query)

        new_triples = []
        print("Producing new triples")
        for subject, object in tqdm(result):
            if not reflexiveness and subject == object:
                continue
            fact = self._produce_fact(subject, object)
            new_triples.append(fact)
        print(f"Produced {len(new_triples)} new facts for rule {self}")

        graph_size = len(graph)
        print("Adding new triples to graph")
        for triple in tqdm(new_triples):
            graph.add(triple)
        print(f"Added {len(graph) - graph_size} new facts")

        return graph

    def _enforce_negative(self, graph: Graph):
        raise NotImplementedError

    def break_by_literal(self, graph: Graph, relation: URIRef):
        print(f"Breaking for {self} with literal {relation}")
        query = self.full_query_pattern(include_conclusion=True)
        print(f"Query for positive facts: {query}")
        result = list(graph.query(query))

    def break_by_birth_date(self,
                            graph: Graph,
                            birth_date_relation: URIRef,
                            break_chance: float,
                            threshold: datetime.date,
                            less_than: bool = True) -> Tuple[Graph, list]:

        query = self.full_query_pattern(include_conclusion=True)
        print(f"Query for {self}: {query}")
        result = list(graph.query(query))

        positive_facts = set(result)
        all_facts = list(graph.query(self.full_query_pattern()))

        # how many facts are true (in the ground truth)
        correctness_ratio = len(positive_facts) / len(all_facts)
        # how many facts we break as noise
        brokenness_ratio = 0
        facts_to_correctness = {self._produce_fact(fact[0], fact[1]): fact in positive_facts for fact in all_facts}

        for person, film in tqdm(result):
            birth_date_query = list(graph.triples((person, birth_date_relation, None)))
            if not birth_date_query:
                continue
            birth_date_literal: Literal = birth_date_query[0][2]
            if (
                (less_than and birth_date_literal.value < threshold) or
                (not less_than and birth_date_literal.value >= threshold)
            ):
                number = random.random()
                if number < break_chance:
                    fact = self._produce_fact(person, film)
                    graph.remove(fact)
                    brokenness_ratio += 1

        brokenness_ratio /= len(all_facts)
        oracle_data = [facts_to_correctness, correctness_ratio, brokenness_ratio]
        return graph, oracle_data

    def get_distribution(self, graph: Graph):
        print("Gathering objects")
        objects = set()
        for _, _, object in tqdm(graph.triples((None, self.premise[0].relation, None))):
            objects.add(object)
        print("Building distribution")
        distribution = {}
        for query_object in tqdm(objects):
            triples = graph.triples((None, self.conclusion.relation, query_object))
            num_founders = len(list(triples))
            if num_founders not in distribution:
                distribution[num_founders] = 0
            distribution[num_founders] += 1
        return distribution

    def evaluate_rule(self, graph: Graph, reflexiveness: bool = False):
        result = list(graph.query(self.full_query_pattern()))
        invalid_count = 0
        for subject, object in tqdm(result):
            if not reflexiveness and subject == object:
                continue
            invalid_count += self._produce_fact(subject, object) not in graph
        return 1.0 - (invalid_count / len(result))

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
                   rudik_premise_str=rule_dict["premise"],
                   rudik_conclusion=rudik_conclusion,
                   rudik_conclusion_str=rule_dict["conclusion"],
                   hashcode=hashcode,
                   rule_type=rule_type,
                   graph_iri=graph_iri)

    @classmethod
    def parse_amie(cls, line: Dict[str, str]) -> 'RealWorldRule':
        literal_length = 3

        rule_str = line["Rule"]
        assert "=>" in rule_str, "Rule string does not contain implication arrow (\"=>\")!"
        premise, conclusion = [string.strip() for string in rule_str.split("=>")]

        conclusion_triple = conclusion.split()
        assert conclusion_triple[0] == "?a" and conclusion_triple[2] == "?b", f"Conclusion of amie rule {rule_str} " \
            f"does not have the format ?a predicate ?b"
        conclusion = RealWorldLiteral.parse_amie(conclusion_triple)

        premise_elements = premise.split()
        assert len(premise_elements) % literal_length == 0, f"Premise {premise} number of elements is not divisible " \
            f"by {literal_length}"
        premise_literals: List[Tuple[str, str, str]] = list(zip(*[iter(premise_elements)] * literal_length))
        premise = [RealWorldLiteral.parse_amie(literal_triple) for literal_triple in premise_literals]

        id_to_role = {
            str(conclusion.literal_subject()): "subject",
            str(conclusion.literal_object()): "object"
        }

        # serialized rudik format (for later serialization of the rule)
        rudik_premise = []
        rudik_premise_str = []
        for literal in premise:
            triple, string = literal.serialize_to_rudik(id_to_role)
            rudik_premise.append(triple)
            rudik_premise_str.append(string)
        rudik_premise_str = " & ".join(rudik_premise_str)

        rudik_conclusion, rudik_conclusion_str = conclusion.serialize_to_rudik(id_to_role)

        hashcode = hash(f"{rudik_premise_str} => {rudik_conclusion_str}")

        return cls(premise=premise,
                   conclusion=conclusion,
                   rudik_premise=rudik_premise,
                   rudik_premise_str=rudik_premise_str,
                   rudik_conclusion=rudik_conclusion,
                   rudik_conclusion_str=rudik_conclusion_str,
                   hashcode=hashcode,
                   rule_type=True,
                   graph_iri=None)
