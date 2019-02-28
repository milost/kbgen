import json
import re
from typing import Dict, Optional, List, Tuple

from rdflib import URIRef, Graph

from .literal import Literal
from ..util_models import URIRelation


class Rule(object):
    """
    Rules are assumed to have their consequent always with first argument ?a and second ?b
    """
    def __init__(self,
                 antecedents: List[Literal] = None,
                 consequents: List[Literal] = None,
                 standard_confidence: float = 1.0,
                 pca_confidence: float = 1.0):
        """
        Create a rule from a premise and a conclusion along with two confidence scores produced by AMIE. The premise can
        contain multiple literals, while the conclusion only contains one literal.
        The measures are further explained in the AMIE paper(http://resources.mpi-inf.mpg.de/yago-naga/amie/amie.pdf)

        :param antecedents: list of literals that describe the premise
        :param consequents: list of literals that describe the conclusion (should only contain one literal in AMIE)
        :param standard_confidence: takes all facts that are not in the KB as negative evidence. Thus it is the ratio
                                    of its predictions that are in the kB
        :param pca_confidence: the confidence of the partial completeness assumption (PCA). It identifies more
                               productive rules than the other measures
        """
        self.antecedents: List[Literal] = antecedents or []
        self.consequents: List[Literal] = consequents or []
        self.standard_confidence = standard_confidence
        self.pca_confidence = pca_confidence

    def __str__(self):
        rule_string = ""

        # add premise literals
        for antecedent in self.antecedents:
            rule_string += antecedent.__str__() + "  "

        # add implication arrow
        rule_string += "=>  "

        # add conclusion literals
        for consequent in self.consequents:
            rule_string += consequent.__str__() + "  "

        return rule_string

    def full_query_pattern(self) -> str:
        query_pattern = ""
        for antecedent in self.antecedents:
            query_pattern += antecedent.sparql_patterns()

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

    def antecedents_patterns(self,
                             graph: Graph,
                             subject_uri: URIRef,
                             relation_uri: URIRef,
                             object_uri: URIRef) -> Tuple[str, Optional[Literal]]:
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
        for antecedent in self.antecedents:
            antecedent_relation_uri = antecedent.relation.uri
            if antecedent_relation_uri == relation_uri:
                matched_literal_subject = f"?{antecedent.literal_subject}"
                matched_literal_object = f"?{antecedent.literal_object}"
                matched_literal = antecedent
                break

        # concatenate the SPARQL pattern fo every literal to query nodes matching all literals
        # exclude the literal with a matching relation type since it is already satisfied by the new fact that will be
        # added
        for antecedent in self.antecedents:
            if antecedent.relation != relation_uri:
                patterns += antecedent.sparql_patterns()

        subject_entity = f"<{subject_uri}>"
        object_entity = f"<{object_uri}>"

        if matched_literal_subject is not None:
            patterns = patterns.replace(matched_literal_subject, subject_entity)

        if matched_literal_object is not None:
            patterns = patterns.replace(matched_literal_object, object_entity)

        return patterns, matched_literal

    def to_dict(self) -> dict:
        return {
            "pattern": self.full_query_pattern()
        }

    def to_rudik(self):
        # this can't be a top level import since that will cause circular imports
        from .rudik_rule import RudikRule
        conclusion_literal = self.consequents[0]
        id_to_role = {
            str(conclusion_literal.literal_subject): "subject",
            str(conclusion_literal.literal_object): "object"
        }

        def convert_param(param: str):
            if param not in id_to_role:
                v_index = len(id_to_role) - 2
                id_to_role[param] = f"v{v_index}"
            return id_to_role[param]

        def convert_literal(literal: Literal) -> dict:
            subject_param = str(literal.literal_subject)
            subject_param = convert_param(subject_param)

            predicate = str(literal.relation)
            object_param = str(literal.literal_object)
            object_param = convert_param(object_param)

            return {
                "subject": subject_param,
                "predicate": predicate,
                "object": object_param
            }

        rudik_premise = [convert_literal(literal) for literal in self.antecedents]
        rudik_conclusion = convert_literal(conclusion_literal)
        hashcode = hash(json.dumps(rudik_premise) + json.dumps(rudik_conclusion))

        return RudikRule(premise=self.antecedents,
                         conclusion=self.consequents,
                         rudik_premise=rudik_premise,
                         rudik_conclusion=rudik_conclusion,
                         hashcode=hashcode,
                         rule_type=True,
                         graph_iri=None)

    def is_negative(self):
        return False

    def produce(self,
                graph: Graph,
                subject_uri: URIRef,
                relation_uri: URIRef,
                object_uri: URIRef) -> List[Tuple[URIRef, URIRef, URIRef]]:
        """
        Produces new facts according to this rule given a new input fact.
        :param graph: the synthesized graph
        :param subject_uri: uri of the subject in the new fact
        :param relation_uri: uri of the relation in the new fact
        :param object_uri: uri of the object in the new fact
        :return: a list of facts produced by this rule
        """
        # contains the facts produced by this rule
        new_facts: List[Tuple[URIRef, URIRef, URIRef]] = []

        # QUESTION: apparently AMIE rules can only have one triple in their conclusion. Is this actually the case?

        # if there is only one literal in the premise, simply check if it matches
        # a new fact is only produced if both subject and object of the input fact also appear in the premise literal
        if len(self.antecedents) == 1:

            # relation of the (only) literal in the conclusion
            new_relation = self.consequents[0].relation
            if isinstance(new_relation, URIRelation):
                new_relation_uri = new_relation.uri
            else:
                new_relation_uri = URIRelation(new_relation).uri

            # if the subject and object of the premise and the conclusion are the same entities
            if (
                self.antecedents[0].literal_subject_id == self.consequents[0].literal_subject_id
                and self.antecedents[0].literal_object_id == self.consequents[0].literal_object_id
            ):
                new_facts.append((subject_uri, new_relation_uri, object_uri))

            # if the subject and object of the premise are swapped in the conclusion
            if (
                self.antecedents[0].literal_subject_id == self.consequents[0].literal_object_id
                and self.antecedents[0].literal_object_id == self.consequents[0].literal_subject_id
            ):
                new_facts.append((object_uri, new_relation_uri, subject_uri))

            return new_facts

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

            # relation type of the resulting triple
            new_relation = self.consequents[0].relation
            if isinstance(new_relation, URIRelation):
                new_relation_uri = self.consequents[0].relation.uri
            else:
                new_relation_uri = URIRelation(self.consequents[0].relation).uri

            # handle every possible projection of the query
            if "?a" in query_projection and "?b" in query_projection:
                # both subject and object for each of the new facts were queried

                # add every result tuple as a new fact with the relation of the conclusion
                for new_subject, new_object in query_result:
                    new_facts.append((new_subject, new_relation_uri, new_object))

            elif "?a" in query_projection:
                # only the subject for each of the new facts was queried

                # select the subject or the object of the premise as object for new fact depending on the naming
                # i.e., a subject_id == 2 represents a "b", therefore the subject would be the new object
                if new_literal.literal_subject_id == 2:
                    new_object = subject_uri
                else:
                    # the object in the premise was named "b"
                    new_object = object_uri

                # add every result subject with the previously determined object as new fact with the relation of the
                # conclusion
                for new_subject, in query_result:
                    new_facts.append((new_subject, new_relation_uri, new_object))

            elif "?b" in query_projection:
                # only the object for each of the new facts was queried

                # select the subject or the object of the premise as subject for new fact depending on the naming
                # i.e., a subject_id == 1 represents an "a", therefore the subject would be the new subject
                if new_literal.literal_subject_id == 1:
                    new_subject = subject_uri
                else:
                    # the object in the premise was named "a"
                    new_subject = object_uri

                # add every result object with the previously determined subject as new fact with the relation of the
                # conclusion
                for new_object, in query_result:
                    new_facts.append((new_subject, new_relation_uri, new_object))

            elif bool(query_result):
                # if the result is non empty, or an ask query response is yes

                # if the subject was named "a" and the object named "b", the new fact will have the same subject and
                # object. otherwise they are swapped
                if new_literal.literal_subject_id == 1:
                    new_subject = subject_uri
                else:
                    new_subject = object_uri

                if new_literal.literal_object_id == 2:
                    new_object = object_uri
                else:
                    new_object = subject_uri

                # add the new fact with the original subject and object (possibly swapped) and the relation of the
                # conclusion
                new_facts.append((new_subject, new_relation_uri, new_object))

            return new_facts

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
        raise NotImplementedError

    @staticmethod
    def parse_amie(line: str, relation_to_id: Dict[URIRef, int]) -> Optional['Rule']:
        """
        Parses an AMIE rule from a line in a file, translates the relation URI to an id and creates a rule object.
        :param line: line of a file that contains an AMIE rule
        :param relation_to_id: dictionary pointing from relation URIs to the ids used in the models
        :return: rule object containing the parsed AMIE rule
        """
        # extract fields from tsv-formatted AMIE rule
        cells = line.split("\t")
        rule_string = cells[0]
        std_confidence = float(cells[2].strip())
        pca_confidence = float(cells[3].strip())

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

        return Rule(antecedents, consequents, std_confidence, pca_confidence)
