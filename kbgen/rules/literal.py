import re
from typing import Optional, Dict

from rdflib import URIRef

from ..util_models import URIRelation


class Literal(object):
    """
    Represents a triple in a rule.
    """

    def __init__(self,
                 relation_id: int,
                 literal_subject_id: int,
                 literal_object_id: int,
                 relation_to_id: Dict[URIRef, int] = None):
        """
        Create a literal from the ids of the triples and the relation dictionary.
        :param relation_id: the id of the relation in the literal
        :param literal_subject_id: id of the subject (number of the letter in the alphabet, i.e., a = 1, b = 2, ...)
        :param literal_object_id: id of the object (again a = 1, b = 2, ...)
        :param relation_to_id: dictionary pointing from relation URIs to the ids used in the models
        """
        self.literal_subject_id = literal_subject_id
        self.literal_object_id = literal_object_id
        self.relation = URIRelation(relation_id)
        self.relation_id_to_uri: Dict[int, URIRef] = None
        if relation_to_id is not None:
            self.relation_id_to_uri: Dict[int, URIRef] = {relation_id: relation_uri
                                                          for relation_uri, relation_id in relation_to_id.items()}

    def __str__(self):
        """
        Represent subject and object of the literal via their letters and the relation via its uri.
        :return: string of the literal triple separated by spaces
        """
        if self.relation_id_to_uri is not None:
            literal_relation = self.relation_id_to_uri.get(self.relation.id, str(self.relation.id))
        else:
            literal_relation = str(self.relation.id)

        return f"?{self.literal_subject}  {literal_relation}  ?{self.literal_object}"

    def sparql_patterns(self):
        """
        Return the SPARQL filter expression to match a query to this literal.
        """
        return f"?{self.literal_subject} <{self.relation.__str__()}> ?{self.literal_object} . "

    @property
    def literal_subject(self) -> str:
        """
        Replace subject id with its letter (i.e., 1 = a, 2 = b, ...).
        """
        return chr(self.literal_subject_id + 96)

    @property
    def literal_object(self) -> str:
        """
        Replace object id with its letter (i.e., 1 = a, 2 = b, ...).
        """
        return chr(self.literal_object_id + 96)

    @staticmethod
    def parse_amie(literal_string: str, relation_to_id: Dict[URIRef, int]) -> Optional['Literal']:
        """
        Parses a triple that is element of a rules premise or conclusion into a literal object/
        :param literal_string: an element of a premise or conclusion of a rule (i.e., a triple)
        :param relation_to_id: dictionary pointing from relation URIs to the ids used in the models
        :return: literal object containing the triple
        """
        # parse literal triple into subject, relation and object parts
        literal_string = literal_string.strip()
        literal_parts = [part for part in literal_string.split(" ") if part]
        assert len(literal_parts) == 3, f"Literal {literal_parts} had length {len(literal_parts)} instead of 3."

        # extract subject and object and assert that they have the form "?a", "?b", ...
        literal_subject = literal_parts[0]
        literal_object = literal_parts[2]
        subject_object_string = f"subject {literal_subject} or object {literal_object}"
        assert literal_subject.startswith("?") and literal_object.startswith("?"), f"Either {subject_object_string} " \
                                                                                   "of the literal did not start with ?"
        assert len(literal_subject) == len(literal_object) == 2, f"Either {subject_object_string} did not have length 2"

        # match the different possible prefixes of the relation (the default case is wikidata)
        literal_relation = literal_parts[1]
        if literal_relation.startswith("<http") or literal_relation.startswith("http"):
            relation_uri = re.match("<?(.+)>?", literal_relation).group(1)
            relation = URIRef(relation_uri)
        elif "dbo:" in literal_relation:
            relation_name = re.match("<?dbo:(.+)>?", literal_relation).group(1)
            relation = URIRef(f"http://dbpedia.org/ontology/{relation_name}")
        else:
            relation_name = re.match("<?(.+)>?", literal_relation).group(1)
            relation = URIRef(f"http://wikidata.org/ontology/{relation_name}")

        # only continue if either the relation dictionary does not exist or it exists and contains the relation of the
        # literal
        if relation_to_id is not None:
            if relation not in relation_to_id:
                return None
        else:
            relation_to_id = {relation: 0}

        # get ids of the triple elements and return literal object
        relation_id = relation_to_id[relation]
        literal_subject_id = ord(literal_subject[1]) - 96
        literal_object_id = ord(literal_object[1]) - 96

        return Literal(relation_id, literal_subject_id, literal_object_id, relation_to_id)

    @staticmethod
    def parse_rudik(literal_triple: Dict[str, str],
                    relation_to_id: Dict[URIRef, int],
                    graph_iri: str) -> Optional['Literal']:
        relation_str = literal_triple["predicate"]
        # TODO: handle the predicate if it's a simple comparison (e.g., <, >, <=, >=)
        # if graph_iri not in relation_str:
        #     return None

        relation = URIRef(literal_triple["predicate"])
        relation_id = relation_to_id[relation]

        identifier_to_id = {"subject": 0, "object": 1}
        subject_str = literal_triple["subject"]
        if graph_iri in subject_str:
            # TODO: handle literal subject (i.e., dbpedia uri)
            return None
        if subject_str in identifier_to_id:
            subject_id = identifier_to_id[subject_str]
        else:
            subject_id = int(subject_str[1:]) + 2

        object_str = literal_triple["object"]
        if graph_iri in object_str:
            # TODO: handle literal object (i.e., dbpedia uri)
            return None
        if object_str in identifier_to_id:
            object_id = identifier_to_id[object_str]
        else:
            object_id = int(object_str[1:]) + 2

        return Literal(relation_id, subject_id, object_id, relation_to_id)
