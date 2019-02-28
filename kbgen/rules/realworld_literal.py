from typing import Dict, Union

from rdflib import URIRef


class RealWorldLiteral(object):
    """
    Represents a triple in a rule containing the original URIs instead of ids.
    """

    def __init__(self,
                 subject: Union[URIRef, int],
                 relation: URIRef,
                 object: Union[URIRef, int]):
        """
        Create a literal from the uris or ids of the triple.
        :param subject: either a subject literal or the parameter id (number of the letter in the alphabet, i.e., a = 1,
                        b = 2, ...). For Rudik a and b are used for subject and object and the parameters v0, v1, ...
                        become c, d, e, ...
        :param relation: id of the subject (number of the letter in the alphabet, i.e., a = 1, b = 2, ...)
        :param object: either a object literal or the parameter id
        """
        self.is_literal_subject: bool = isinstance(subject, URIRef)
        self.is_literal_object: bool = isinstance(object, URIRef)
        self.relation: URIRef = relation

        if self.is_literal_subject:
            self.subject: URIRef = subject
            self.subject_id: int = None
        else:
            self.subject: URIRef = None
            self.subject_id: int = subject

        if self.is_literal_object:
            self.object: URIRef = object
            self.object_id: int = None
        else:
            self.object: URIRef = None
            self.object_id: int = object

    def __str__(self):
        """
        Represent subject and object of the literal via their letters and the relation via its uri.
        :return: string of the literal triple separated by spaces
        """
        return f"{self.literal_subject(False)} {self.relation} {self.literal_object(False)}"

    __repr__ = __str__

    def sparql_patterns(self):
        """
        Return the SPARQL filter expression to match a query to this literal.
        """
        return f"{self.literal_subject()} <{self.relation}> {self.literal_object()} . "

    def literal_subject(self, escape_literal: bool = True) -> str:
        """
        Replace subject id with its letter (i.e., 1 = a, 2 = b, ...) if the subject is a variable. Otherwise, use the
        literal.
        :param escape_literal: if True and the object is a literal it will be escaped with chevrons.
        """
        if self.is_literal_subject:
            literal = str(self.subject)
            if escape_literal:
                literal = f"<{literal}>"
            return literal
        else:
            return f"?{chr(self.subject_id + 96)}"

    def literal_object(self, escape_literal: bool = True) -> str:
        """
        Replace object id with its letter (i.e., 1 = a, 2 = b, ...) if the object is a variable. Otherwise, use the
        literal.
        :param escape_literal: if True and the object is a literal it will be escaped with chevrons.
        """
        if self.is_literal_object:
            literal = str(self.object)
            if escape_literal:
                literal = f"<{literal}>"
            return literal
        else:
            return f"?{chr(self.object_id + 96)}"

    @staticmethod
    def parse_amie(literal_string: str) -> 'RealWorldLiteral':
        raise NotImplementedError

    @classmethod
    def parse_rudik(cls,
                    literal_triple: Dict[str, str],
                    graph_iri: str) -> 'RealWorldLiteral':
        subject_str = literal_triple["subject"]
        relation_str = literal_triple["predicate"]
        object_str = literal_triple["object"]

        # TODO: handle the predicate if it's a simple comparison (e.g., <, >, <=, >=)
        if graph_iri and graph_iri not in relation_str:
            raise RuntimeError(f"Can't parse literal with not yet supported comparison {relation_str}")
        relation = URIRef(relation_str)

        identifier_to_id = {"subject": 1, "object": 2}
        # the v parameters start with v0 while the resulting id of v0 should be 3 (= 0 + 3)
        v_parameter_offset = 3

        # the subject is a literal
        if graph_iri and graph_iri in subject_str:
            subject = URIRef(subject_str)
        # the subject is either the subject or the object parameter
        elif subject_str in identifier_to_id:
            subject = identifier_to_id[subject_str]
        # the subject is a v parameter (v0, v1, ...)
        else:
            subject = int(subject_str[1:]) + v_parameter_offset

        # the object is a literal
        if graph_iri and graph_iri in object_str:
            object = URIRef(object_str)
        # the object is either the subject or the object parameter
        elif object_str in identifier_to_id:
            object = identifier_to_id[object_str]
        # the object is a v parameter (v0, v1, ...)
        else:
            object = int(object_str[1:]) + v_parameter_offset

        return cls(subject, relation, object)
