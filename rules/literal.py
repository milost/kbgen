import re

from rdflib import URIRef

from util_models import URIRelation


class Literal(object):
    def __init__(self, relation, arg1, arg2, rel_dict=None):
        self.arg1 = arg1
        self.arg2 = arg2
        self.relation = URIRelation(relation)
        self.rel_dict = None
        if rel_dict is not None:
            self.rel_dict = {v: k for k, v in rel_dict.items()}

    def __str__(self):
        if self.rel_dict is None or self.relation.id not in self.rel_dict:
            return f"?{chr(self.arg1+96)}  {str(self.relation.id)}  ?{chr(self.arg2+96)}"
        else:
            return f"?{chr(self.arg1+96)}  {self.rel_dict[self.relation.id]}  ?{chr(self.arg2+96)}"

    def sparql_patterns(self):
        return "?" + chr(self.arg1+96) + \
               " <" + self.relation.__str__() + "> " \
               "?" + chr(self.arg2+96) + " . "

    @staticmethod
    def parse_amie(literal_string, rel_dict):
        literal_string = literal_string.strip()
        args = literal_string.split(" ")
        args = list(filter(None, args))
        assert len(args) == 3
        arg1 = args[0]
        arg2 = args[2]
        assert arg1.startswith("?") and arg2.startswith("?") and len(arg1) == 2 and len(arg2) == 2

        if args[1].startswith("<http") or args[1].startswith("http"):
            rel_uri = re.match("<?(.+)>?", args[1]).group(1)
            rel = URIRef(rel_uri)
        elif "dbo:" in args[1]:
            rel_name = re.match("<?dbo:(.+)>?", args[1]).group(1)
            rel = URIRef("http://dbpedia.org/ontology/"+rel_name)
        else:
            rel_name = re.match("<?(.+)>?", args[1]).group(1)
            rel = URIRef("http://wikidata.org/ontology/"+rel_name)

        if rel_dict is not None:
            if rel not in rel_dict:
                return None
        else:
            rel_dict = {rel: 0}

        rel_id = rel_dict[rel]
        arg1_id = ord(arg1[1]) - 96
        arg2_id = ord(arg2[1]) - 96

        return Literal(rel_id, arg1_id, arg2_id, rel_dict)
