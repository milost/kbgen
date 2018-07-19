import re

from rdflib import URIRef

from rules import Literal
from util_models import URIRelation


class Rule(object):
    """
    Rules are assumed to have their consequent always with first argument ?a and second ?b
    """
    def __init__(self, antecedents=None, consequents=None, std_conf=1.0, pca_conf=1.0):
        self.antecedents = antecedents or []
        self.consequents = consequents or []
        self.std_conf = std_conf
        self.pca_conf = pca_conf

    def __str__(self):
        rule_str = ""
        for ant in self.antecedents:
            rule_str += ant.__str__() + "  "
        rule_str += "=>  "
        for con in self.consequents:
            rule_str += con.__str__() + "  "
        return rule_str

    def antecedents_sparql(self):
        patterns = ""
        for ant in self.antecedents:
            patterns += ant.sparql_patterns() + " . "
        return patterns

    def antecedents_patterns(self, g, s, p, o):
        patterns = ""
        arg_s = None
        arg_o = None
        new_lit = None
        p_uri = URIRef(p)
        for ant in self.antecedents:
            ant_p_uri = ant.relation.uri
            if ant_p_uri == p_uri:
                arg_s = "?"+chr(ant.arg1+96)
                arg_o = "?"+chr(ant.arg2+96)
                new_lit = ant
                break

        for ant in self.antecedents:
            if ant.relation != p:
                patterns += ant.sparql_patterns()

        s_ent = "<"+s+">"
        o_ent = "<"+o+">"

        patterns = patterns.replace(arg_s, s_ent) if arg_s is not None else patterns
        patterns = patterns.replace(arg_o, o_ent) if arg_s is not None else patterns

        return patterns, new_lit

    def produce(self, g, s, p, o):
        ss, ps, os = [], [], []
        if len(self.antecedents) == 1:
            r_j = self.consequents[0].relation
            if isinstance(r_j, URIRelation):
                new_p = r_j.uri
            else:
                new_p = URIRelation(r_j).uri

            if self.antecedents[0].arg1 == self.consequents[0].arg1 and \
               self.antecedents[0].arg2 == self.consequents[0].arg2:
                ss.append(s), ps.append(new_p), os.append(o)

            if self.antecedents[0].arg1 == self.consequents[0].arg2 and \
               self.antecedents[0].arg2 == self.consequents[0].arg1:
                ss.append(o), ps.append(new_p), os.append(s)
            return zip(ss, ps, os)
        else:
            patterns, new_lit = self.antecedents_patterns(g, s, p, o)

            if "?b" not in patterns and "?a" not in patterns:
                projection = "ask "
            else:
                projection = "select where "
                if "?b" in patterns:
                    projection = projection.replace("select ", "select ?b ")
                if "?a" in patterns:
                    projection = projection.replace("select ", "select ?a ")

            patterns = "{"+patterns+"}"

            sparql_query = projection + patterns

            qres = g.query(sparql_query)

            r_j = self.consequents[0].relation
            if isinstance(r_j, URIRelation):
                new_p = self.consequents[0].relation.uri
            else:
                new_p = URIRelation(self.consequents[0].relation).uri

            if "?a" in projection and "?b" in projection:
                for a, b in qres:
                    new_s = a
                    new_o = b
                    ss.append(new_s), ps.append(new_p), os.append(new_o)
            elif "?a" in projection:
                new_o = s if new_lit.arg1 == 2 else o
                for a in qres:
                    new_s = a[0]
                    ss.append(new_s), ps.append(new_p), os.append(new_o)
            elif "?b" in projection:
                new_s = s if new_lit.arg1 == 1 else o
                for b in qres:
                    new_o = b[0]
                    ss.append(new_s), ps.append(new_p), os.append(new_o)
            else:
                if bool(qres):
                    new_s = s if new_lit.arg1 == 1 else o
                    new_o = o if new_lit.arg2 == 2 else s
                    ss.append(new_s), ps.append(new_p), os.append(new_o)

            return zip(ss, ps, os)

    @staticmethod
    def parse_amie(line, rel_dict):
        cells = line.split("\t")
        rule_string = cells[0]
        std_conf = float(cells[2].strip())
        pca_conf = float(cells[3].strip())
        assert "=>" in rule_string
        ant_cons = rule_string.split("=>")
        ant_cons = list(filter(None, ant_cons))
        ant_string = ant_cons[0].strip()
        con_string = ant_cons[1].strip()

        ant_string = re.sub("(\?\w+)\s+\?", "\g<1>|?", ant_string)
        con_string = re.sub("(\?\w+)\s+\?", "\g<1>|?", con_string)

        antecedents = []
        for ant in ant_string.split("|"):
            lit = Literal.parse_amie(ant, rel_dict)
            if lit is None:
                return None
            antecedents.append(lit)

        consequents = []
        for con in con_string.split("|"):
            lit = Literal.parse_amie(con, rel_dict)
            if lit is None:
                return None
            consequents.append(lit)

        return Rule(antecedents, consequents, std_conf, pca_conf)
