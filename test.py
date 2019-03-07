import datetime
from typing import List, Tuple, Dict

from rdflib import Graph, URIRef, Literal
from tqdm import tqdm

from kbgen.rules import RealWorldRule


def evaluate_rule(graph: Graph, rule: RealWorldRule):
    query = rule.full_query_pattern()
    result = list(graph.query(query))
    total_count = len(result)
    invalid_count = 0
    for subject, object in tqdm(result):
        if subject == object:
            continue
        fact = (subject, rule.conclusion.relation, object)
        invalid_count += fact not in graph
    return 1.0 - (invalid_count / total_count)


graph: Graph = None


def query(subject, predicate, object):
    return graph.triples((subject, predicate, object))


non_reflexive_spouses: List[Tuple[URIRef, URIRef, URIRef]] = None
birth_date_uri: URIRef = URIRef('http://dbpedia.org/property/birthDate')


def get_person_to_date() -> Dict[URIRef, datetime.date]:
    person_to_date = {}
    all_person = set()
    for subject, spouse_relation, object in non_reflexive_spouses:
        for person in [subject, object]:
            all_person.add(person)
            for query_result in query(subject, birth_date_uri, None):
                literal = query_result[2]
                if isinstance(literal, Literal):
                    if isinstance(literal.value, int):
                        person_to_date[person] = datetime.date(year=literal.value, month=1, day=1)
                    elif isinstance(literal.value, datetime.date):
                        person_to_date[person] = literal.value
    skipped = len(all_person.difference(person_to_date.keys()))
    print(f"Skipped {skipped} entities since there is no {birth_date_uri} with a literal value.")
    return person_to_date


def filter_by_date(person_to_date: Dict[URIRef, datetime.date], threshold: datetime.date) -> tuple:
    earlier = {}
    later = {}
    for person, birth_date in person_to_date.items():
        if birth_date < threshold:
            earlier[person] = birth_date
        else:
            later[person] = birth_date
    return earlier, later


def year_distribution(person_to_date: Dict[URIRef, datetime.date]) -> Dict[int, int]:
    birth_year_distribution: Dict[int, int] = {}
    for person, birth_date in person_to_date.items():
        birth_year = birth_date.year
        if birth_year not in birth_year_distribution:
            birth_year_distribution[birth_year] = 0
        birth_year_distribution[birth_year] += 1
    return birth_year_distribution


def read_yago(file_name: str) -> Graph:
    print(f"Reading graph from {file_name}")
    graph = Graph()
    dirty_chars = "<>"
    with open(file_name) as yago_file:
        for line in tqdm(yago_file, total=948358):
            triple = line.strip().split("\t")
            cleaned_triple = []
            for element in triple:
                cleaned = "".join([char for char in element if char not in dirty_chars])
                cleaned = cleaned.replace('"', "''")
                cleaned = cleaned.replace("`", "'")
                cleaned = cleaned.replace("\\", "U+005C")
                cleaned = cleaned.replace("^", "U+005E")
                cleaned_triple.append(cleaned)
            graph.add(tuple([URIRef(element) for element in cleaned_triple]))
    print()
    print(f"Created graph with {len(graph)} triples")
    return graph



