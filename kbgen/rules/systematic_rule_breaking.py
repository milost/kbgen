import operator
from datetime import date
from random import random
from typing import Callable, Tuple

from rdflib import Graph, URIRef, Literal
from tqdm import tqdm

from .realworld_rule import RealWorldRule


def break_by_birth_date(rule: RealWorldRule,
                        graph: Graph,
                        birth_date_relation: URIRef,
                        comparison_date: date,
                        comparison: Callable[[date, date], bool] = operator.lt,
                        break_chance: float = 1.0) -> Tuple[Graph, list]:
    """
    Break a rule by the birth date of the (conclusion) subject. A fact is removed if
    comparison(subject_date, comparison_date) returns True. This removal can be further influences by the break_chance
    parameter (e.g., only break 70% of the tuples for which the comparison returns true).

    Call to remove 80% of facts whose subject was born before 1950:
    break_by_birth_date(rule, graph, relation_uri, datetime.date(year=1950, month=1, day=1), operator.lt, 0.8)

    :param rule: the rule that is broken systematically
    :param graph: the graph object in which the rule is broken
    :param birth_date_relation: the URI of the birth date relation
    :param comparison_date: the date used for the comparison.
    :param comparison: comparison function that takes two dates and returns a boolean. The first date is the birth date
                       of the subject and the second date is the comparison_date parameter.
    :param break_chance: how many of the facts that would be removed are actually removed (e.g., 0.7 => 70% of facts
                         that would be removed by the comparison are removed)
    :return: the graph in which the rule was broken systematically
    """
    print(f"Breaking {rule} by birth date")
    positive_facts = list(graph.query(rule.full_query_pattern(include_conclusion=True)))
    positive_fact_set = set(positive_facts)
    all_facts = list(graph.query(rule.full_query_pattern()))

    # count how many facts are true (in the ground truth) and how many will be broken
    correctness_ratio = len(positive_fact_set) / len(all_facts)
    brokenness_ratio = 0

    # create oracle data (which fact is true and which is false)
    facts_to_correctness = {}
    for subject, object in all_facts:
        facts_to_correctness[rule.produce_fact(subject, object)] = (subject, object) in positive_fact_set

    # break facts by the birth date of the subject
    for subject, object in tqdm(positive_facts):
        birth_date_query = list(graph.triples((subject, birth_date_relation, None)))

        # skip this fact if the subject does not have a birth date
        if not birth_date_query:
            continue

        birth_date_literal: Literal = birth_date_query[0][2]
        number = random()
        if comparison(birth_date_literal.value, comparison_date) and number < break_chance:
            fact = rule.produce_fact(subject, object)
            graph.remove(fact)
            brokenness_ratio += 1

    brokenness_ratio /= len(all_facts)
    oracle_data = [facts_to_correctness, correctness_ratio, brokenness_ratio]
    return graph, oracle_data
