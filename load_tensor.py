from typing import Dict, Tuple, List

from rdflib import Graph
import numpy as np
from rdflib.namespace import RDF, RDFS, OWL
from argparse import ArgumentParser, Namespace
from load_tensor_tools import get_ranges, get_domains, get_type_dag, get_prop_dag
from scipy.sparse import coo_matrix


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Load tensor from rdf data")
    parser.add_argument("input", type=str, default=None, help="path to dataset on which to perform the evaluation")
    parser.add_argument("-td", "--typedict", type=str, default=None,
                        help="path to types dictionary (tensor from which the synthesized data was loaded from)")
    return parser.parse_args()


def load_graph(input_data: str) -> Tuple[Graph, str]:
    rdf_format = input_data[input_data.rindex(".") + 1:]

    print("Loading data...")
    # g is a graph of the loaded RDF data
    graph = Graph()
    graph.parse(input_data, format=rdf_format)
    return graph, rdf_format


def extract_entity_types(graph: Graph) -> Dict[str, int]:
    # dictionary of entity types/classes
    # contains tuples of (type, type_id)
    entity_type_to_id = {}

    print("Loading entity types..")
    entity_type_id = 0
    # iterate over all triples that express a "is type of" relation
    for subject, predicate, object in graph.triples((None, RDF.type, None)):
        # if the type hasn't been seen yet, add it to the type dictionary with a unique id
        if object not in entity_type_to_id:
            entity_type_to_id[object] = entity_type_id
            entity_type_id += 1

    # iterate over all triples that define the classes
    for subject, predicate, object in graph.triples((None, RDF.type, OWL.Class)):
        # if the class hasn't been seen yet, add it to the type dictionary with a unique id
        if subject not in entity_type_to_id:
            entity_type_to_id[subject] = entity_type_id
            entity_type_id += 1

    # iterate over all triples that define subclasses
    for subject, predicate, object in graph.triples((None, RDFS.subClassOf, None)):
        # if the subclass hasn't been seen yet, add it to the type dictionary with a unique id
        if subject not in entity_type_to_id:
            entity_type_to_id[subject] = entity_type_id
            entity_type_id += 1

        # if the superclass hasn't been seen yet, add it to the type dictionary with a unique id
        if object not in entity_type_to_id:
            entity_type_to_id[object] = entity_type_id
            entity_type_id += 1
    print(f"{len(entity_type_to_id)} types loaded...")
    return entity_type_to_id


def extract_entities(graph: Graph, entity_type_to_id: Dict[str, int]) -> Dict[str, int]:
    # dictionary of all subjects that are not types (entities)
    # contains tuples of (entity, entity_id)
    entity_to_id = {}

    print("Loading subjects dict...")
    subject_id = 0
    # add all subjects that are not types to the subject dictionary with a new id
    for subject in graph.subjects():
        if subject not in entity_to_id and subject not in entity_type_to_id:
            entity_to_id[subject] = subject_id
            subject_id += 1
    print(f"{len(entity_to_id)} subjects loaded...")
    return entity_to_id


def extract_properties(graph: Graph, entity_to_id: Dict[str, int]) -> Dict[str, int]:
    # dictionary of object properties
    # contains tuples of (property, property_id)
    property_to_id = {}

    print("Loading object properties...")
    predicate_id = 0
    # iterate over all triples
    for subject, predicate, object in graph.triples((None, None, None)):
        # if the predicate hasn't been seen yet and if both subject and object are non-type entities, then add the
        # predicate to the predicate dictionary with a new id
        if predicate not in property_to_id:
            if subject in entity_to_id and object in entity_to_id:
                property_to_id[predicate] = predicate_id
                predicate_id += 1

    # iterate over all triples that define the subject as an object property
    for subject, predicate, object in graph.triples((None, OWL.ObjectProperty, None)):
        # if the object property hasn't been seen as a predicate yet, add it to the predicate dictionary with a new id
        if subject not in property_to_id:
            property_to_id[predicate] = predicate_id
            predicate_id += 1

    print(f"{len(property_to_id)} object properties loaded...")
    return property_to_id


def create_property_adjacency_matrices(graph: Graph,
                                       entity_to_id: Dict[str, int],
                                       property_to_id: Dict[str, int]) -> List[coo_matrix]:
    print("Allocating adjacency matrices...")
    data_coo = [{"rows": [], "cols": [], "vals": []}] * len(property_to_id)

    print("Populating adjacency matrices...")
    # iterate over every triple that defines a relationship (object property) between two subjects
    for subject, predicate, object in graph.triples((None, None, None)):
        if subject in entity_to_id and object in entity_to_id and predicate in property_to_id:
            # unique ids of the subject, predicate and object
            subject_id = entity_to_id[subject]
            object_id = entity_to_id[object]
            predicate_id = property_to_id[predicate]
            # triple that is used to create adjacency matrix
            data_coo[predicate_id]["rows"].append(subject_id)
            data_coo[predicate_id]["cols"].append(object_id)
            data_coo[predicate_id]["vals"].append(1)
            # data[p_i][s_i,o_i] = 1

    # create sparse matrices for every object property by aggregating (row, column, value) tuples
    # these tuples have the values: (subject_id, object_id, 1)
    return [coo_matrix((p["vals"], (p["rows"], p["cols"])), shape=(len(entity_to_id), len(entity_to_id))) for p in
            data_coo]


def create_entity_type_adjacency_matrix(graph: Graph,
                                        entity_to_id: Dict[str, int],
                                        entity_type_to_id: Dict[str, int]) -> coo_matrix:
    # create matrix for object types/classes
    type_coo = {"rows": [], "cols": [], "vals": []}
    print("Populating type matrix with type assertions...")
    type_assertions = 0
    # iterate over all triples that define a type relationship between an entity and an entity type
    for subject, predicate, object in graph.triples((None, RDF.type, None)):
        if subject in entity_to_id and object in entity_type_to_id:
            # unique ids of the subject and object
            subject_id = entity_to_id[subject]
            object_id = entity_type_to_id[object]
            # triple that is used to create adjacency matrix
            type_coo["rows"].append(subject_id)
            type_coo["cols"].append(object_id)
            type_coo["vals"].append(1)
            type_assertions += 1

    # create sparse matrix for the entity types by aggregating (row, column, value) tuples
    # these tuples have the values: (entity_id, type_id, 1)
    typedata = coo_matrix((type_coo["vals"], (type_coo["rows"], type_coo["cols"])),
                          shape=(len(entity_to_id), len(entity_type_to_id)),
                          dtype=int)
    print(f"{type_assertions} type assertions loaded...")
    return typedata


def main():
    args = cli_args()
    print(args)

    graph, rdf_format = load_graph(args.input)
    entity_type_to_id = extract_entity_types(graph)
    entity_to_id = extract_entities(graph, entity_type_to_id)
    property_to_id = extract_properties(graph, entity_to_id)

    property_adjaceny_matrices = create_property_adjacency_matrices(graph, entity_to_id, property_to_id)

    entity_type_adjacency_matrix = create_entity_type_adjacency_matrix(graph, entity_to_id, entity_type_to_id)

    # TODO: doc...
    type_hierarchy = get_type_dag(graph, entity_type_to_id)
    prop_hierarchy = get_prop_dag(graph, property_to_id)

    # change from objects to indices to avoid "maximum recursion depth exceeded" when pickling
    for i, n in type_hierarchy.items():
        n.children = [c.node_id for c in n.children]
        n.parents = [p.node_id for p in n.parents]
    for i, n in prop_hierarchy.items():
        n.children = [c.node_id for c in n.children]
        n.parents = [p.node_id for p in n.parents]

    type_total = len(entity_type_to_id) if entity_type_to_id else 0
    type_matched = len(type_hierarchy) if type_hierarchy else 0
    prop_total = len(entity_type_to_id) if entity_type_to_id else 0
    prop_matched = len(prop_hierarchy) if prop_hierarchy else 0

    print("load types hierarchy: total=%d matched=%d" % (type_total, type_matched))
    print("load relations hierarchy: total=%d matched=%d" % (prop_total, prop_matched))

    domains = get_domains(graph, property_to_id, entity_type_to_id)
    print("load relation domains: total=%d" % (len(domains)))

    ranges = get_ranges(graph, property_to_id, entity_type_to_id)
    print("load relation ranges: total=%d" % (len(ranges)))

    rdfs = {"type_hierarchy": type_hierarchy,
            "prop_hierarchy": prop_hierarchy,
            "domains": domains,
            "ranges": ranges}

    np.savez(args.input.replace("." + rdf_format, ".npz"),
             data=property_adjaceny_matrices,
             types=entity_type_adjacency_matrix,
             entities_dict=entity_to_id,
             relations_dict=property_to_id,
             types_dict=entity_type_to_id,
             type_hierarchy=type_hierarchy,
             prop_hierarchy=prop_hierarchy,
             domains=domains,
             ranges=ranges)


if __name__ == '__main__':
    main()
