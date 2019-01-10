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


def load_graph(input_file_path: str) -> Tuple[Graph, str]:
    """
    Loads the RDF graph from a tensor file into a Knowledge Graph.
    :param input_file_path: path to the .n3 file of the RDF data
    :return: tuple of the created graph object and the format of the rdf file
    """
    rdf_format = input_file_path[input_file_path.rindex(".") + 1:]

    print("Loading data...")
    graph = Graph()
    graph.parse(input_file_path, format=rdf_format)
    return graph, rdf_format


def extract_entity_types(graph: Graph) -> Dict[str, int]:
    """
    Extracts the "type of" and "class of" relations from the Knowledge Graph and creates a dictionary of the entity
    types and their unique identifiers.
    :param graph: the graph object of the Knowledge Graph
    :return: a dictionary of entity types pointing to their id
    """
    # dictionary of entity types/classes
    # contains tuples of (entity_type, entity_type_id)
    entity_type_to_id = {}

    print("Loading entity types..")
    entity_type_id = 0
    # iterate over all triples that express a "is type of" relation
    for subject, predicate, object in graph.triples((None, RDF.type, None)):
        # if the type hasn't been seen yet, add it to the type dictionary with a unique id
        if object not in entity_type_to_id:
            entity_type_to_id[object] = entity_type_id
            entity_type_id += 1

    print("Loading classes...")
    # iterate over all triples that define the classes
    for subject, predicate, object in graph.triples((None, RDF.type, OWL.Class)):
        # if the class hasn't been seen yet, add it to the type dictionary with a unique id
        if subject not in entity_type_to_id:
            entity_type_to_id[subject] = entity_type_id
            entity_type_id += 1

    print("Loading subclasses...")
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
    """
    Extracts the entities from the Knowledge Graph and creates a dictionary of the entities and their unique
    identifiers.
    :param graph: the graph object of the Knowledge Graph
    :param entity_type_to_id: the dictionary of the entity types and their unique identifies
    :return: a dictionary of entity types pointing to their id
    """
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
    """
    Extracts the object properties/entity properties from the Knowledge graph and creates a dictionary of the properties
    and their unique identifiers.
    :param graph: the graph object of the Knowledge Graph
    :param entity_to_id: the dictionary of the entities and their unique identifies
    :return: a dictionary of object properties/entity properties pointing to their id
    """
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

    print("Iterating over OWL.ObjectProperty...")
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
    """
    Creates an adjacency matrix for every object property.
    :param graph: the graph object of the Knowledge Graph
    :param entity_to_id: the dictionary of the entities and their unique identifies
    :param property_to_id: the dictionary of the object properties and their unique identifies
    :return: a list of adjacency matrices, where the adjacency matrix at list index i belongs to the object property
    with the unique identifier i
    """
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
    """
    Creates an adjacency matrix for the "type of" relation, i.e., it contains all the type information of the graph.
    :param graph: the graph object of the Knowledge Graph
    :param entity_to_id: the dictionary of the entities and their unique identifies
    :param entity_type_to_id: the dictionary of the entity types and their unique identifies
    :return: an adjacency matrix for the "type of" relation
    """
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

    # load the graph and extract entities, entity types and object properties
    graph, rdf_format = load_graph(args.input)
    entity_type_to_id = extract_entity_types(graph)
    np.save(args.input.replace("." + rdf_format, "_types_dict.npz"), entity_type_to_id)

    entity_to_id = extract_entities(graph, entity_type_to_id)
    np.save(args.input.replace("." + rdf_format, "_entities_dict.npz"), entity_to_id)

    property_to_id = extract_properties(graph, entity_to_id)
    np.save(args.input.replace("." + rdf_format, "_relations_dict.npz"), property_to_id)

    # build adjacency matrices for all relations (object properties and type relations) in the graph
    property_adjaceny_matrices = create_property_adjacency_matrices(graph, entity_to_id, property_to_id)
    np.save(args.input.replace("." + rdf_format, "_data.npz"), property_adjaceny_matrices)

    entity_type_adjacency_matrix = create_entity_type_adjacency_matrix(graph, entity_to_id, entity_type_to_id)
    np.save(args.input.replace("." + rdf_format, "_types.npz"), entity_type_adjacency_matrix)

    # DAG of the entity type/class hierarchy
    entity_type_hierarchy_dag = get_type_dag(graph, entity_type_to_id)
    # DAG of the object property hierarchy
    object_property_hierarchy_dag = get_prop_dag(graph, property_to_id)

    print("Replacing DAG nodes with ids in entity type DAG")
    # Replace the DAG nodes with the ids of the entity types they represent to avoid recursion errors when pickling the
    # graph
    for entity_type_id, entity_type_dag_node in entity_type_hierarchy_dag.items():
        entity_type_dag_node.children = [child.node_id for child in entity_type_dag_node.children]
        entity_type_dag_node.parents = [parent.node_id for parent in entity_type_dag_node.parents]

    np.save(args.input.replace("." + rdf_format, "_type_hierarchy.npz"), entity_type_hierarchy_dag)

    print("Replacing DAG nodes with ids in property type DAG")
    # Replace the DAG nodes with the ids of the properties they represent to avoid recursion errors when pickling the
    # graph
    for property_id, property_dag_node in object_property_hierarchy_dag.items():
        property_dag_node.children = [child.node_id for child in property_dag_node.children]
        property_dag_node.parents = [parent.node_id for parent in property_dag_node.parents]

    np.save(args.input.replace("." + rdf_format, "_prop_hierarchy.npz"), object_property_hierarchy_dag)

    num_entity_types = len(entity_type_to_id or {})
    num_entity_types_in_hierarchy = len(entity_type_hierarchy_dag or {})
    num_object_properties = len(entity_type_to_id or {})
    num_object_properties_in_hierarchy = len(object_property_hierarchy_dag or {})

    print(f"Loaded {num_entity_types} entity types, of which {num_entity_types_in_hierarchy} are contained in the "
          f"hierarchy graph...")
    print(f"Loaded {num_object_properties} object properties, of which {num_object_properties_in_hierarchy} are "
          f"contained in the hierarchy graph...")

    # explanation of domains and ranges: https://stackoverflow.com/a/9066520
    domains = get_domains(graph, property_to_id, entity_type_to_id)
    print(f"Loaded {len(domains)} relation domains...")
    np.save(args.input.replace("." + rdf_format, "_domains.npz"), domains)

    ranges = get_ranges(graph, property_to_id, entity_type_to_id)
    print(f"Loaded {len(ranges)} relation ranges...")
    np.save(args.input.replace("." + rdf_format, "_ranges.npz"), ranges)

    # TODO: don't know what this is for
    rdfs = {"type_hierarchy": entity_type_hierarchy_dag,
            "prop_hierarchy": object_property_hierarchy_dag,
            "domains": domains,
            "ranges": ranges}

    # serialize graph as .npz file
    # np.savez(args.input.replace("." + rdf_format, ".npz"),
    #          data=property_adjaceny_matrices,  # WARNING: this could be a problem with a huge dataset (e.g., DBpedia)
    #          types=entity_type_adjacency_matrix,
    #          entities_dict=entity_to_id,
    #          relations_dict=property_to_id,
    #          types_dict=entity_type_to_id,
    #          type_hierarchy=entity_type_hierarchy_dag,
    #          prop_hierarchy=object_property_hierarchy_dag,
    #          domains=domains,
    #          ranges=ranges)


if __name__ == '__main__':
    main()
