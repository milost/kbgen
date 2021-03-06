from pathlib import Path
from typing import Dict, Tuple, List

from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL
from argparse import ArgumentParser, Namespace

from tqdm import tqdm

from kbgen import load_tensor_tools as ltt
from scipy.sparse import coo_matrix

from kbgen.kb_models.multiprocessing.tensor_loader import TensorLoader


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Load tensor from rdf data")
    parser.add_argument("input", type=str, default=None, help="path to dataset on which to perform the evaluation")
    parser.add_argument("-p", "--num-processes", type=int, default=0, help="the number of processes to use")
    return parser.parse_args()


def load_graph(input_file_path: str) -> Graph:
    """
    Loads the RDF graph from a tensor file into a Knowledge Graph.
    :param input_file_path: path to the .n3 file of the RDF data
    :return: the created graph object
    """
    rdf_format = input_file_path[input_file_path.rindex(".") + 1:]

    print("Loading data...")
    graph = Graph()
    graph.parse(input_file_path, format=rdf_format)
    return graph


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
    :return: a dictionary of entities pointing to their id
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
    data_coo = [{"rows": [], "cols": [], "vals": []} for _ in range(len(property_to_id))]

    print("Populating adjacency matrices...")
    # iterate over every triple that defines a relationship (object property) between two subjects
    for subject, predicate, object in tqdm(graph):
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

    print("Splitting adjacency into separate matrices...")
    # create sparse matrices for every object property by aggregating (row, column, value) tuples
    # these tuples have the values: (subject_id, object_id, 1)
    return [coo_matrix((p["vals"], (p["rows"], p["cols"])), shape=(len(entity_to_id), len(entity_to_id))) for p in
            tqdm(data_coo)]


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

    if args.num_processes:
        loader = TensorLoader(args.input, args.num_processes)
        loader.load_tensor()
        return

    rdf_format = args.input[args.input.rindex(".") + 1:]
    input_dir = args.input.replace(f".{rdf_format}", "")
    Path(input_dir).mkdir(exist_ok=True)

    # load the graph and extract entities, entity types and object properties
    try:
        graph = ltt.load_graph_binary(input_dir)
    except FileNotFoundError:
        print("File does not exist. Loading graph from input data...")
        graph = load_graph(args.input)
        ltt.save_graph_binary(input_dir, graph)

    try:
        entity_type_to_id = ltt.load_types_dict(input_dir)
    except FileNotFoundError:
        entity_type_to_id = extract_entity_types(graph)
        ltt.save_types_dict(input_dir, entity_type_to_id)

    try:
        entity_to_id = ltt.load_entities_dict(input_dir)
    except FileNotFoundError:
        entity_to_id = extract_entities(graph, entity_type_to_id)
        ltt.save_entities_dict(input_dir, entity_to_id)

    try:
        property_to_id = ltt.load_relations_dict(input_dir)
    except FileNotFoundError:
        property_to_id = extract_properties(graph, entity_to_id)
        ltt.save_relations_dict(input_dir, property_to_id)

    if Path(ltt.graph_npz_dir(input_dir)).exists():
        property_adjaceny_matrices = ltt.load_graph_npz(input_dir)
    else:
        # build adjacency matrices for all relations (object properties and type relations) in the graph
        property_adjaceny_matrices = create_property_adjacency_matrices(graph, entity_to_id, property_to_id)
        ltt.save_graph_npz(input_dir, property_adjaceny_matrices)

    try:
        entity_type_adjacency_matrix = ltt.load_types_npz(input_dir)
    except FileNotFoundError:
        entity_type_adjacency_matrix = create_entity_type_adjacency_matrix(graph, entity_to_id, entity_type_to_id)
        ltt.save_types_npz(input_dir, entity_type_adjacency_matrix)

    # DAG of the entity type/class hierarchy
    try:
        entity_type_hierarchy_dag = ltt.load_type_hierarchy(input_dir)
    except FileNotFoundError:
        entity_type_hierarchy_dag = ltt.get_type_dag(graph, entity_type_to_id)
        ltt.save_type_hierarchy(input_dir, entity_type_hierarchy_dag)

    # DAG of the object property hierarchy
    try:
        object_property_hierarchy_dag = ltt.load_prop_hierarchy(input_dir)
    except FileNotFoundError:
        object_property_hierarchy_dag = ltt.get_prop_dag(graph, property_to_id)
        ltt.save_prop_hierarchy(input_dir, object_property_hierarchy_dag)

    num_entity_types = len(entity_type_to_id or {})
    num_entity_types_in_hierarchy = len(entity_type_hierarchy_dag or {})
    num_object_properties = len(entity_type_to_id or {})
    num_object_properties_in_hierarchy = len(object_property_hierarchy_dag or {})

    print(f"Loaded {num_entity_types} entity types, of which {num_entity_types_in_hierarchy} are contained in the "
          f"hierarchy graph...")
    print(f"Loaded {num_object_properties} object properties, of which {num_object_properties_in_hierarchy} are "
          f"contained in the hierarchy graph...")

    # explanation of domains and ranges: https://stackoverflow.com/a/9066520
    try:
        domains = ltt.load_domains(input_dir)
    except FileNotFoundError:
        domains = ltt.get_domains(graph, property_to_id, entity_type_to_id)
        print(f"Loaded {len(domains)} relation domains...")
        ltt.save_domains(input_dir, domains)

    try:
        ranges = ltt.load_ranges(input_dir)
    except FileNotFoundError:
        ranges = ltt.get_ranges(graph, property_to_id, entity_type_to_id)
        print(f"Loaded {len(ranges)} relation ranges...")
        ltt.save_ranges(input_dir, ranges)

    # # TODO: don't know what this is for
    # rdfs = {"type_hierarchy": entity_type_hierarchy_dag,
    #         "prop_hierarchy": object_property_hierarchy_dag,
    #         "domains": domains,
    #         "ranges": ranges}


if __name__ == '__main__':
    main()
