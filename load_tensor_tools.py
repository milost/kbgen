import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
from rdflib import OWL, RDFS, Graph
from scipy.sparse import coo_matrix, load_npz, save_npz

from tensor_models import DAGNode


def filename(input_dir: str, file: str) -> str:
    file_suffixes = {
        "entity_type_dict": ["types_dict", "npy"],
        "entity_dict": ["entities_dict", "npy"],
        "relations_dict": ["relations_dict", "npy"],
        "type_matrix": ["entity_type_matrix", "npz"],
        "type_hierachy": ["type_hierarchy", "npy"],
        "property_hierarchy": ["prop_hierarchy", "npy"],
        "domains": ["domains", "npy"],
        "ranges": ["ranges", "npy"]
    }
    suffix, filetype = file_suffixes[file]
    return f"{input_dir}/{suffix}.{filetype}"


def to_triples(X, order="pso"):
    h, t, r = [], [], []
    for i in range(len(X)):
        r.extend(np.full(X[i].nnz, i))
        h.extend(X[i].row.tolist())
        t.extend(X[i].col.tolist())
    if order == "spo":
        triples = zip(h, r, t)
    if order == "pso":
        triples = zip(r, h, t)
    if order == "sop":
        triples = zip(h, t, r)
    return np.array(triples)


def load_domains(input_dir: str) -> Dict[int, int]:
    file = filename(input_dir, "domains")
    print(f"Loading domains from {file}")
    domains: Dict[int, int] = np.load(file).item()
    return domains


def save_domains(input_dir: str, domains: Dict[int, int]):
    file = filename(input_dir, "domains")
    np.save(file, domains)
    print(f"Saved domains to {file}")


def load_ranges(input_dir: str) -> Dict[int, int]:
    file = filename(input_dir, "ranges")
    print(f"Loading ranges from {file}")
    ranges: Dict[int, int] = np.load(file).item()
    return ranges


def save_ranges(input_dir: str, ranges: Dict[int, int]):
    file = filename(input_dir, "ranges")
    np.save(file, ranges)
    print(f"Saved ranges to {file}")


def load_type_hierarchy(input_dir: str) -> Dict[int, DAGNode]:
    file = filename(input_dir, "type_hierachy")
    print(f"Loading entity type hierarchy DAG from {file}")
    hierarchy: Dict[int, DAGNode] = np.load(file).item()
    # Reverse the serialization by replacing the node ids with the actual nodes
    for entity_type_id, entity_type_dag_node in hierarchy.items():
        try:
            entity_type_dag_node.children = [hierarchy[child] for child in entity_type_dag_node.children]
            entity_type_dag_node.parents = [hierarchy[parent] for parent in entity_type_dag_node.parents]
        except:
            pass
    return hierarchy


def save_type_hierarchy(input_dir: str, entity_type_hierarchy_dag: Dict[int, DAGNode]):
    file = filename(input_dir, "type_hierachy")
    print("Replacing DAG nodes with ids in entity type DAG")
    # Replace the DAG nodes with the ids of the entity types they represent to avoid recursion errors when pickling
    # the graph
    for entity_type_id, entity_type_dag_node in entity_type_hierarchy_dag.items():
        entity_type_dag_node.children = [child.node_id for child in entity_type_dag_node.children]
        entity_type_dag_node.parents = [parent.node_id for parent in entity_type_dag_node.parents]

    np.save(file, entity_type_hierarchy_dag)
    print(f"Saved entity type hierarchy DAG to {file}")


def load_prop_hierarchy(input_dir: str) -> Dict[int, DAGNode]:
    file = filename(input_dir, "property_hierarchy")
    print(f"Loading property hierarchy DAG from {file}")
    hierarchy: Dict[int, DAGNode] = np.load(file).item()
    # Reverse the serialization by replacing the node ids with the actual nodes
    for property_id, property_dag_node in hierarchy.items():
        try:
            property_dag_node.children = [hierarchy[children] for children in property_dag_node.children]
            property_dag_node.parents = [hierarchy[parent] for parent in property_dag_node.parents]
        except:
            pass
    return hierarchy


def save_prop_hierarchy(input_dir: str, property_hierarchy_dag: Dict[int, DAGNode]):
    file = filename(input_dir, "property_hierarchy")
    print("Replacing DAG nodes with ids in property type DAG")
    # Replace the DAG nodes with the ids of the properties they represent to avoid recursion errors when pickling
    # the graph
    for property_id, property_dag_node in property_hierarchy_dag.items():
        property_dag_node.children = [child.node_id for child in property_dag_node.children]
        property_dag_node.parents = [parent.node_id for parent in property_dag_node.parents]

    np.save(file, property_hierarchy_dag)
    print(f"Saved property hierarchy DAG to {file}")


def load_entities_dict(input_dir: str) -> Dict[str, int]:
    file = filename(input_dir, "entity_dict")
    print(f"Loading entity type dict from {file}")
    entity_to_id: Dict[str, int] = np.load(file).item()
    return entity_to_id


def save_entities_dict(input_dir: str, entity_to_id: Dict[str, int]):
    file = filename(input_dir, "entity_dict")
    np.save(file, entity_to_id)
    print(f"Saved entity dict to {file}")


def load_types_dict(input_dir: str) -> Dict[str, int]:
    file = filename(input_dir, "entity_type_dict")
    print(f"Loading entity type dict from {file}")
    entity_type_to_id: Dict[str, int] = np.load(file).item()
    return entity_type_to_id


def save_types_dict(input_dir: str, entity_type_to_id: Dict[str, int]):
    file = filename(input_dir, "entity_type_dict")
    np.save(file, entity_type_to_id)
    print(f"Saved entity type dict to {file}")


def load_relations_dict(input_dir: str) -> Dict[str, int]:
    file = filename(input_dir, "relations_dict")
    print(f"Loading property type dict from {file}")
    property_to_id: Dict[str, int] = np.load(file).item()
    return property_to_id


def save_relations_dict(input_dir: str, property_to_id: Dict[str, int]):
    file = filename(input_dir, "relations_dict")
    np.save(file, property_to_id)
    print(f"Saved property type dict to {file}")


def load_graph_npz(input_dir: str) -> List[coo_matrix]:
    directory = f"{input_dir}/adjacency_matrices"
    print(f"Loading property adjacency matrices from {directory}")
    loaded_matrices: List[coo_matrix] = []
    index = 0
    while True:
        file_name = f"{directory}/{index}.npz"
        try:
            matrix: coo_matrix = load_npz(file_name)
            loaded_matrices.append(matrix)
            index += 1
            print(f"Loaded {index} property adjacency matrices.", end="\r")
        except FileNotFoundError:
            break
    print(f"Loaded {index} property adjacency matrices.")
    return loaded_matrices


def save_graph_npz(input_dir: str, property_adjaceny_matrices: List[coo_matrix]):
    directory = f"{input_dir}/adjacency_matrices"
    Path(input_dir).mkdir(exist_ok=True)
    print(f"Saving {len(property_adjaceny_matrices)} property adjacency matrices to {directory}")
    for index, matrix in enumerate(property_adjaceny_matrices):
        file_name = f"{directory}/{index}.npz"
        save_npz(file_name, matrix)
    print(f"Saved property adjacency matrices.")


def load_types_npz(input_dir: str) -> coo_matrix:
    file = filename(input_dir, "type_matrix")
    print(f"Loading entity type adjacency matrix from {file}")
    entity_type_adjacency_matrix: coo_matrix = load_npz(file)
    return entity_type_adjacency_matrix


def save_types_npz(input_dir: str, entity_type_adjacency_matrix: coo_matrix):
    file = filename(input_dir, "type_matrix")
    save_npz(input_dir, entity_type_adjacency_matrix)
    print(f"Saved entity type adjacency matrix to {file}")


# def coo_matrix_to_dict(matrix: coo_matrix):
#     data = matrix.data
#     rows: np.ndarray = matrix.row
#     columns: np.ndarray = matrix.col
#     shape = matrix.shape
#     return {"data": data, "rows": rows, "columns": columns, "shape": shape}
#
#
# def dict_to_coo_matrix(data: dict):
#     return coo_matrix((data["data"], (data["rows"], data["columns"])), shape=data["shape"])
#
#
# def save_coo_matrix(input_dir: str, matrix: coo_matrix):
#     data = coo_matrix_to_dict(matrix)
#     np.savez(input_dir, data=data["data"], rows=data["rows"], columns=data["columns"], shape=data["shape"])
#
#
# def load_coo_matrix(input_dir: str) -> coo_matrix:
#     loaded = np.load(input_dir)
#     return dict_to_coo_matrix(loaded)
#
#
# def save_adjacency_matrices(input_dir: str, data: List[coo_matrix]):
#     serialized = [coo_matrix_to_dict(matrix) for matrix in data]
#     np.savez(input_dir, serialized)
#
#
# def load_adjacency_matrices(input_dir: str) -> List[coo_matrix]:
#     loaded = np.load(input_dir)
#     data = loaded.tolist()
#     return [dict_to_coo_matrix(element) for element in data]


def save_graph_binary(input_dir: str, graph: Graph):
    file = f"{input_dir}/graph.bin"
    print(f"Saving graph to {file}")
    with open(file, "wb") as graph_file:
        pickle.dump(graph, graph_file, protocol=pickle.HIGHEST_PROTOCOL)
    print("Saved graph.")


def load_graph_binary(input_dir: str):
    file = f"{input_dir}/graph.bin"
    print(f"Loading graph from {file}")
    with open(file, "rb") as graph_file:
        graph = pickle.load(graph_file)
    print("Loaded graph.")
    rdf_format = "ttl"
    return graph, rdf_format


def get_prop_dag(graph: Graph, property_to_id: Dict[str, int]) -> Dict[int, DAGNode]:
    """
    Creates a DAG of the object property hierarchy in the graph. This hierarchy is defined by the RDFS
    "sub-property of" property.
    :param graph: the graph object of the Knowledge Graph
    :param property_to_id: the dictionary of the object properties and their unique identifies
    :return: dictionary of object property ids and the corresponding nodes in the DAG
    """
    # dictionary pointing from object property id to the corresponding node in the entity type DAG
    property_dag = {}

    # iterate over property hierarchy
    for subject, predicate, object in graph.triples((None, RDFS.subPropertyOf, None)):
        # if both subject and object are properties
        if (subject in property_to_id) and (object in property_to_id):
            subject_id = property_to_id[subject]
            object_id = property_to_id[object]
            # add subject and object to the DAG if subject and object are different properties
            if subject_id != object_id:
                if object_id not in property_dag:
                    property_dag[object_id] = DAGNode(object_id, object, parents=[], children=[])
                if subject_id not in property_dag:
                    property_dag[subject_id] = DAGNode(subject_id, subject, parents=[], children=[])

                # add DAG node of object as parent to the subject DAG node
                property_dag[subject_id].parents.append(property_dag[object_id])
                # add DAG node of the subject as child to the object DAG node
                property_dag[object_id].children.append(property_dag[subject_id])

    return property_dag


def get_type_dag(graph: Graph, entity_type_to_id: Dict[str, int]) -> Dict[int, DAGNode]:
    """
    Creates a DAG of the entity type hierarchy in the graph. This hierarchy is defined by the RDFS "subclass of"
    property.
    :param graph: the graph object of the Knowledge Graph
    :param entity_type_to_id: the dictionary of the entity types and their unique identifies
    :return: dictionary of entity type ids and the corresponding nodes in the DAG
    """
    # dictionary pointing from entity type id to the corresponding node in the entity type DAG
    entity_type_dag = {}

    # extract equivalence class relation
    equivalent_classes = {}
    for subject, predicate, object in graph.triples((None, OWL.equivalentClass, None)):
        equivalent_classes[subject] = object
        equivalent_classes[object] = subject

    # iterate over class hierarchy
    for subject, predicate, object in graph.triples((None, RDFS.subClassOf, None)):

        # is the subject is an entity type or equivalent to an entity type
        subject_is_entity_type = (subject in entity_type_to_id or
                                  (subject in equivalent_classes and equivalent_classes[subject] in entity_type_to_id))
        # is the object is an entity type or equivalent to an entity type
        object_is_entity_type = (object in entity_type_to_id or
                                 (object in equivalent_classes and equivalent_classes[object] in entity_type_to_id))

        # if the subject is an entity type or equivalent to an entity type AND the object is an entity type or
        # equivalent to an entity type
        if subject_is_entity_type and object_is_entity_type:
            # replace subject and object with their equivalent entity type if thhey are not an entity type themselves
            if subject not in entity_type_to_id:
                subject = equivalent_classes[subject]
            if object not in entity_type_to_id:
                object = equivalent_classes[object]

            subject_id = entity_type_to_id[subject]
            object_id = entity_type_to_id[object]
            # add subject and object and their relation to the DAG
            if subject_id != object_id:
                if object_id not in entity_type_dag:
                    entity_type_dag[object_id] = DAGNode(object_id, object)
                if subject_id not in entity_type_dag:
                    entity_type_dag[subject_id] = DAGNode(subject_id, subject)

                # add DAG node of object as parent to the subject DAG node
                entity_type_dag[subject_id].parents.append(entity_type_dag[object_id])
                # add DAG node of the subject as child to the object DAG node
                entity_type_dag[object_id].children.append(entity_type_dag[subject_id])

    return entity_type_dag


def get_domains(graph: Graph, property_to_id: Dict[str, int], entity_type_to_id: Dict[str, int]) -> Dict[int, int]:
    """
    Extracts the domains from the Knowledge Graph. These are defined by the RDFS "domain" relation. These relations are
    defined between an object property (subject) and an entity type (object) and express that the object property
    describes entities with that entity type. (for further reading: https://stackoverflow.com/a/9066520)
    :param graph: the graph object of the Knowledge Graph
    :param property_to_id: the dictionary of the object properties and their unique identifies
    :param entity_type_to_id: the dictionary of the entity types and their unique identifies
    :return: a dictionary pointing from property ids to the entity type id of that properties' domain
    """
    # dictionary pointing from object property id to an entity type id
    domains = {}

    # add all domain triples for which the subject is an object property and the object is an entity type
    for subject, predicate, object in graph.triples((None, RDFS.domain, None)):
        if subject in property_to_id and object in entity_type_to_id:
            domains[property_to_id[subject]] = entity_type_to_id[object]

    return domains


def get_ranges(graph: Graph, property_to_id: Dict[str, int], entity_type_to_id: Dict[str, int]) -> Dict[int, int]:
    """
    Extracts the ranges from the Knowledge Graph. These are defined by the RDFS "range" relation. These relations
    are defined between an object property (subject) and an entity type (object) and express that the value of the
    object property is of the entity type. (for further reading: https://stackoverflow.com/a/9066520)
    :param graph: the graph object of the Knowledge Graph
    :param property_to_id: the dictionary of the object properties and their unique identifies
    :param entity_type_to_id: the dictionary of the entity types and their unique identifies
    :return: a dictionary pointing from property ids to the entity type id of that properties' range
    """
    # dictionary pointing from object property id to an entity type id
    ranges = {}

    # add all range triples for which the subject is an object property and the object is an entity type
    for subject, predicate, object in graph.triples((None, RDFS.range, None)):
        if subject in property_to_id and object in entity_type_to_id:
            ranges[property_to_id[subject]] = entity_type_to_id[object]
    return ranges


################################################################################
# unused methods
################################################################################
#
# def get_prop_tree(g, dict_rel):
#     prop_tree = {}
#     for s, p, o in g.triples((None, RDFS.subPropertyOf, None)):
#         if (s in dict_rel) and (o in dict_rel):
#             s_id = dict_rel[s]
#             o_id = dict_rel[o]
#             if s_id != o_id:
#                 if o_id not in prop_tree:
#                     prop_tree[o_id] = TreeNode(o_id, o, children=[])
#                 if s_id not in prop_tree:
#                     prop_tree[s_id] = TreeNode(s_id, s, prop_tree[o_id], children=[])
#
#                 if prop_tree[s_id].parent is None:
#                     prop_tree[s_id].parent = prop_tree[o_id]
#                 if prop_tree[s_id].parent == prop_tree[o_id]:
#                     prop_tree[o_id].children.append(prop_tree[s_id])
#
#     return prop_tree
#
#
# def load_type_dict(input_path):
#     dataset = np.load(input_path)
#     dict_type = dataset["types_dict"]
#     if not isinstance(dict_type, dict):
#         dict_type = dict_type.item()
#     return dict_type
#
#
# def get_type_tree(g, dict_type):
#     type_tree = {}
#
#     # getting equivalent classes
#     equi_classes = {}
#     for s, p, o in g.triples((None, OWL.equivalentClass, None)):
#         equi_classes[s] = o
#         equi_classes[o] = s
#
#     for s, p, o in g.triples((None, RDFS.subClassOf, None)):
#         if (s in dict_type or (s in equi_classes and equi_classes[s] in dict_type)) and \
#                 (o in dict_type or (o in equi_classes and equi_classes[o] in dict_type)):
#
#             if s not in dict_type:
#                 s = equi_classes[s]
#             if o not in dict_type:
#                 o = equi_classes[o]
#
#             s_id = dict_type[s]
#             o_id = dict_type[o]
#             if s_id != o_id:
#                 if o_id not in type_tree:
#                     type_tree[o_id] = TreeNode(o_id, o, children=[])
#                 if s_id not in type_tree:
#                     type_tree[s_id] = TreeNode(s_id, s, type_tree[o_id], children=[])
#
#                 if type_tree[s_id].parent is None:
#                     type_tree[s_id].parent = type_tree[o_id]
#                 if type_tree[s_id].parent == type_tree[o_id]:
#                     type_tree[o_id].children.append(type_tree[s_id])
#
#     return type_tree
#
#
# def dag_to_tree(dag):
#     tree = {}
#     for i, n in dag.items():
#         tree[i] = TreeNode(n.node_id, n.name)
#     for i, n in dag.items():
#         tree[i].children = [tree[c.node_id] for c in n.children]
#         tree[i].parent = None if not n.parents else tree[min(n.parents).node_id]
#     for i, n in tree.items():
#         for c in n.children:
#             if c.parent != n:
#                 n.children.remove(c)
#     return tree
#
#
# def get_roots(hier):
#     if not hier:
#         return []
#     else:
#         roots = []
#         for i, n in hier.items():
#             if isinstance(n, DAGNode):
#                 if not n.parents:
#                     roots.append(n)
#             if isinstance(n, TreeNode):
#                 if n.parent is None:
#                     roots.append(n)
#         return roots
