from typing import Dict

import numpy as np
from rdflib import OWL, RDFS, Graph

from tensor_models import DAGNode


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


def load_domains(input_dir):
    dataset = np.load(input_dir)
    return dataset["domains"].item() if "domains" in dataset else None


def load_ranges(input_dir):
    dataset = np.load(input_dir)
    return dataset["ranges"].item() if "ranges" in dataset else None


def load_type_hierarchy(input_dir):
    dataset = np.load(input_dir)
    hierarchy = dataset["type_hierarchy"].item() if "type_hierarchy" in dataset else None
    for i, n in hierarchy.items():
        try:
            n.children = [hierarchy[c] for c in n.children]
            n.parents = [hierarchy[p] for p in n.parents]
        except:
            pass
    return hierarchy


def load_prop_hierarchy(input_dir):
    dataset = np.load(input_dir)
    hierarchy = dataset["prop_hierarchy"].item() if "prop_hierarchy" in dataset else None
    for i, n in hierarchy.items():
        try:
            n.children = [hierarchy[c] for c in n.children]
            n.parents = [hierarchy[p] for p in n.parents]
        except:
            pass
    return hierarchy


def load_entities_dict(input_dir):
    dataset = np.load(input_dir)
    return dataset["entities_dict"].item() if "entities_dict" in dataset else None


def load_types_dict(input_dir):
    dataset = np.load(input_dir)
    return dataset["types_dict"].item() if "types_dict" in dataset else None


def load_relations_dict(input_dir):
    dataset = np.load(input_dir)
    return dataset["relations_dict"].item() if "relations_dict" in dataset else None


def loadGraphNpz(input_dir):
    dataset = np.load(input_dir)
    data = dataset["data"]
    return data.tolist()


def loadTypesNpz(input_dir):
    dataset = np.load(input_dir)
    return dataset["types"].item()


def load_relations_dict(input_path):
    dataset = np.load(input_path)
    dict_rel = dataset["relations_dict"]
    if not isinstance(dict_rel, dict):
        dict_rel = dict_rel.item()
    return dict_rel


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
