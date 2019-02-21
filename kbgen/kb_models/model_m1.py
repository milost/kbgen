from multiprocessing import Process, Queue
from multiprocessing.managers import SyncManager
from queue import Empty

from tqdm import tqdm
import datetime
import logging
import numpy as np

from rdflib import Graph, RDF, OWL, RDFS, URIRef
from numpy.random import choice, randint
from typing import Dict, Tuple, List, Iterable, Optional
from scipy.sparse import csr_matrix, coo_matrix

from ..load_tensor_tools import load_graph_npz, load_types_npz, load_types_dict, load_relations_dict, \
    load_type_hierarchy, load_prop_hierarchy, load_domains, load_ranges, load_single_adjacency_matrix, \
    num_adjacency_matrices
from .model import KBModel
from ..tensor_models import DAGNode
from ..util_models import URIEntity, URIRelation, URIType, MultiType
from ..util import normalize, create_logger


class KBModelM1(KBModel):
    """
    Simplest knowledge base model composed of the distribution of entities over types and the joint distribution of
    relations, subject and object type sets (represented with the chain rule)
    """

    def __init__(self,
                 entity_type_hierarchy: Dict[int, DAGNode],
                 object_property_hierarchy: Dict[int, DAGNode],
                 domains: Dict[int, int],
                 ranges: Dict[int, int],
                 entity_count: int,
                 relation_count: int,
                 edge_count: int,
                 entity_type_count: int,
                 entity_type_distribution: Dict[MultiType, float],
                 relation_distribution: Dict[int, int],
                 relation_domain_distribution: Dict[int, Dict[MultiType, int]],
                 relation_range_distribution: Dict[int, Dict[MultiType, Dict[MultiType, int]]],
                 relation_to_id: Dict[URIRef, int],
                 entity_type_to_id: Dict[URIRef, int],
                 multitype_index: Dict[frozenset, int]):
        """
        Creates an M1 model with the passed data.
        :param entity_type_hierarchy: dictionary pointing from an entity type's id to its DAGNode in the hierarchy of
                                      entity types
        :param object_property_hierarchy: dictionary pointing from an object property's id to its DAGNode in the
                                          hierarchy of object properties
        :param domains: dictionary pointing from property ids to the id of the entity type, whose instances are
                        described by that property
        :param ranges: dictionary pointing from property ids to the id of the entity type id, whose instances are the
                       values of that property
        :param entity_count: the number of entities in the rdf data
        :param relation_count: the number of relations in the rdf data (size of relation_id)
        :param edge_count: the number of edges of object properties in the graph (facts)
        :param entity_type_count: the number of entity types in the rdf data (size of entity_type_id)
        :param entity_type_distribution: the distribution of entities over entity types. A dictionary that points from a
                                         multi type (a set of entity types) to its number of occurrences
        :param relation_distribution: the distribution of facts over relations (object properties). A dictionary
                                      pointing from the id of an object property (relation) to the number of edges that
                                      exist for that relation
        :param relation_domain_distribution: the distribution of subject entity types given the relation
                                             (object property). A dictionary pointing from an relation id to a
                                             dictionary that points from a subject's multi type to a number of
                                             occurrences of that subject's multi type
        :param relation_range_distribution: the distribution of object entity types given relation and subject entity
                                            type. A dictionary pointing from an relation id to a dictionary that points
                                            from a subject's multi type to a dictionary that points from an object's
                                            multi type to the number of occurrences of that object's multi type
        :param relation_to_id: dictionary of the RDF relations and their ids
        :param entity_type_to_id: dictionary of the RDF entity types and their ids
        :param multitype_index: index pointing from multitype to its index used in the distributions
        """
        super(KBModelM1, self).__init__()
        self.entity_count = entity_count
        self.relation_count = relation_count
        self.edge_count = edge_count
        self.entity_type_count = entity_type_count

        self.entity_type_distribution = entity_type_distribution
        self.relation_distribution = relation_distribution
        self.relation_domain_distribution = relation_domain_distribution
        self.relation_range_distribution = relation_range_distribution

        self.relation_to_id = relation_to_id
        self.entity_type_to_id = entity_type_to_id
        self.domains = domains
        self.ranges = ranges

        self.entity_type_hierarchy = entity_type_hierarchy
        self.object_property_hierarchy = object_property_hierarchy
        self.fix_hierarchies()

        self.replace_multitype_indices(multitype_index)

        # used for logging
        self.name = "M1 (NAIVE)"

        # initialized in other methods
        #
        # normal logger and logger for the synthesizing performance
        self.logger = None
        self.synth_time_logger = None

        # used to scale the synthesized knowledge base
        self.step: float = None
        # points from synthetic entity id to the multi type of that synthetic entity
        self.synthetic_id_to_type: Dict[int, MultiType] = None
        # points from a multi type to the list of synthetic entity ids of the synthetic entity with that multi type
        self.entity_types_to_entity_ids: Dict[MultiType, List[int]] = None

        # the number of facts that were added to the synthesized graph
        self.fact_count: int = 0
        # the number of facts that were generate but were a duplicate of a previously generated fact
        self.duplicate_fact_count: int = 0
        # progress bar used for synthesizing relations
        self.progress_bar: tqdm = None
        # start time delta used for measuring of synthesizing relations
        self.start_t: datetime = None

    def fix_hierarchies(self):
        """
        Replaces the references to other DAG nodes in the lists of children and parents in every DAG node with the
        unique node ids of that DAG node to prevent recursion (e.g., when serializing the DAG). This is also done
        in load_tensor and is only done here if it wasn't already done before.
        """
        for entity_type_id, entity_type_dag_node in self.entity_type_hierarchy.items():
            entity_type_dag_node.children = [child if isinstance(child, int) else child.node_id
                                             for child in entity_type_dag_node.children]
            entity_type_dag_node.parents = [parent if isinstance(parent, int) else parent.node_id
                                            for parent in entity_type_dag_node.parents]

        for property_id, property_dag_node in self.object_property_hierarchy.items():
            property_dag_node.children = [child if isinstance(child, int) else child.node_id
                                          for child in property_dag_node.children]
            property_dag_node.parents = [parent if isinstance(parent, int) else parent.node_id
                                         for parent in property_dag_node.parents]

    def print_synthesis_details(self) -> None:
        """
        Print the statistics of the synthesization of a knowledge base with this M1 model.
        """
        self.logger.debug(f"{self.duplicate_fact_count} facts had to be disregarded because they already existed")
        self.logger.debug(f"{self.fact_count} facts added")

    def check_for_quadratic_relations(self) -> List[int]:
        """
        This check is only done properly in the M2 and M3 model, since measures are used that are only defined there.
        Therefore, this method is only a placeholder and returns an empty list.
        :return: a list of relation ids, each of which is a quadratic relation
        """
        return []

    def adjust_quadratic_relation_distributions(self,
                                                relation_distribution: Dict[int, float],
                                                quadratic_relations: List[int]) -> Dict[int, float]:
        """
        The number of occurrences of the passed quadratic relations is scaled with the size parameter of the synthesized
        knowledge base (i.e., num_occurrences = num_occurrences * size).
        TODO: why is this done
        :param relation_distribution: the distribution of the relations. A dictionary pointing from relation ids to
                                      the number of occurrences of that relation type
        :param quadratic_relations: a list of relation ids of which each relation is a quadratic relation
        :return: the relation distribution with the adjusted occurrences of the quadratic relations
        """
        self.logger.debug(f"Adjusting distribution because of quadratic relations {quadratic_relations}")
        for relation_id in quadratic_relations:
            relation_distribution[relation_id] = relation_distribution[relation_id] / self.step
        return relation_distribution

    def add_fact(self, graph: Graph, fact: Tuple[str, str, str]) -> bool:
        """
        Add a fact (triple) to the graph. Afterwards check if this fact was not previously generated or if it was
        already known and increment the appropriate counters.
        :param graph: the synthesized graph object
        :param fact: triple of subject, relation, object
        :return: boolean indicating if the fact was a new fact
        """
        graph_size = len(graph)
        graph.add(fact)
        # if the fact was a new fact and increased the size of the graph
        if len(graph) > graph_size:
            # increment counters and log synthesis details every n facts
            self.progress_bar.update(1)
            self.fact_count += 1
            if self.fact_count % 10000 == 0:
                self.print_synthesis_details()

            # log the time it took for the current graph size
            delta = datetime.datetime.now() - self.start_t
            self.synth_time_logger.debug(f"{len(graph)}, {delta.microseconds / 1000}")
            return True
        else:
            self.duplicate_fact_count += 1
            return False

    def synthesize_entity_types(self, graph: Graph, entity_type_count: int) -> Graph:
        """
        Adds a triple defining the entity type relation for every entity type. The ids of these entity types are in
        [0, entity_type_count). These triples look like (entity_type, is of type, class).
        :param graph: the synthesized graph object
        :param entity_type_count: the number of entity types that should be synthesized
        :return: the synthesized graph object with triples that define the entity types
        """
        self.logger.info("Synthesizing entity types...")

        for entity_type_id in range(entity_type_count):
            entity_type = URIType(entity_type_id).uri
            graph.add((entity_type, RDF.type, OWL.Class))

        self.logger.info(f"{entity_type_count} types added")
        return graph

    def synthesize_schema(self, graph: Graph) -> Graph:
        """
        Adds the triples defining the schema related information to the synthesized graph. This schema is the entity
        type hierarchy, the property hierarchy, the property domains and the property ranges.
        :param graph: the synthesized graph object
        :return: the synthesized graph object with the triples defining the entity type hierarchy, the property
                 hierarchy, the property domains and the property ranges
        """
        self.logger.info("Synthesizing schema...")

        # remove recursions from entity type and object property hierarchies if not already done
        self.fix_hierarchies()

        # add the entity type hierarchy relations to the synthesized graph
        if self.entity_type_hierarchy:
            for entity_type_id, entity_type_node in self.entity_type_hierarchy.items():
                # the actual entity type (uri)
                entity_type = URIType(entity_type_id).uri

                # add the subclass relations between the current entity type and its parents/superclasses to the graph
                for parent in entity_type_node.parents:
                    # remove recursion from hierarchy by replacing parent DAG nodes with the ids of these nodes if they
                    # aren't the ids yet
                    # this should never happen
                    if not isinstance(parent, int):
                        parent_node = self.entity_type_hierarchy[parent]
                        parent = parent_node.node_id

                    # the actual entity type of the current parent (superclass)
                    entity_type_parent = URIType(parent).uri

                    # add the subclass relation between the current entity type and its current parent
                    graph.add((entity_type, RDFS.subClassOf, entity_type_parent))

        # add the property type hierarchy relations to the synthesized graph
        if self.object_property_hierarchy:
            for property_id, property_type_node in self.object_property_hierarchy.items():
                # the actual property type (uri)
                property_type = URIRelation(property_id).uri

                # add the subclass relations between the current property type and its parents/superclasses to the graph
                for parent in property_type_node.parents:
                    # remove recursion from hierarchy by replacing parent DAG nodes with the ids of these nodes if they
                    # aren't the ids yet
                    # this should never happen
                    if not isinstance(parent, int):
                        parent_node = self.object_property_hierarchy[parent]
                        parent = parent_node.node_id

                    # the actual property type of the current parent (superclass)
                    property_type_parent = URIRelation(parent).uri

                    # add the subclass relation between the current property type and its current parent
                    graph.add((property_type, RDFS.subPropertyOf, property_type_parent))

        # add the domain triples to the synthesized graph
        if self.domains:
            for property_id, entity_type_id in self.domains.items():
                object_property = URIRelation(property_id).uri
                object_property_domain = URIType(entity_type_id).uri
                graph.add((object_property, RDFS.domain, object_property_domain))

        # add the range triples to the synthesized graph
        if self.ranges:
            for property_id, entity_type_id in self.ranges.items():
                object_property = URIRelation(property_id).uri
                object_property_range = URIType(entity_type_id).uri
                graph.add((object_property, RDFS.range, object_property_range))

        return graph

    def synthesize_relations(self, graph: Graph, relation_count: int) -> Graph:
        """
        Adds a triple defining the relation (type) for every relation. The ids of these relations are in
        [0, relation_count). These triples look like (relation, is of type, object property).
        :param graph: the synthesized graph object
        :param relation_count: the number of different relations that should be synthesized
        :return: the synthesized graph object with triples that define the different relations
        """
        self.logger.info("Synthesizing relations...")

        for relation_id in range(relation_count):
            relation = URIRelation(relation_id).uri
            graph.add((relation, RDF.type, OWL.ObjectProperty))

        self.logger.info(f"{self.relation_count} relations added")
        return graph

    def synthesize_entities(self,
                            graph: Graph,
                            synthetic_entity_count: int) -> Tuple[Graph, Dict[MultiType, List[int]]]:
        """
        Adds all triples defining the is type of relation for every synthetic entity. Every entity is assigned a
        multi type at random (via the entity type distribution) and a type of relation is added for every entity type
        in that multi type.
        :param graph: the synthesized graph object
        :param synthetic_entity_count: the number of synthetic entities that are added
        :return: a tuple of the synthesized graph object with triples added that define the entity types of the
                 synthetic entities and a dictionary of the multi types pointing to a list of ids of the synthethic
                 entities of that multi type
        """
        self.logger.info("Synthesizing entities...")

        # dictionary that points from an entity type set (multi type) to a list of synthetic entity ids of that type
        entity_types_to_entity_ids = {}

        # initialize the value of every multi type as empty list
        for multi_type in self.entity_type_distribution.keys():
            entity_types_to_entity_ids[multi_type] = []

        # select a random entity type set (multi type) of every entity that will be synthesized determined by the
        # normalized number of occurrences of every entity type
        entity_types = choice(
            list(self.entity_type_distribution.keys()),  # the values that are sampled (multi types)
            size=synthetic_entity_count,  # the number of samples that are returned
            replace=True,
            p=normalize(self.entity_type_distribution.values()))  # normalized occurrences here used as probabilities

        # the number of is type of relations that are added for synthetic entities
        type_assertions = 0

        # progess bar used for cli
        progess_bar = tqdm(total=synthetic_entity_count)

        # add all triples that define the is type of entity type relations for every synthetic entity
        for synthetic_entity_id in range(synthetic_entity_count):
            # randomly selected entity type for the current synthetic entity
            type_of_synthetic_entity = entity_types[synthetic_entity_id]

            # append entity id of the current synthetic entity to the list of synthetic entities with that entity type
            entity_types_to_entity_ids[type_of_synthetic_entity].append(synthetic_entity_id)

            # the actual synthetic entity
            synthetic_entity = URIEntity(synthetic_entity_id).uri

            # add the rdf is type of relation to every entity type in the multi type for the current synthetic entity
            for multi_type_subtype_id in entity_types[synthetic_entity_id].types:
                multi_type_subtype = URIType(multi_type_subtype_id).uri
                graph.add((synthetic_entity, RDF.type, multi_type_subtype))
                type_assertions += 1

            progess_bar.update(1)

        self.logger.debug(f"{synthetic_entity_count} entities with {type_assertions} type assertions added")
        return graph, entity_types_to_entity_ids

    # TODO
    def select_instance(self, n, model=None):
        return randint(n)

    def select_subject_model(self, relation_id: int, relation_domain: MultiType):
        """
        Implemented in model_emi. Return a model that will select one of many entities of the given multi type as a
        subject for a relation.
        :param relation_id: id of the relation type for which a subject will be chosen
        :param relation_domain: the domain of the relation, i.e., the multi type of the entity that will be the subject
        :return: TODO
        """
        return None

    def select_object_model(self, relation_id: int, relation_domain: MultiType, relation_range: MultiType):
        """
        Implemented in model_emi. Return a model that will select one of many entities of the given range multi type as
        an object for a relation with a given subject multi type (the domain).
        :param relation_id: id of the relation type for which a subject will be chosen
        :param relation_domain: the domain of the relation, i.e., the multi type of the entity that will be the subject
        :param relation_range: the range of the relation, i.e., the multi type of the entity that will be the object
                               given the multi type of the subject
        :return: TODO
        """
        return None

    def initialize_graph_with_metadata(self) -> Graph:
        """
        Initializes the graph with the triples defining the entity types as such, the triples defining the relations as
        such and the triples defining the type hierarchies of entity types and relations.
        :return: the graph initialized with the triples defining the "metadata" (types and type hierarchies)
        """
        # the synthesized graph (initialised emtpy)
        graph = Graph()

        # adds the triples that define the entity types as such (classes)
        graph = self.synthesize_entity_types(graph, self.entity_type_count)

        # adds the triples that define the relations as such (object properties)
        graph = self.synthesize_relations(graph, self.relation_count)

        # adds the triples that define the entity type and property type hierarchies and the property domains and ranges
        graph = self.synthesize_schema(graph)

        return graph

    def insert_synthetic_entities(self, graph: Graph, num_synthetic_entities: int) -> Graph:
        """
        Insert synthetic entities into the graph according to the distribution of entity types (multi types).
        :param graph: the synthetic graph object
        :param num_synthetic_entities: the number of entities to synthesize
        :return: the synthetic graph object with synthetic entities
        """
        # add synthetic entities to the graph by adding the triples that define the type of relation for every entity
        # synthetic entities are assigned multi types at random (via distribution) and the type of relation is added
        # for every entity type in that multi type
        # returns the new graph and a dictionary containing the used multi types pointing to the entity ids of the
        # synthetic entities of that type
        graph, entity_types_to_entity_ids = self.synthesize_entities(graph, num_synthetic_entities)

        # reverse the dictionary so that every synthetic entity id points to its multi type
        self.synthetic_id_to_type = {synthetic_entity_id: multi_type for multi_type in entity_types_to_entity_ids.keys()
                                     for synthetic_entity_id in entity_types_to_entity_ids[multi_type]}
        self.entity_types_to_entity_ids = entity_types_to_entity_ids

        return graph

    def normalize_relation_distribution(self) -> Dict[int, float]:
        """
        Normalize the relation distribution so that it can be used as probabilities for random selections.
        :return: dictionary pointing from a relation id to the normalized occurrences of that relation
        """
        # TODO
        quadratic_relations = self.check_for_quadratic_relations()
        adjusted_relations_distribution = self.adjust_quadratic_relation_distributions(self.relation_distribution,
                                                                                       quadratic_relations)

        # normalize relation distribution
        # first sort the dictionary by the relation id (which are in [0, num_relations)) then normalize them
        sorted_values = [occurrences for relation_id, occurrences in sorted(adjusted_relations_distribution.items())]
        normalized_values = list(normalize(sorted_values))
        # dictionary of relation ids pointing to their frequency (in [0, 1])
        normalized_relation_distribution: Dict[int, float] = {}
        # add normalized relation occurrences to the distribution dictionary
        for relation_id in adjusted_relations_distribution.keys():
            normalized_relation_distribution[relation_id] = normalized_values[relation_id]
        return normalized_relation_distribution

    def normalize_relation_domain_distribution(self) -> Dict[int, Iterable[float]]:
        """
        Normalize the relation domain distribution so that it can be used as probabilities for random selections.
        :return: dictionary pointing from a relation id to a list of the normalized occurrences of the multi types
                 that are the entity types of the subjects of that relation. The indices of the normalized values
                 correspond with the multi types in the un-normalized distribution
        """
        # normalize relation domain distribution
        # this is basically just copies relation_domain_distribution and replaces the occurrences with normalized values
        normalized_relation_domain_distribution = {}

        # iterate over the ids of the synthesized relation types
        for relation_id in range(self.relation_count):
            normalized_distribution = normalize(self.relation_domain_distribution[relation_id].values())
            normalized_relation_domain_distribution[relation_id] = normalized_distribution

        return normalized_relation_domain_distribution

    def normalize_relation_range_distribution(self) -> Dict[int, Dict[MultiType, Iterable[float]]]:
        """
        Normalize the relation range distribution so that it can be used as probabilities for random selections.
        :return: dictionary pointing from a relation id to a dictionary pointing from the multi types that occurred as
                 subject in the relation pointing to a list of the normalized occurrences of the multi types that
                 occurred as object in the relation with that specific subject. The indices of the normalized values
                 correspond with the multitypes in the un-normalized distribution
        """

        # normalize relation range distribution
        # this is basically just copies relation_range_distribution and replaces the occurrences with normalized values
        normalized_relation_range_distribution: Dict[int, Dict[MultiType, Iterable[float]]] = {}

        # iterate over the ids of the synthesized relation types
        for relation_id in range(self.relation_count):
            normalized_relation_range_distribution[relation_id] = {}
            for relation_domain in self.relation_range_distribution[relation_id].keys():
                relation_range_values = self.relation_range_distribution[relation_id][relation_domain].values()
                normalized_relation_range_distribution[relation_id][relation_domain] = normalize(relation_range_values)

        return normalized_relation_range_distribution

    def synthesize_fact(self,
                        relation_id: int,
                        normalized_relation_domain_distribution: Dict[int, Iterable[float]],
                        normalized_relation_range_distribution: Dict[int, Dict[MultiType, Iterable[float]]]
                        ) -> Optional[Tuple[str, str, str]]:
        # select random domain of the valid domains for this relation type according to the domain distribution
        # a domain is a multi type
        relation_domain = choice(list(self.relation_domain_distribution[relation_id].keys()),
                                 p=normalized_relation_domain_distribution[relation_id])

        # the number of synthetic entities with the multi type of the selected domain
        entity_domain_count = len(self.entity_types_to_entity_ids[relation_domain])

        # select random range of the valid ranges for the selected domain according to the range distribution
        # a range is a multi type
        relation_range = choice(list(self.relation_range_distribution[relation_id][relation_domain].keys()),
                                p=normalized_relation_range_distribution[relation_id][relation_domain])

        # the number of synthetic entities with the multi type of the selected range
        entity_range_count = len(self.entity_types_to_entity_ids[relation_range])

        # only continue if there exist synthetic entities with the correct multi type ofr the domain as well as
        # the range
        if entity_domain_count > 0 and entity_range_count > 0:
            # TODO
            subject_model = self.select_subject_model(relation_id, relation_domain)
            # TODO
            object_model = self.select_object_model(relation_id, relation_domain, relation_range)

            # select one of the possible entities as a subject according to the subject model
            possible_subject_entities = self.entity_types_to_entity_ids[relation_domain]
            subject_entity = possible_subject_entities[self.select_instance(entity_domain_count, subject_model)]

            # select one of the possible entities as an object according to the object model
            possible_object_entities = self.entity_types_to_entity_ids[relation_range]
            object_entity = possible_object_entities[self.select_instance(entity_range_count, object_model)]

            # create fact with the ids of the entities and add it to the graph
            rdf_relation = URIRelation(relation_id).uri
            rdf_subject = URIEntity(subject_entity).uri
            rdf_object = URIEntity(object_entity).uri

            fact = (rdf_subject, rdf_relation, rdf_object)

            return fact

    def is_fact_valid(self, graph: Graph, relation_id: int, fact: Tuple[str, str, str]):
        """
        Function used to filter synthesized facts before they are added to the synthesized knowledge graph. In this M1
        Model there is no filtering and thus it is only a placeholder.
        :param fact: the synthesized that will be added to the knowledge graph
        :return: True if the fact is valid and can be added to the knowledge graph
        """
        return True

    def synthesize(self,
                   size: float = 1.0,
                   number_of_entities: int = None,
                   number_of_edges: int = None,
                   debug: bool = False,
                   pca: bool = True) -> Graph:
        """
        Synthesizes a knowledge base of a given size either determined by a scaling factor or a static number of
        entities and edges.
        :param size: scale of the synthesized knowledge base (e.g., 1.0 means it should have the same size as the KB
                     the model was trained on, 2.0 means it should have twice the size)
        :param number_of_entities: the number of entities the synthesized graph should have. If not set, this number
                                   will be determined by the number of entities on which the model was trained and the
                                   size parameter
        :param number_of_edges: the number of edges (facts) the synthesized graph should have. If not set this number
                                will be determined by the number of edges on which the model was trained and the size
                                parameter
        :param debug: boolean if logging should be on debug level
        :param pca: boolean if PCA (partial completeness assumption) should be used. This parameter is not used
        :return: the synthesized graph as rdf graph object
        """
        print(f"Synthesizing {self.name} model...")

        level = logging.DEBUG if debug else logging.INFO
        self.logger = create_logger(level, name="kbgen")
        self.synth_time_logger = create_logger(level, name="synth_time")

        # scale the entity and edge count by the given size
        self.step = 1.0 / size
        num_synthetic_entities = int(self.entity_count / self.step)
        num_synthetic_facts = int(self.edge_count / self.step)

        # overwrite dynamic sizes with static sizes if they were set
        if number_of_entities is not None:
            num_synthetic_entities = number_of_entities
        if number_of_edges is not None:
            num_synthetic_facts = number_of_edges

        # initialize graph with the triples defining the entity types, relations and type hierarchies
        graph = self.initialize_graph_with_metadata()

        # insert synthetic entities whose entity types mirror the distribution of entity types
        graph = self.insert_synthetic_entities(graph, num_synthetic_entities)

        # normalize the distributions of the relation properties so that they can be used for random selections
        normalized_relation_distribution = self.normalize_relation_distribution()
        normalized_relation_domain_distribution = self.normalize_relation_domain_distribution()
        normalized_relation_range_distribution = self.normalize_relation_range_distribution()

        # counter for the number of facts in the graph
        self.fact_count = 0
        # counter of duplicate facts that were generated
        self.duplicate_fact_count = 0

        self.logger.info(f"{num_synthetic_facts} facts to be synthesized")
        # progress bar
        self.progress_bar = tqdm(total=num_synthetic_facts)
        # start delta used for time logging
        self.start_t = datetime.datetime.now()

        # repeat until enough facts are generated
        while self.fact_count < num_synthetic_facts:
            # choose a random relation type according to the relation distribution
            relation_id = choice(list(self.relation_distribution.keys()),
                                 replace=True,
                                 p=list(normalized_relation_distribution.values()))
            if relation_id in self.relation_distribution.keys():
                # synthesize a fact using the distributions of the relations and their domains and ranges
                fact = self.synthesize_fact(
                    relation_id=relation_id,
                    normalized_relation_domain_distribution=normalized_relation_domain_distribution,
                    normalized_relation_range_distribution=normalized_relation_range_distribution
                )

                # test if a fact was synthesized and if it passes the validity checks of this model
                if fact is not None and self.is_fact_valid(graph, relation_id, fact):
                    self.add_fact(graph, fact)

        self.print_synthesis_details()
        self.logger.info(f"Synthesized facts = {self.fact_count} from {num_synthetic_facts}")
        return graph

    def replace_multitype_indices(self, multitype_index: Dict[frozenset, int]):
        """
        Replaces the multitypes indices in the distributions with the proper objects. This needs to be called after
        learning this model.
        :param multitype_index: the multitype index dictionary pointing from mutlitype to the index
        """
        print("Replacing multitype indices with objects...")
        domain_distributions = self.relation_domain_distribution
        range_distribution = self.relation_range_distribution
        reverse_index = {index: MultiType(multitype) for multitype, index in multitype_index.items()}

        domain_distributions = {relation_id: {reverse_index[multitype]: count
                                              for multitype, count in distribution.items()}
                                for relation_id, distribution in domain_distributions.items()}
        self.relation_domain_distribution = domain_distributions

        range_distribution = {relation_id: {reverse_index[subject_type]: {reverse_index[object_type]: count
                                                                          for object_type, count in distribution.items()
                                                                          }
                                            for subject_type, distribution in domains.items()}
                              for relation_id, domains in range_distribution.items()}
        self.relation_range_distribution = range_distribution


    @staticmethod
    def extract_relation_distribution(relation_adjacency_matrices: List[coo_matrix], entity_types: csr_matrix):

        class MultiTypeIndex(dict):
            def __init__(self, *args):
                super(MultiTypeIndex, self).__init__(args)

            def __getitem__(self, item):
                if item not in self:
                    index = len(self)
                    self.__setitem__(item, index)
                    return index
                else:
                    return super().__getitem__(item)

            def to_dict(self):
                return dict(self)

        multitype_index = MultiTypeIndex()
        dense_entity_types = entity_types.toarray()

        # the distribution of subject entity types given the relation (object property)
        # the number of times that an entity type set (multi type) occurred as a subject in a specific relation for
        # every relation
        # dictionary pointing from an relation id to a dictionary that points from a subject's multi type to a
        # number of occurrences of that subject's multi type
        relation_domain_distribution: Dict[int, Dict[int, int]] = {}

        # the distribution of object entity types given relation and subject entity type
        # the number of times that a multi types occurs in a relation as an object for every relation and for every
        # subject's multi type (i.e., the number of occurrences for a specific relation with a specific subject's entity
        # type)
        # dictionary pointing from an relation id to a dictionary that points from a subject's multi type to a
        # dictionary that points from an object's multi type to the number of occurrences of that object's multi type
        relation_range_distribution: Dict[int, Dict[int, Dict[int, int]]] = {}

        for relation_id in tqdm(range(len(relation_adjacency_matrices))):
            # create empty inner dictionaries for the current relation
            domain_distribution = {}
            range_distribution = {}

            subject_ids_row = relation_adjacency_matrices[relation_id].row
            object_ids_row = relation_adjacency_matrices[relation_id].col

            # iterate over all non-zero fields of the adjacency matrix
            # iterate over all edges that exist for the current relation
            for index in range(relation_adjacency_matrices[relation_id].nnz):
                # get subject and object id from the adjacency matrix
                subject_id = subject_ids_row[index]
                object_id = object_ids_row[index]
                # create multi types from the two sets of entity types for each the subject and the object

                multi_type, = dense_entity_types[subject_id].nonzero()
                subject_multi_type = multitype_index[frozenset(multi_type)]

                multi_type, = dense_entity_types[object_id].nonzero()
                object_multi_type = multitype_index[frozenset(multi_type)]

                # if the subject's multi type is not known add it to the relation domain distribution and create an
                # empty relation range distribution for that multi type
                if subject_multi_type not in domain_distribution:
                    domain_distribution[subject_multi_type] = 0
                    range_distribution[subject_multi_type] = {}

                # if the object's multi type is not known add it to the relation range distribution of the subject's
                # multi type
                if object_multi_type not in range_distribution[subject_multi_type]:
                    range_distribution[subject_multi_type][object_multi_type] = 0

                # increment the number of occurrences of the subject's multi type in the relation domain distribution
                    domain_distribution[subject_multi_type] += 1
                # increment the number of occurrences of the object's multi type in the relation range distribution
                # of the subject's multi type
                    range_distribution[subject_multi_type][object_multi_type] += 1

            relation_domain_distribution[relation_id] = domain_distribution
            relation_range_distribution[relation_id] = range_distribution

        return relation_domain_distribution, relation_range_distribution, multitype_index.to_dict()

    @staticmethod
    def generate_from_tensor(input_path: str, debug: bool = False) -> 'KBModelM1':
        """
        Generates an M1 model from the specified tensor file.
        :param input_path: path to the directory containing the numpy-serialized data
        :param debug: boolean indicating if the logging level is on debug
        :return: an M1 model generated from the tensor file
        """
        if debug:
            logger = create_logger(logging.DEBUG, log_to_console=True)
        else:
            logger = create_logger(logging.INFO, log_to_console=True)

        logger.info("Loading data...")

        # the list of adjacency matrices of the object property relations created in load_tensor
        relation_adjacency_matrices = load_graph_npz(input_path)

        # the entity type adjacency matrix created in load_tensor
        entity_types = load_types_npz(input_path)

        # the rdf domains and ranges created in load_tensor
        domains = load_domains(input_path)
        ranges = load_ranges(input_path)

        # the entity type and the object property hierarchies created in load_tensor
        entity_type_hierarchy = load_type_hierarchy(input_path)
        object_property_hierarchy = load_prop_hierarchy(input_path)

        # the dictionaries of entity types and of object properties pointing to their ids
        entity_type_to_id = load_types_dict(input_path)
        # relations are object properties
        relation_to_id = load_relations_dict(input_path)

        # compress entity type adjacency matrix if it is not already compressed
        if not isinstance(entity_types, csr_matrix):
            entity_types = entity_types.tocsr()

        logger.info("Learning types distributions...")
        # the entity type adjacency matrix has the dimensions num_entities x num_entity_types
        count_entities = entity_types.shape[0]
        count_types = entity_types.shape[1]
        # number of different relations (object properties)
        count_relations = len(relation_adjacency_matrices)
        # number of object property edges (number of non zero fields in every adjacency matrix)
        count_facts = sum([Xi.nnz for Xi in relation_adjacency_matrices])

        # the distribution of entities over entity types
        # the number of occurrences of every set of entity types that occurs
        # dictionary that points from a multi type (a set of entity types) to its number of occurrences
        entity_type_distribution = {}
        for entity_type in entity_types:
            entity_type_set = MultiType(entity_type.indices)

            # add the multi type to the distribution if it is new
            if entity_type_set not in entity_type_distribution:
                entity_type_distribution[entity_type_set] = 0.0

            entity_type_distribution[entity_type_set] += 1

        logger.info("Building learning relations distributions...")
        # the distribution of facts over relations (object properties)
        # dictionary pointing from the id of an object property (relation) to the number of edges that exist for that
        # relation
        relation_distribution = {relation_id: relation_adjacency_matrices[relation_id].nnz
                                 for relation_id in range(len(relation_adjacency_matrices))}

        computed_distributions = KBModelM1.extract_relation_distribution(relation_adjacency_matrices, entity_types)
        domain_distribution, range_distribution, multitype_index = computed_distributions

        naive_model = KBModelM1(
            entity_type_hierarchy=entity_type_hierarchy,
            object_property_hierarchy=object_property_hierarchy,
            domains=domains,
            ranges=ranges,
            entity_count=count_entities,
            relation_count=count_relations,
            edge_count=count_facts,
            entity_type_count=count_types,
            entity_type_distribution=entity_type_distribution,
            relation_distribution=relation_distribution,
            relation_domain_distribution=domain_distribution,
            relation_range_distribution=range_distribution,
            relation_to_id=relation_to_id,
            entity_type_to_id=entity_type_to_id,
            multitype_index=multitype_index
        )

        return naive_model
