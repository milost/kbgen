from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import Dict, List

from rdflib import Graph
from rdflib.namespace import RDF, RDFS, OWL

from tqdm import tqdm
from scipy.sparse import coo_matrix

from kbgen import load_tensor_tools as ltt
from kbgen.tensor_models import DAGNode

from .interfaces import MultiProcessingTask


class TensorLoader(MultiProcessingTask):
    def __init__(self, input: str, num_processes: int):
        super(TensorLoader, self).__init__(num_processes)
        self.input = input
        self.process_type = LearnAdjacencyMatrixProcess

    def _load_graph(self, input_dir: str, rdf_format: str) -> Graph:
        """
        Loads the RDF graph from a tensor file into a Knowledge Graph.
        :param input_dir: path to the .n3 file of the RDF data
        :param rdf_format: the rdf format of the input file
        :return: tuple of the created graph object and the format of the rdf file
        """
        try:
            # Check if graph exists and load it from that file if possible
            graph = ltt.load_graph_binary(input_dir)
            return graph
        except FileNotFoundError:
            print("File does not exist. Loading graph from input data...")

        graph = Graph()
        graph.parse(self.input, format=rdf_format)

        # Save graph as binary file
        ltt.save_graph_binary(input_dir, graph)
        return graph

    def _load_entity_types(self, input_dir: str, graph: Graph) -> Dict[str, int]:
        """
        Extracts the "type of" and "class of" relations from the Knowledge Graph and creates a dictionary of the entity
        types and their unique identifiers.
        :param input_dir: path to the .n3 file of the RDF data
        :param graph: the graph object of the Knowledge Graph
        :return: a dictionary of entity types pointing to their id
        """
        try:
            entity_type_to_id = ltt.load_types_dict(input_dir)
            return entity_type_to_id
        except FileNotFoundError:
            pass

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

        ltt.save_types_dict(input_dir, entity_type_to_id)
        return entity_type_to_id

    def _load_entities(self, input_dir: str, graph: Graph, entity_type_to_id: Dict[str, int]) -> Dict[str, int]:
        """
        Extracts the entities from the Knowledge Graph and creates a dictionary of the entities and their unique
        identifiers.
        :param input_dir: path to the .n3 file of the RDF data
        :param graph: the graph object of the Knowledge Graph
        :param entity_type_to_id: the dictionary of the entity types and their unique identifies
        :return: a dictionary of entities pointing to their id
        """
        try:
            entity_to_id = ltt.load_entities_dict(input_dir)
            return entity_to_id
        except FileNotFoundError:
            pass

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

        ltt.save_entities_dict(input_dir, entity_to_id)
        return entity_to_id

    def _load_properties(self, input_dir: str, graph: Graph, entity_to_id: Dict[str, int]) -> Dict[str, int]:
        """
        Extracts the object properties/entity properties from the Knowledge graph and creates a dictionary of the properties
        and their unique identifiers.
        :param input_dir: path to the .n3 file of the RDF data
        :param graph: the graph object of the Knowledge Graph
        :param entity_to_id: the dictionary of the entities and their unique identifies
        :return: a dictionary of object properties/entity properties pointing to their id
        """
        try:
            property_to_id = ltt.load_relations_dict(input_dir)
            return property_to_id
        except FileNotFoundError:
            pass

        # dictionary of object properties
        # contains tuples of (property, property_id)
        property_to_id = {}

        print("Loading object properties...")
        predicate_id = 0
        # iterate over all triples
        for subject, predicate, object in graph:
            # if the predicate hasn't been seen yet and if both subject and object are non-type entities, then add the
            # predicate to the predicate dictionary with a new id
            if predicate not in property_to_id:
                if subject in entity_to_id and object in entity_to_id:
                    property_to_id[predicate] = predicate_id
                    predicate_id += 1

        print("Iterating over OWL.ObjectProperty...")
        # iterate over all triples that define the subject as an object property
        for subject, predicate, object in graph.triples((None, OWL.ObjectProperty, None)):
            # if the object property is an unknown predicate, add it to the predicate dictionary with a new id
            if subject not in property_to_id:
                property_to_id[predicate] = predicate_id
                predicate_id += 1

        print(f"{len(property_to_id)} object properties loaded...")

        ltt.save_relations_dict(input_dir, property_to_id)
        return property_to_id

    def _load_property_adjacency_matrices(self,
                                          input_dir: str,
                                          graph: Graph,
                                          entity_to_id: Dict[str, int],
                                          property_to_id: Dict[str, int]) -> List[coo_matrix]:
        """
        Creates an adjacency matrix for every object property.
        :param input_dir: path to the .n3 file of the RDF data
        :param graph: the graph object of the Knowledge Graph
        :param entity_to_id: the dictionary of the entities and their unique identifies
        :param property_to_id: the dictionary of the object properties and their unique identifies
        :return: a list of adjacency matrices, where the adjacency matrix at list index i belongs to the object property
        with the unique identifier i
        """
        if Path(ltt.graph_npz_dir(input_dir)).exists():
            property_adjaceny_matrices = ltt.load_graph_npz(input_dir)
            return property_adjaceny_matrices

        num_relations = len(property_to_id)
        matrices: List[coo_matrix] = [None for _ in range(num_relations)]
        task_queue = Queue()
        result_queue = Queue()

        self.processes = self.create_processes(task_queue=task_queue,
                                               result_queue=result_queue,
                                               graph=graph,
                                               entity_to_id=entity_to_id,
                                               property_to_id=property_to_id)

        print(f"Filling task queue with {num_relations} tasks")
        for relation_id in range(num_relations):
            task_queue.put(relation_id)

        self.start_processes()

        # parse the results added to the result queue
        progress_bar = tqdm(total=num_relations)
        finished = 0
        while True:
            relation_id, matrix = result_queue.get(block=True)
            matrices[relation_id] = matrix
            progress_bar.update(1)
            finished += 1
            if finished == num_relations:
                break

        # kill processes when we are done
        self.kill_processes()

        ltt.save_graph_npz(input_dir, matrices)
        return matrices

    def _load_entity_type_adjacency_matrix(self,
                                           input_dir: str,
                                           graph: Graph,
                                           entity_to_id: Dict[str, int],
                                           entity_type_to_id: Dict[str, int]) -> coo_matrix:
        """
        Creates an adjacency matrix for the "type of" relation, i.e., it contains all the type information of the graph.
        :param input_dir: path to the .n3 file of the RDF data
        :param graph: the graph object of the Knowledge Graph
        :param entity_to_id: the dictionary of the entities and their unique identifies
        :param entity_type_to_id: the dictionary of the entity types and their unique identifies
        :return: an adjacency matrix for the "type of" relation
        """
        try:
            entity_type_adjacency_matrix = ltt.load_types_npz(input_dir)
            return entity_type_adjacency_matrix
        except FileNotFoundError:
            pass

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
        entity_type_adjacency_matrix = coo_matrix((type_coo["vals"], (type_coo["rows"], type_coo["cols"])),
                                                  shape=(len(entity_to_id), len(entity_type_to_id)),
                                                  dtype=int)

        print(f"{type_assertions} type assertions loaded...")
        ltt.save_types_npz(input_dir, entity_type_adjacency_matrix)
        return entity_type_adjacency_matrix

    def _load_property_dag(self, input_dir: str, graph: Graph, property_to_id: Dict[str, int]) -> Dict[int, DAGNode]:
        """
        Creates a DAG of the object property hierarchy in the graph. This hierarchy is defined by the RDFS
        "sub-property of" property.
        :param input_dir: path to the .n3 file of the RDF data
        :param graph: the graph object of the Knowledge Graph
        :param property_to_id: the dictionary of the object properties and their unique identifies
        :return: dictionary of object property ids and the corresponding nodes in the DAG
        """
        try:
            property_dag = ltt.load_prop_hierarchy(input_dir)
            return property_dag
        except FileNotFoundError:
            pass

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

        ltt.save_prop_hierarchy(input_dir, property_dag)
        return property_dag

    def _load_entity_type_dag(self,
                              input_dir: str,
                              graph: Graph,
                              entity_type_to_id: Dict[str, int]) -> Dict[int, DAGNode]:
        """
        Creates a DAG of the entity type hierarchy in the graph. This hierarchy is defined by the RDFS "subclass of"
        property.
        :param input_dir: path to the .n3 file of the RDF data
        :param graph: the graph object of the Knowledge Graph
        :param entity_type_to_id: the dictionary of the entity types and their unique identifies
        :return: dictionary of entity type ids and the corresponding nodes in the DAG
        """
        try:
            entity_type_dag = ltt.load_type_hierarchy(input_dir)
            return entity_type_dag
        except FileNotFoundError:
            pass

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
                                      (subject in equivalent_classes and equivalent_classes[
                                          subject] in entity_type_to_id))
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

        ltt.save_type_hierarchy(input_dir, entity_type_dag)
        return entity_type_dag

    def _load_domains(self,
                      input_dir: str,
                      graph: Graph,
                      property_to_id: Dict[str, int],
                      entity_type_to_id: Dict[str, int]) -> Dict[int, int]:
        """
        Extracts the domains from the Knowledge Graph. These are defined by the RDFS "domain" relation. These relations are
        defined between an object property (subject) and an entity type (object) and express that the object property
        describes entities with that entity type. (for further reading: https://stackoverflow.com/a/9066520)
        :param input_dir: path to the .n3 file of the RDF data
        :param graph: the graph object of the Knowledge Graph
        :param property_to_id: the dictionary of the object properties and their unique identifies
        :param entity_type_to_id: the dictionary of the entity types and their unique identifies
        :return: a dictionary pointing from property ids to the entity type id of that properties' domain
        """
        try:
            domains = ltt.load_domains(input_dir)
            return domains
        except FileNotFoundError:
            pass

        # dictionary pointing from object property id to an entity type id
        domains = {}

        # add all domain triples for which the subject is an object property and the object is an entity type
        for subject, predicate, object in graph.triples((None, RDFS.domain, None)):
            if subject in property_to_id and object in entity_type_to_id:
                domains[property_to_id[subject]] = entity_type_to_id[object]

        ltt.save_domains(input_dir, domains)
        return domains

    def _load_ranges(self,
                     input_dir: str,
                     graph: Graph,
                     property_to_id: Dict[str, int],
                     entity_type_to_id: Dict[str, int]) -> Dict[int, int]:
        """
        Extracts the ranges from the Knowledge Graph. These are defined by the RDFS "range" relation. These relations
        are defined between an object property (subject) and an entity type (object) and express that the value of the
        object property is of the entity type. (for further reading: https://stackoverflow.com/a/9066520)
        :param input_dir: path to the .n3 file of the RDF data
        :param graph: the graph object of the Knowledge Graph
        :param property_to_id: the dictionary of the object properties and their unique identifies
        :param entity_type_to_id: the dictionary of the entity types and their unique identifies
        :return: a dictionary pointing from property ids to the entity type id of that properties' range
        """
        try:
            ranges = ltt.load_ranges(input_dir)
            return ranges
        except FileNotFoundError:
            pass

        # dictionary pointing from object property id to an entity type id
        ranges = {}

        # add all range triples for which the subject is an object property and the object is an entity type
        for subject, predicate, object in graph.triples((None, RDFS.range, None)):
            if subject in property_to_id and object in entity_type_to_id:
                ranges[property_to_id[subject]] = entity_type_to_id[object]

        ltt.save_ranges(input_dir, ranges)
        return ranges

    def load_tensor(self):
        rdf_format = self.input[self.input.rindex(".") + 1:]
        input_dir = self.input.replace(f".{rdf_format}", "")
        Path(input_dir).mkdir(exist_ok=True)

        # each of these methods loads a previously extracted version from a file if that exists
        graph = self._load_graph(input_dir, rdf_format)
        entity_type_to_id = self._load_entity_types(input_dir, graph)
        entity_to_id = self._load_entities(input_dir, graph, entity_type_to_id)
        property_to_id = self._load_properties(input_dir, graph, entity_to_id)
        property_adjaceny_matrices = self._load_property_adjacency_matrices(input_dir,
                                                                            graph,
                                                                            entity_to_id,
                                                                            property_to_id)
        entity_type_adjacency_matrix = self._load_entity_type_adjacency_matrix(input_dir,
                                                                               graph,
                                                                               entity_to_id,
                                                                               entity_type_to_id)
        # DAG of the entity type/class hierarchy
        entity_type_hierarchy_dag = self._load_entity_type_dag(input_dir, graph, entity_type_to_id)
        # DAG of the object property hierarchy
        object_property_hierarchy_dag = self._load_property_dag(input_dir, graph, property_to_id)

        num_entity_types = len(entity_type_to_id or {})
        num_entity_types_in_hierarchy = len(entity_type_hierarchy_dag or {})
        num_object_properties = len(entity_type_to_id or {})
        num_object_properties_in_hierarchy = len(object_property_hierarchy_dag or {})

        print(f"Loaded {num_entity_types} entity types, of which {num_entity_types_in_hierarchy} are contained in the "
              f"hierarchy graph...")
        print(f"Loaded {num_object_properties} object properties, of which {num_object_properties_in_hierarchy} are "
              f"contained in the hierarchy graph...")

        # explanation of domains and ranges: https://stackoverflow.com/a/9066520
        domains = self._load_domains(input_dir, graph, property_to_id, entity_type_to_id)
        print(f"Loaded {len(domains)} relation domains...")
        ranges = self._load_ranges(input_dir, graph, property_to_id, entity_type_to_id)
        print(f"Loaded {len(ranges)} relation ranges...")


class LearnAdjacencyMatrixProcess(Process):
    def __init__(self,
                 task_queue: Queue,
                 result_queue: Queue,
                 graph: Graph,
                 entity_to_id: Dict[str, int],
                 property_to_id: Dict[str, int]):
        super(Process, self).__init__()
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.graph = graph
        self.entity_to_id = entity_to_id
        self.property_to_id = property_to_id
        self.property_id_to_uri = {property_id: uri for uri, property_id in property_to_id.items()}

    def run(self):
        while True:
            try:
                relation_id = self.task_queue.get()
                self.learn_matrix(relation_id)
            except Empty:
                break

    def learn_matrix(self, relation_id: int):
        matrix_data = {"rows": [], "cols": [], "vals": []}
        predicate = self.property_id_to_uri[relation_id]

        for subject, predicate, object in self.graph.triples((None, predicate, None)):
            if subject in self.entity_to_id and object in self.entity_to_id:
                # unique ids of the subject, predicate and object
                subject_id = self.entity_to_id[subject]
                object_id = self.entity_to_id[object]

                # triple that is used to create adjacency matrix
                matrix_data["rows"].append(subject_id)
                matrix_data["cols"].append(object_id)
                matrix_data["vals"].append(1)

        # create sparse matrix for this object property by aggregating (row, column, value) tuples
        # these tuples have the values: (subject_id, object_id, 1)
        matrix = coo_matrix((matrix_data["vals"], (matrix_data["rows"], matrix_data["cols"])),
                            shape=[len(self.entity_to_id), len(self.entity_to_id)])
        self.result_queue.put([relation_id, matrix])
