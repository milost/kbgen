import pickle
from argparse import ArgumentParser, Namespace
from typing import Dict

from rdflib import Graph, URIRef

from kbgen import KBModelM3
from kbgen.kb_models import KBModelM1
from kbgen.kb_models import KBModelM4
from kbgen.kb_models.multiprocessing.m3_synthesization import M3Synthesization
from kbgen.util import dump_tsv
from kbgen.util_models import URIRelation, URIType


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Synthesizes a dataset from a given model and size")
    parser.add_argument("input", type=str, default=None, help="path to kb model")
    parser.add_argument("output", type=str, default=None, help="path to the synthetic rdf file")
    parser.add_argument("-s", "--size", type=float, default=1,
                        help="sample size as number of original facts divided by step")
    parser.add_argument("-ne", "--nentities", type=int, default=None, help="number of entities")
    parser.add_argument("-nf", "--nfacts", type=int, default=None, help="number of facts")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="debug mode")
    parser.add_argument("-p", "--num-processes", type=int, default=0, help="the number of processes to use")
    parser.set_defaults(debug=False)
    return parser.parse_args()


class URINameReplacer(object):
    def __init__(self, model: KBModelM1):
        self.relation_id_to_uri: Dict[int, URIRef] = {relation_id: relation_uri
                                                      for relation_uri, relation_id in model.relation_to_id.items()}

        self.entity_type_id_to_uri: Dict[int, URIRef] = {type_id: type_uri
                                                         for type_uri, type_id in model.entity_type_to_id.items()}

    def replace_name(self, rdf_entity: URIRef) -> URIRef:
        """
        Replace the name of URIRelations and URITypes while keeping the names of URIEntities.
        :param rdf_entity: the synthesized URI that is resolved to its original name
        :return: the resolved URI if the original URI was an URIRelation or an URIType, otherwise the synthesized URI
        """
        name = rdf_entity

        if str(rdf_entity).startswith(URIRelation.prefix):
            name = self.relation_id_to_uri[URIRelation.extract_id(rdf_entity).id]
        elif str(rdf_entity).startswith(URIType.prefix):
            name = self.entity_type_id_to_uri[URIType.extract_id(rdf_entity).id]

        return URIRef(name)


def replace_id_with_name(graph: Graph, model: KBModelM1) -> Graph:
    """
    Replace the numeric ids in the URIs of the entity types and relations with the original URIs from the knowledge base
    on which the models were trained.
    :param graph: the synthesized graph
    :param model: the model that was used to synthesize the graph
    :return: a copy of the synthesized graph, in which the relations and entity types have their original URIs
    """
    graph_with_names = Graph()
    name_replacer = URINameReplacer(model)

    for subject_uri, predicate_uri, object_uri in graph.triples((None, None, None)):
        subject_with_name = name_replacer.replace_name(subject_uri)
        relation_with_name = name_replacer.replace_name(predicate_uri)
        object_with_name = name_replacer.replace_name(object_uri)
        graph_with_names.add((subject_with_name, relation_with_name, object_with_name))

    return graph_with_names


def main():
    args = cli_args()
    print(args)

    # deserialize model
    model = pickle.load(open(args.input, "rb"))

    # synthesize graph using the model
    if args.num_processes and isinstance(model, KBModelM3):
        synthesizer = M3Synthesization(model, args.num_processes)
        graph = synthesizer.synthesize(size=args.size)
    else:
        graph = model.synthesize(size=args.size, number_of_entities=args.nentities, number_of_edges=args.nfacts, debug=args.debug)
    print()
    print(f"Synthesized graph contains {len(graph)} triples")

    print("Replacing ids with uris...")
    # replace the numbered rdf relations with the proper names from the original knowledge base
    graph = replace_id_with_name(graph, model)

    if isinstance(model, KBModelM4):
        print("Saving oracle...")
        with open(f"{args.output}-oracle.json", "w") as oracle_file:
            model.oracle.to_json(oracle_file)

    formats = ["n3", "ttl"]

    print("Saving synthesized graph...")
    # serialize the generated graph and write it to a .tsv file
    for rdf_format in formats:
        graph.serialize(f"{args.output}.{rdf_format}", format=rdf_format)
    dump_tsv(graph, f"{args.output}.tsv")


if __name__ == '__main__':
    main()
