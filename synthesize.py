import pickle
from argparse import ArgumentParser, Namespace
from typing import Dict

from rdflib import Graph, URIRef

from kbgen.kb_models import KBModelM1
from kbgen.kb_models import KBModelM4
from util import dump_tsv
from util_models import URIRelation, URIType


def cli_args() -> Namespace:
    parser = ArgumentParser(description="Synthesizes a dataset from a given model and size")
    parser.add_argument("input", type=str, default=None, help="path to kb model")
    parser.add_argument("output", type=str, default=None, help="path to the synthetic rdf file")
    parser.add_argument("-s", "--size", type=float, default=1,
                        help="sample size as number of original facts divided by step")
    parser.add_argument("-ne", "--nentities", type=int, default=None, help="number of entities")
    parser.add_argument("-nf", "--nfacts", type=int, default=None, help="number of facts")
    parser.add_argument("-d", "--debug", dest="debug", action="store_true", help="debug mode")
    parser.set_defaults(debug=False)
    return parser.parse_args()


def replace_id_with_name(graph: Graph, model: KBModelM1) -> Graph:
    """
    Replace the numeric ids in the URIs of the entity types and relations with the original URIs from the knowledge base
    on which the models were trained.
    :param graph: the synthesized graph
    :param model: the model that was used to synthesize the graph
    :return: a copy of the synthesized graph, in which the relations and entity types have their original URIs
    """
    graph_with_names = Graph()
    relation_id_to_uri: Dict[int, URIRef] = {relation_id: relation_uri
                                             for relation_uri, relation_id in model.relation_to_id.items()}

    entity_type_id_to_uri: Dict[int, URIRef] = {type_id: type_uri
                                                for type_uri, type_id in  model.entity_type_to_id.items()}

    def replace_name(rdf_entity: URIRef) -> URIRef:
        """
        Replace the name of URIRelations and URITypes while keeping the names of URIEntities.
        :param rdf_entity: the synthesized URI that is resolved to its original name
        :return: the resolved URI if the original URI was an URIRelation or an URIType, otherwise the synthesized URI
        """
        name = rdf_entity

        if str(rdf_entity).startswith(URIRelation.prefix):
            name = relation_id_to_uri[URIRelation.extract_id(rdf_entity).id]
        elif str(rdf_entity).startswith(URIType.prefix):
            name = entity_type_id_to_uri[URIType.extract_id(rdf_entity).id]

        return URIRef(name)

    for subject_uri, predicate_uri, object_uri in graph.triples((None, None, None)):
        subject_with_name = replace_name(subject_uri)
        relation_with_name = replace_name(predicate_uri)
        object_with_name = replace_name(object_uri)
        graph_with_names.add((subject_with_name, relation_with_name, object_with_name))

    return graph_with_names


def main():
    args = cli_args()
    print(args)

    # deserialize model
    model = pickle.load(open(args.input, "rb"))

    # synthesize graph using the model
    graph = model.synthesize(size=args.size, number_of_entities=args.nentities, number_of_edges=args.nfacts, debug=args.debug)

    # replace the numbered rdf relations with the proper names from the original knowledge base
    graph = replace_id_with_name(graph, model)

    if isinstance(model, KBModelM4):
        with open(f"{args.output}-oracle.json", "w") as oracle_file:
            model.oracle.to_json(oracle_file)
        with open(f"{args.output}-oracle.bin", "wb") as oracle_file:
            pickle.dump(model.oracle, oracle_file, protocol=pickle.HIGHEST_PROTOCOL)

    formats = ["n3", "ttl"]

    # serialize the generated graph and write it to a .tsv file
    for rdf_format in formats:
        graph.serialize(f"{args.output}.{rdf_format}", format=rdf_format)
    dump_tsv(graph, f"{args.output}.tsv")


if __name__ == '__main__':
    main()
