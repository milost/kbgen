import pickle
from argparse import ArgumentParser, Namespace
from util import dump_tsv


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


def main():
    args = cli_args()
    print(args)

    # deserialize model
    model = pickle.load(open(args.input, "rb"))

    # synthesize graph using the model
    graph = model.synthesize(size=args.size, ne=args.nentities, nf=args.nfacts, debug=args.debug)

    # serialize the generated graph and write it to a .tsv file
    rdf_format = args.output[args.output.rindex(".") + 1:]
    graph.serialize(open(args.output, "wb"), format=rdf_format)
    dump_tsv(graph, args.output.replace("." + rdf_format, ".tsv"))


if __name__ == '__main__':
    main()
