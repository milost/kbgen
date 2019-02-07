import pickle
from argparse import ArgumentParser, Namespace
from typing import Tuple, List

from kbgen.kb_models import KBModelM1, KBModelM2, KBModelM3, KBModelM4, KBModelEMi
from kbgen.rules import RuleSet


def cli_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input", type=str, default=None, help="path to the directory containing npy and npz files")
    parser.add_argument("-m", "--model", type=str, default="M1",
                        help="choice of model [M1, M2, M2, e] (e requires -sm)")
    parser.add_argument("-sm", "--source-kb-models", type=str, nargs="+", default=["M1", "M2", "M3"],
                        help="source model with entity selection bias [M1, M2, M3]")
    parser.add_argument("-r", "--rules-path", type=str, default=None, help="path to txt file with Amie horn rules")
    return parser.parse_args()


def build_m1_model(tensor_files_dir: str, output_name: str) -> Tuple[KBModelM1, str]:
    """
    Build an M1 model from a Knowledge Graph.
    :param tensor_files_dir: path to directory containing the numpy serialized Knowledge Graph data structures
    :param output_name: name that is used for the output file name
    :return: tuple of the generated M1 model and the output file name
    """
    model = KBModelM1.generate_from_tensor(tensor_files_dir)
    return model, f"{output_name}-M1.pkl"


def build_m2_model(tensor_files_dir: str, output_name: str) -> Tuple[KBModelM2, str]:
    """
    Build an M2 model from a Knowledge Graph and a previously built and serialized M1 model. The M1 model must have been
    serialized with the same output_name passed to this method.
    :param tensor_files_dir: path to directory containing the numpy serialized Knowledge Graph data structures
    :param output_name: name that is used for the output file name
    :return: tuple of the generated M2 model and the output file name
    """
    m1_model_path = f"{output_name}/{output_name}-M1.pkl"
    m1_model = pickle.load(open(m1_model_path, "rb"))
    assert isinstance(m1_model, KBModelM1)

    model = KBModelM2.generate_from_tensor_and_model(m1_model, tensor_files_dir)
    return model, f"{output_name}-M2.pkl"


def build_m3_model(rule_file: str, output_name: str) -> Tuple[KBModelM3, str]:
    """
    Build an M3 model from a previously built and serialized M2 model and AMIE rules. The M2 model must have been
    serialized with the same output_name passed to this method.
    :param rule_file: path to the file containing the AMIE rules
    :param output_name: name that is used for the output file name
    :return: tuple of the generated M3 model and the output file name
    """
    m2_model_path = f"{output_name}/{output_name}-M2.pkl"
    m2_model = pickle.load(open(m2_model_path, "rb"))
    assert isinstance(m2_model, KBModelM2)

    assert isinstance(m2_model, KBModelM2)
    # dictionary pointing from the relations (entities) to their ids
    relation_to_id = m2_model.relation_to_id
    rules = RuleSet.parse_amie(rule_file, relation_to_id)

    model = KBModelM3(m2_model, rules)
    return model, f"{output_name}-M3.pkl"


def build_m4_model(output_name: str) -> Tuple[KBModelM2, str]:
    """
    Build an M4 model from a previously built and serialized M3 model. The M4 model must have been
    serialized with the same output_name passed to this method.
    :param output_name: name that is used for the output file name
    :return: tuple of the generated M4 model and the output file name
    """
    m3_model_path = f"{output_name}/{output_name}-M3.pkl"
    m3_model = pickle.load(open(m3_model_path, "rb"))
    assert isinstance(m3_model, KBModelM3)

    model = KBModelM4(m3_model)
    return model, f"{output_name}-M4.pkl"


def build_e_models(tensor_files_dir: str, kb_models: List[str], output_name: str) -> Tuple[List[KBModelM1], List[str]]:
    """
    Builds e-models for every model type passed. These are generated from the Knowledge Graph and the subject and
    object distributions from the previous e-model, i.e. an eM2 model uses the distributions from an eM1 model.
    :param tensor_files_dir: path to directory containing the numpy serialized Knowledge Graph data structures
    :param kb_models: list of the model types for which e-models are generated (M1, M2, M3)
    :param output_name: name that is used for the output file name
    :return: tuple of a list of generated eMi models and a list of their output file names
    """
    models = []
    models_output = []
    dist_subjects, dist_objects = None, None
    for source_model_name in kb_models:
        m1_model_path = f"{output_name}-{source_model_name}.pkl"
        m1_model = pickle.load(open(m1_model_path, "rb"))
        assert isinstance(m1_model, KBModelM1)

        if dist_subjects is None and dist_objects is None:
            model = KBModelEMi.generate_from_tensor_and_model(m1_model, tensor_files_dir)
            dist_subjects = model.dist_subjects
            dist_objects = model.dist_objects
        else:
            model = KBModelEMi(m1_model, dist_subjects, dist_objects)
        models.append(model)
        models_output.append(f"{output_name}-e{source_model_name}.pkl")

    return models, models_output


def main():
    args = cli_args()
    print(args)
    base = args.input.replace(".npz", "")
    print(f"Learning {args.model} model...")

    models = []
    models_output = []

    if args.model == "M1":
        model, output_name = build_m1_model(args.input, base)
        models.append(model)
        models_output.append(output_name)

    if args.model == "M2":
        model, output_name = build_m2_model(args.input, base)
        models.append(model)
        models_output.append(output_name)

    if args.model == "M3":
        model, output_name = build_m3_model(args.rules_path, base)
        models.append(model)
        models_output.append(output_name)

    if args.model == "M4":
        model, output_name = build_m4_model(base)
        models.append(model)
        models_output.append(output_name)

    if args.model == "e":
        e_models, output_names = build_e_models(args.input, args.source_kb_models, base)
        models.extend(e_models)
        models_output.extend(output_names)

    if models and models_output:
        for model, model_output in zip(models, models_output):
            output_file = f"{args.input}/{model_output}"
            print(f"Saving model to {output_file}...")
            pickle.dump(model, open(output_file, "wb"))


if __name__ == '__main__':
    main()
