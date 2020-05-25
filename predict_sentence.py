from typing import List, Iterator, Optional
import argparse
import sys
import json

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
# from allennlp.common.util import lazy_groups_of
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

from allennlp_models.syntax.srl.srl_predictor import SemanticRoleLabelerPredictor

desc = "Run predictor on a single sentence."

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("archive_file", type=str, help="the archived model to make predictions with")
parser.add_argument("input_sentence", type=str, help="the sentence to predict on")
parser.add_argument("--cuda_device", type=int, default=-1, help="id of GPU to use (if any)")
parser.add_argument("--output_file", type=str, default="output.txt", help="path to output file")

args = parser.parse_args()


def _get_predictor(args) -> SemanticRoleLabelerPredictor:
    check_for_gpu(args.cuda_device)
    archive = load_archive(
            args.archive_file,
            cuda_device=args.cuda_device,
            )
    return SemanticRoleLabelerPredictor.from_archive(archive)


class _PredictManager:
    def __init__(
        self,
        predictor: SemanticRoleLabelerPredictor,
        input_sentence: str,
        output_file: Optional[str],
    ) -> None:

        self._predictor = predictor
        self._input_sentence = input_sentence
        if output_file is not None:
            self._output_file = open(output_file, "w")
        else:
            self._output_file = None


    def _print_to_file(
        self, prediction: str
    ) -> None:
        if self._output_file is not None:
            self._output_file.write(prediction)
            self._output_file.close()

    def run(self) -> None:
        jsonified = self._predictor.load_line(self._input_sentence)
        
        result = self._predictor.predict_json(jsonified)
        self._print_to_file(json.dumps(result))


predictor = _get_predictor(args)

manager = _PredictManager(
    predictor,
    args.input_sentence,
    args.output_file,
    )    

manager.run()
