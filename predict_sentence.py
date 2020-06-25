from typing import List, Iterator, Optional
import argparse
import sys
import json

from overrides import overrides

from allennlp.commands.subcommand import Subcommand
from allennlp.common.checks import check_for_gpu, ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.common.util import sanitize
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor, JsonDict
from allennlp.data import Instance

from allennlp_models.syntax.srl.srl_predictor import SemanticRoleLabelerPredictor

import predict_utils

desc = "Run predictor on a single sentence, and write to a TextAnnotation view."

parser = argparse.ArgumentParser(description=desc)
parser.add_argument("archive_file", type=str, help="the archived model to make predictions with")
parser.add_argument("input_sentence", type=str, help="the sentence to predict on")
parser.add_argument("--cuda_device", type=int, default=-1, help="id of GPU to use (if any)")
parser.add_argument("--output_file", type=str, default="textannotation.txt", help="path to output TextAnnotation view file")

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

    def create_token_char_offsets(
        self, text: str
    ) -> JsonDict:
        char_offsets = []
        last_space_idx = -1
        while True:
            space_idx = text.find(' ', last_space_idx+1)
            if space_idx == -1:
                space_idx = len(text)
            entry = {"form": text[last_space_idx+1:space_idx], "startCharOffset": last_space_idx+1, "endCharOffset": space_idx}
            char_offsets.append(entry)
            last_space_idx = space_idx
            if last_space_idx == len(text):
                break
        return char_offsets

    def create_text_annotation(
        self, srl_output: JsonDict
    ) -> JsonDict:
        ta = {"corpusId": "", "id": ""}
        tokens = srl_output.pop("words")
        text = " ".join(tokens)
        ta["text"] = text
        ta["tokens"] = tokens
        ta["tokenOffsets"] = self.create_token_char_offsets(text)
        sentence_end_positions = [i+1 for i, x in enumerate(tokens) if x == "."]
        sentences = {"generator": "", "score": 1.0, "sentenceEndPositions": sentence_end_positions}
        ta["sentences"] = sentences
        
        # Create views.
        views = []
        views.append(predict_utils.create_sentence_view(tokens))
        views.append(predict_utils.create_tokens_view(tokens))
        views.append(self.create_srl_verb_view(srl_output.pop("verbs")))
        ta["views"] = views
        return sanitize(ta)


    def create_srl_verb_view(
        self, verb_srl_frames
    ) -> JsonDict:
        srl_verb_view = {"viewName": "SRL_VERB"}
        constituents = []
        relations = []
        for frame in verb_srl_frames:
            predicate = frame.pop("verb")
            description = frame.pop("description")
            tags = frame.pop("tags")
            predicate_index = tags.index("B-V")
            properties = {"SenseNumber": "NA", "predicate": predicate}
            constituent = {"label": "Predicate", "score": 1.0, "start": predicate_index, "end": predicate_index+1, "properties": properties}
            predicate_constituent = len(constituents)
            constituents.append(constituent)
            active_tag = ""
            active_tag_start_idx = -1
            for tag_idx, tag in enumerate(tags):
                if tag in {"O", "B-V"}:
                    if active_tag != "":
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent, "targetConstituent": len(constituents)}
                        relations.append(relation)
                        constituents.append(constituent)
                        active_tag = ""
                        active_tag_start_idx = -1
                    continue
                if tag[2:] == active_tag:
                    continue
                else:
                    if active_tag != "":
                        constituent = {"label": active_tag, "score": 1.0, "start": active_tag_start_idx, "end": tag_idx}
                        relation = {"relationName": active_tag, "srcConstituent": predicate_constituent, "targetConstituent": len(constituents)}
                        relations.append(relation)
                        constituents.append(constituent)
                    active_tag = tag[2:]
                    active_tag_start_idx = tag_idx
                    
        verb_view_data = [{"viewType": "", "viewName": "SRL_VERB", "generator": "allennlp-verbsrl", "score": 1.0, "constituents": constituents, "relations": relations}]
        srl_verb_view["viewData"] = verb_view_data
        return srl_verb_view


    def _print_to_file(
        self, prediction: str
    ) -> None:
        if self._output_file is not None:
            self._output_file.write(prediction)
            self._output_file.close()
        else:
            print("No output file was specified, so writing it to STDOUT.")
            print(prediction)

    def run(self) -> None:
        result = self._predictor.predict(self._input_sentence)
        ta = self.create_text_annotation(result)
        self._print_to_file(json.dumps(ta, indent=4))


predictor = _get_predictor(args)

manager = _PredictManager(
    predictor,
    args.input_sentence,
    args.output_file,
    )    

manager.run()
