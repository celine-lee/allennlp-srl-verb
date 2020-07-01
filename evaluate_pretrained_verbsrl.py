import os
import shutil
import tempfile
import tarfile
import atexit
import argparse

from allennlp.models.archival import load_archive, Archive
from allennlp.common.util import dump_metrics, prepare_environment
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data import DataLoader
from allennlp.common.file_utils import cached_path
from allennlp.training.util import evaluate
from allennlp.common.params import Params

from allennlp_models.syntax.srl.srl_model import SemanticRoleLabeler

desc = "Evaluation script for pre-trained Bert Base SRL from 2020.03.24."
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--archive_file', '-a', type=str, default="https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz", help="path to tar.gz archive file")
parser.add_argument('--evaluation_data_path', '-e', type=str, default="/shared/celinel/LDC2013T19/conll-formatted-ontonotes-5.0/conll-formatted-ontonotes-5.0/data/test", help="path to evaluation dataset")
parser.add_argument('--output_file', '-o', type=str, default="pretrained_verbsrl_evaluation.txt", help="path to output file to put computed metrics")

args = parser.parse_args()

def _cleanup_archive_dir(path: str):
    if os.path.exists(path):
        shutil.rmtree(path)

resolved_archive_file = cached_path(args.archive_file)
# Create temporary directory and extract archive file.
if os.path.isdir(resolved_archive_file):
    serialization_dir = resolved_archive_file
else:
    tempdir = tempfile.mkdtemp()
    with tarfile.open(resolved_archive_file, "r:gz") as archive:
        archive.extractall(tempdir)
    atexit.register(_cleanup_archive_dir, tempdir)
    serialization_dir = tempdir

config = Params.from_file(os.path.join(serialization_dir, "config.json"), "")
model = SemanticRoleLabeler.from_archive(args.archive_file)
archive = Archive(model=model, config=config)

prepare_environment(config)
model.eval()
validation_dataset_reader_params = config.pop("validation_dataset_reader", None)
if validation_dataset_reader_params is not None:
    dataset_reader = DatasetReader.from_params(validation_dataset_reader_params)
else:
    dataset_reader = DatasetReader.from_params(config.pop("dataset_reader"))
instances = dataset_reader.read(args.evaluation_data_path)
instances.index_with(model.vocab)
data_loader_params = config.pop("validation_data_loader", None)
if data_loader_params is None:
    data_loader_params = config.pop("data_loader")
    
data_loader = DataLoader.from_params(dataset=instances, params=data_loader_params)

metrics = evaluate(model, data_loader, -1, "")
dump_metrics(args.output_file, metrics)
