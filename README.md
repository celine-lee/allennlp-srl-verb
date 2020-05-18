To set up the environment and run the model, follow the following workflow:

Create the virtual environment and set up installs:
```
python3 -m venv verb_srl_venv
cd verb_srl_venv
source bin/activate
pip install allennlp==1.0.0rc3
pip install allennlp-models
```
Check that the `lib/python3.6/site-packages/allennlp_models/syntax/srl/` folder has `srl-eval.pl`. If it does not, wget it from [here](https://raw.githubusercontent.com/allenai/allennlp-models/master/allennlp_models/syntax/srl/srl-eval.pl) and put it in that folder.

The GPUs on the CCG machines are CUDA version 10.1, so we set Pytorch back to version 1.4:
```
pip install torch===1.4.0 torchvision===0.5.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Then we set up the paths and model files, which are at `bert_base_srl_24.jsonnet` and `set_paths.sh`. Note that these paths are set up as of 5/18/20 and only work on the CCG machines, and the paths may have changed since then.

To train:
```
. ./set_paths.sh
allennlp train bert_base_srl_24.jsonnet -s srl-bert-test --include_package allennlp_models
```

To evaluate:
```
allennlp evaluate srl-bert-test/model.tar.gz /shared/celinel/LDC2013T19/conll-formatted-ontonotes-5.0/conll-formatted-ontonotes-5.0/data/test
```

To predict:
```
allennlp predict srl-bert-model/model.tar.gz input.txt --output-file output.txt
```
The input and output files for the predict call are in JSON format. Inputs should be formatted such that sentences are separated into individual sentences, rather than one large paragraph. 
