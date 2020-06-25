from allennlp.predictors.predictor import Predictor, JsonDict

def create_sentence_view(tokens) -> JsonDict:
    sentence_view = {"viewName": "SENTENCE"}
    constituents = []
    sentence_end_positions = [i+1 for i,x in enumerate(tokens) if x=="."]
    sentence_end_positions = [0] + sentence_end_positions
    constituents = [{"label": "SENTENCE", "score": 1.0, "start": sentence_end_positions[idx-1], "end": sentence_end_positions[idx]} for idx in range(1, len(sentence_end_positions))]
    view_data = [{"viewType": "", "viewName": "SENTENCE", "generator": "UserSpecified", "score": 1.0, "constituents": constituents}]
    sentence_view["viewData"] = view_data
    return sentence_view

def create_tokens_view(tokens) -> JsonDict:
    token_view = {"viewName": "TOKENS"}
    constituents = []
    for idx, token in enumerate(tokens):
        constituents.append({"label": token, "score": 1.0, "start": idx, "end": idx+1})
    view_data = [{"viewType": "", "viewName": "TOKENS", "generator": "UserSpecified", "score": 1.0, "constituents": constituents}]
    token_view["viewData"] = view_data
    return token_view
