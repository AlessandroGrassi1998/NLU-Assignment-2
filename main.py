import nltk.tag.hmm as hmm
import spacy
from conll import read_corpus_conll
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")
class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

spacyToConllMap = {
    # https://github.com/explosion/spaCy/blob/master/spacy/glossary.py
    "PERSON": "PER",
    "NORP": "MISC",
    "FACILITY": "ORG",
    "FAC": "MISC",
    "ORG": "ORG",
    "GPE": "LOC",
    "LOC": "LOC",
    "PRODUCT": "MISC",
    "EVENT": "MISC",
    "WORK_OF_ART": "MISC",
    "LAW": "MISC",
    "LANGUAGE": "MISC",
    "DATE": "MISC",
    "TIME": "MISC",
    "PERCENT": "MISC",
    "MONEY": "MISC",
    "QUANTITY": "MISC",
    "ORDINAL": "MISC",
    "CARDINAL": "MISC",
    "PER": "PER",
    "MISC": "MISC",
    "EVT": "MISC",
    "PROD": "MISC",
    "DRV": "MISC",
    "GPE_LOC": "LOC",
    "GPE_ORG": "ORG",
    "": ""
}


def train_hmm_model(output_desired):
    outputs = ["part_of_speech_tag", "chunck_tag", "named_entity_tag"]
    outputIndex = outputs.index(output_desired) + 1
    if outputIndex == 0:
        return False
    train = [[]]
    sentence_index = 0
    with open('./conll2003/final_training.txt') as f:
        for line in f:
            if line == "\n":
                sentence_index += 1
                train.append([])
            else:
                line_splitted = line.split(" ")
                used_tuple = (line_splitted[0],
                              line_splitted[outputIndex].replace('\n', ''))
                train[sentence_index].append(used_tuple)

    hmm_model = hmm.HiddenMarkovModelTrainer()

    hmm_ner = hmm_model.train(train)
    return hmm_ner


def grouping_of_entities(conll_data):
    total_tokens = 0
    correctly_classified = 0
    for sentence in conll_data:
        token_array = []
        part_of_speech_tag_array = []
        chunck_tag_array = []
        named_entity_tag_array = []
        for element in sentence:
            token = element[0].split()[0]
            part_of_speech_tag = element[0].split()[1]
            chunck_tag = element[0].split()[2]
            named_entity_tag = element[0].split()[3]
            token_array.append(token)
            part_of_speech_tag_array.append(part_of_speech_tag)
            chunck_tag_array.append(chunck_tag)
            named_entity_tag_array.append(named_entity_tag)
        #doc = Doc(nlp.vocab, words=token_array)
        doc = nlp(" ".join(token_array))
        token_index = 0
        for token in doc:
            total_tokens += 1
            ent_type_converted_to_conll = token.ent_iob_
            if(spacyToConllMap[token.ent_type_] != ""):
                ent_type_converted_to_conll += "-" + spacyToConllMap[token.ent_type_]
            if(ent_type_converted_to_conll == named_entity_tag_array[token_index]):
                correctly_classified += 1
            token_index += 1
    print(total_tokens, correctly_classified)
    return correctly_classified / total_tokens


conll_data = read_corpus_conll("./conll2003/test.txt")
print(grouping_of_entities(conll_data))

"""doc = Doc(nlp.vocab, words=["Hi",",", "how", "are", "you","?"])
doc = nlp(" ".join(["Hi",",", "how", "are", "you","?"]))
for token in doc:
    print(token.ent_type_, token.ent_iob_)"""
