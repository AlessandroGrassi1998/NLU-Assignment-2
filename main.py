import spacy
from conll import read_corpus_conll
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")
nlp_standard = spacy.load("en_core_web_sm")


class WhitespaceTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split(" ")
        return Doc(self.vocab, words=words)


nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

spacy_to_conll_map = {
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




# task 0.1
def token_level_performance(conll_data):
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
        doc = nlp(" ".join(token_array))
        token_index = 0
        for token in doc:
            total_tokens += 1
            ent_type_converted_to_conll = token.ent_iob_
            if(spacy_to_conll_map[token.ent_type_] != ""):
                ent_type_converted_to_conll += "-" + \
                    spacy_to_conll_map[token.ent_type_]
            if(ent_type_converted_to_conll == named_entity_tag_array[token_index]):
                correctly_classified += 1
            token_index += 1
    print(total_tokens, correctly_classified)
    return correctly_classified / total_tokens



# task 0.2
def chunk_level_performance(conll_data):
    effective_class_counts = {
        "MISC": 0,
        "ORG": 0,
        "PER": 0,
        "LOC": 0,
    }
    recognized_class_counts = {
        "MISC": 0,
        "ORG": 0,
        "PER": 0,
        "LOC": 0,
    }
    counter = 0
    chunk_counter = 0
    recognized_chunk_counter = 0
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
        doc = nlp(" ".join(token_array))
        actual_chunks, actual_chunk_label_array, non_empty_chunks, current_effective_class_counts = get_chunks(
            token_array, named_entity_tag_array)
        effective_class_counts["MISC"] += current_effective_class_counts["MISC"]
        effective_class_counts["ORG"] += current_effective_class_counts["ORG"]
        effective_class_counts["PER"] += current_effective_class_counts["PER"]
        effective_class_counts["LOC"] += current_effective_class_counts["LOC"]
        chunk_counter += non_empty_chunks
        for ent in doc.ents:
            if ent.text in actual_chunks:
                recognized_chunk_counter += 1
                actual_chunk_index = actual_chunks.index(ent.text)
                if spacy_to_conll_map[ent.label_] == actual_chunk_label_array[actual_chunk_index]:
                    recognized_class_counts[actual_chunk_label_array[actual_chunk_index]] += 1
    return effective_class_counts, recognized_class_counts, chunk_counter, recognized_chunk_counter


def get_chunks(token_array, named_entity_tag_array):
    effective_class_counts = {
        "MISC": 0,
        "ORG": 0,
        "PER": 0,
        "LOC": 0,
    }
    chunk_array = []
    chunk_label_array = []
    current_chunk = ""
    last_iob = ""
    total_chunks = 0
    for token_index, token in enumerate(token_array):
        if named_entity_tag_array[token_index][0] == "B":
            effective_class_counts[named_entity_tag_array[token_index][2:]] += 1
            total_chunks += 1
            chunk_label_array.append(named_entity_tag_array[token_index][2:])
            if current_chunk != "":
                chunk_array.append(current_chunk)
            current_chunk = token
        if named_entity_tag_array[token_index][0] == "I":
            current_chunk += " " + token
        last_iob = named_entity_tag_array[token_index][0]
    chunk_array.append(current_chunk)
    return chunk_array, chunk_label_array, total_chunks, effective_class_counts,


# task 1
def grouping_entities(sentence):
    doc = nlp_standard(sentence)
    ent_labels = [ent.label_ for ent in doc.ents]
    token_index_of_ents = [ent[0].idx for ent in doc.ents]
    named_entities_NP = []
    for chunk_index, chunk in enumerate(doc.noun_chunks):
        named_entities_NP.append([])
        for ent_index, ent in enumerate(chunk.ents):
            named_entities_NP[chunk_index].append(ent.label_)
            token_index_of_ents[token_index_of_ents.index(ent[0].idx)] = -1

    for token_of_ent_index, token_of_ent in enumerate(token_index_of_ents):
        if(token_of_ent != -1):
            named_entities_NP.append([ent_labels[token_of_ent_index]])

    map_count = create_map(named_entities_NP)
    return map_count


def create_map(named_entities_NP):
    grouped_label_map = {}
    for named_entitity_NP in named_entities_NP:
        label_of_group = ""
        if len(named_entitity_NP) > 1:
            for label_index, label in enumerate(named_entitity_NP):
                label_of_group += label
                if label_index != (len(named_entitity_NP) - 1):
                    label_of_group += "-"
        else:
            label_of_group = named_entitity_NP[0]
        if label_of_group in grouped_label_map:
            grouped_label_map[label_of_group] += 1
        else:
            grouped_label_map[label_of_group] = 1
    return grouped_label_map


# task 2
def expand_entity_with_compound(sentence):
    doc = nlp_standard(sentence)
    ents = doc.ents
    idx_to_tokenindex_map = {}
    token_ent_pair_array= []
    token_to_change = []
    for token_index, token in enumerate(doc):
        idx_to_tokenindex_map[token.idx] = token_index
        if token.dep_ != "compound":
            is_first = True
            is_first_child = True
            for child in token.children:
                if child.dep_ == "compound" and child.idx < token.idx:
                    is_first = False
                    if is_first_child:
                        token_ent_pair_array[idx_to_tokenindex_map[child.idx]] = (child.text, "B-" + token.ent_type_)
                    else:
                        token_ent_pair_array[idx_to_tokenindex_map[child.idx]] = (child.text, "I-" + token.ent_type_)
                    is_first_child = False
            if is_first:
                ent_iob_ = "O"
                if token.ent_iob_ != "O":
                    ent_iob_ = token.ent_iob_ + "-"
                token_ent_pair_array.append((token.text, ent_iob_ + token.ent_type_))
            else:
                token_ent_pair_array.append((token.text, "I-" + token.ent_type_))
        else:
            if token.head.idx < token.idx:
                head_ent_type = token_ent_pair_array[idx_to_tokenindex_map[token.head.idx]][1][2:]
                token_ent_pair_array.append((token.text, "I-" + head_ent_type))
            else:
                token_ent_pair_array.append(())
    
    return token_ent_pair_array

print()
print("Task 0.1")
conll_data = read_corpus_conll("./conll2003/final_training.txt")
print("#_token_classifyied_correctly/#_of_tokens:", token_level_performance(conll_data))

print()
print("Task 0.2")
effective_class_counts, recognized_class_counts, chunk_counter, recognized_chunk_counter = chunk_level_performance(conll_data)
print("total chunk rateo:", (recognized_chunk_counter/chunk_counter))
print("class MISC rateo:", recognized_class_counts["MISC"] / effective_class_counts["MISC"])
print("class ORG rateo:", recognized_class_counts["ORG"] / effective_class_counts["ORG"])
print("class PER rateo:", recognized_class_counts["PER"] / effective_class_counts["PER"])
print("class LOC rateo:", recognized_class_counts["LOC"] / effective_class_counts["LOC"])


test_sentence = "Apple's Steve Jobs died in 2011 in Palo Alto, California."
print()
print("Task 1")
print(grouping_entities(test_sentence))

print()
print("Task 2")
print(expand_entity_with_compound(test_sentence))
