from concurrent.futures import ThreadPoolExecutor

import spacy

nlp = spacy.load("en_core_web_trf")


def extract_entity_labels(text):
    doc = nlp(text)
    return {ent.label_ for ent in doc.ents}


def preprocess_sentences(sentences):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract_entity_labels, sentences))
    return results


def find_missing_labels(all_entity_labels):
    unique_labels = set().union(*all_entity_labels)
    return [list(unique_labels - labels) for labels in all_entity_labels]


def process_merged_fact(merged_fact):
    try:
        sentences = [fact["fact_group_content"] for fact in merged_fact["facts"]]
        all_entity_labels = preprocess_sentences(sentences)
        missing_entities = find_missing_labels(all_entity_labels)

        for fact, missing in zip(merged_fact["facts"], missing_entities):
            fact["missing_entities"] = missing
    except Exception as e:
        print(e)
    return merged_fact


def get_missing_entities(merged_fact_clusters):
    with ThreadPoolExecutor() as executor:
        for cluster in merged_fact_clusters:
            cluster["merged_facts"] = list(
                executor.map(process_merged_fact, cluster["merged_facts"])
            )
    return merged_fact_clusters
