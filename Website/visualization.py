from EvidenceConfig import PALLETE_COLORS
from spacy import displacy


def clean_data(raw_data):
    output = {}
    output["doc_id"] = raw_data["doc_id"]
    output["type of study"] = raw_data["type of study"]
    output["abstract"] = raw_data["abstract"]
    output["title"] = raw_data["title"]

    if "Sentence-level breakdown" in raw_data:
        output["sentences"] = build_sentences(raw_data["Sentence-level breakdown"])
    elif "Sentences-level breakdown" in raw_data:
        output["sentences"] = build_sentences(raw_data["Sentences-level breakdown"])
    return output


def extract_level3(raw_data):
    if "level3" in raw_data:
        return raw_data["level3"]["data"], len(raw_data["level3"]["data"]["proposed_arms"]), raw_data["level3"][
            "data_provider"]
    elif "study design" in raw_data and "study results" in raw_data:
        return {"study design": raw_data["study design"], "study results": raw_data["study results"]}, len(
            raw_data["study results"]) - 1, "legacy"
    else:
        return '', '', ''


def build_sentences(sentences_data):
    output = []

    for datum in sentences_data:
        if datum["Section"] == "!BACKGROUND":
            output.append(parse_other_sentence(datum))
        else:
            output.append(parse_other_sentence(datum))

    return output


def parse_other_sentence(datum):
    temp = {}
    temp["section"] = datum["Section"]
    temp["text"] = datum["Text"]
    ann = annotate_sent_with_entity(datum)  # a0c1b8 f4ebc1
    # Load colors from the Master Config
    colors = {"PAR": PALLETE_COLORS["Participant"], "INT": PALLETE_COLORS["Intervention"],
              "OUT": PALLETE_COLORS["Outcome"]}

    # colors={"PAR":"#CCCCFF","INT":"#9FE2BF","OUT":"#F7DC6F"}
    # https://colorhunt.co/palette/206889
    options = {"ents": ["PAR", "INT", "OUT"], "colors": colors}
    ann_html = displacy.render(ann, style="ent", manual=True, options=options)
    temp["text_ann"] = ann_html
    # temp["Evidence Elements"] = datum["Evidence Elements"]
    temp["Evidence Propositions"] = datum["Evidence Propositions"]
    return temp


def annotate_sent_with_entity(sent_data):
    text = sent_data["Text"]
    elements = sent_data["Evidence Elements"]
    annotation = [{"text": text, "ents": [], "title": None}]
    tags = {"Participant": "PAR", "Intervention": "INT", "Outcome": "OUT"}
    entities = {}  # entities[start]=term+end
    for tag in tags.keys():
        for element in elements[tag]:
            if element["start"] == "NA":
                continue

            entities[element["start"]] = [element["end"], tags[tag]]

    for start in sorted(entities.keys()):
        annotation[0]["ents"].append({"start": start, "end": entities[start][0], "label": entities[start][1]})
    return annotation
