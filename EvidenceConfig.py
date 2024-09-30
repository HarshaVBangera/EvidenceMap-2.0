PALLETE_COLORS = {
    "Intervention": "#C5E0B5",
    "Outcome": "#FFD962",
    "Participant": "#A1C2E1",
    "Count": "#D9D9D9",
    "Observation": "#D9D9D9"
}

PIPELINE_CONFIG = {
    "SentenceClassification": {
        "preferred-driver": "V1",
        "V0": {
            "config": {
                "conda_env": "EvidenceMap-tf1",
            },
        },
        "V1": {
            "config": {
                "model": "NeuML/pubmedbert-base-embeddings",
                "per_device_eval_batch_size": 128
            }
        }
    },
    "BaseNLP": {
        "preferred-driver": "V2",
        "V1": {
            "base_model": "dmis-lab/biobert-base-cased-v1.2",
            "config": {
                "corpora_group": "Combined",
                "lowercase": False,
            },
        },
        "V2": {
            "base_model": "bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16",
            "config": {
                "corpora_group": "Combined",
                "lowercase": True,
                "model_key": "2024-06-01_16-14-48.254720",
                "eval_batch_size": 1
            },
        }
    },
    "Negation": {
        "preferred-driver": "V1",
        "V0": {
            "config": {
                "tag_possible_phrases": False,
            },
        },
        "V1": {
            "config": {
                "task": "negation",
                "cue_model": "bert-base-uncased",
                "scope_model": "xlnet-base-cased",
                "f1_method": "average",
                "per_device_batch_size": 128
            },
        },
    },
    "Proposition": {
        "preferred-driver": "V0",
        "V0": {
            "config": {},
        },
    },
    "EvidenceMap": {
        "preferred-driver": "V1",
        "V1": {
            "config": {
                "intervention_color": PALLETE_COLORS["Intervention"],
                "outcome_color": PALLETE_COLORS["Outcome"],
                "participant_color": PALLETE_COLORS["Participant"],
                "count_color": PALLETE_COLORS["Count"],
                "observation_color": PALLETE_COLORS["Observation"]
            },
        },
    },
}

SOLR_CONFIG = {
    "base_url": "http://localhost:8983/solr/",
    "EvidenceMap_core": "EvidenceMap",
    "UMLS_core": "UMLS"
}
