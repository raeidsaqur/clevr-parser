#!/bin/bash

# https://spacy.io/models/en

echo -e "Downloading spacy core en models, for pipelines: tagger, parser and ner"
python -m spacy download en_core_web_sm
wait;
python -m spacy download en_core_web_lg
wait;

# N.b. the context aware models can't be added for 'parsing' pipeline
# Only for  pipelines: sentencizer, trf_wordpiecer, trf_tok2vec
echo -e "Downloading spacy context-aware en models"
python -m spacy download en_trf_bertbaseuncased_lg
wait;
python -m spacy download en_trf_xlnetbasecased_lg
wait;
python -m spacy download en_trf_robertabase_lg
wait;

echo "Completed"
echo `python -m spacy validate`

return 0