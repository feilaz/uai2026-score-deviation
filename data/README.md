# Data attribution

`inputs.json.gz` contains 300 examples selected from **FACTS Grounding 1.0 Public Examples**, created by Google DeepMind and Google Research.

- Source: [official dataset and attribution](https://huggingface.co/datasets/google/FACTS-grounding-public)
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)
- Reference: Alon Jacovi et al., [FACTS Grounding technical report](https://storage.googleapis.com/deepmind-media/FACTS/FACTS_grounding_paper.pdf), 2024.

Changes: selected 300 examples, renamed `user_request` to `query` and `context_document` to `context`, assigned local `query_id` values, and stored the records as compressed JSON. The question and context text are unchanged; the upstream system instructions and combined prompts are omitted.

The numerical scores, labels, entropies and split files are outputs of this study, not official FACTS annotations. The repository's MIT code license does not replace the input dataset's CC BY 4.0 license. This project is not endorsed by the dataset creators.
