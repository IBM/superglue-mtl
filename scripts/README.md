# scripts for question rewriting

We use a set of rules based on syntactic structures to rewrite the questions
into statements, which can be used as data augmentation for NLI type tasks.

## File dependencies:

- `squad/train-v2.0.json`: raw squad 2.0 training set in json format.
- `squad/dev-v2.0.json`: raw squad 2.0 development set in json format.
- `squad/train.ctx.parse.jsonl`: squad 2.0 training data with parsing information on paragraphs.
- `squad/dev.ctx.parse.jsonl`: squad 2.0 development data with parsing information on paragraphs.
- `squad/train.parse.jsonl`: squad 2.0 training data with parsing information on questions.
- `squad/dev.parse.jsonl`: squad 2.0 development data with parsing information on questions.

## Model dependencies:

- cola\_bert-large\_epoch\_4.bin: a BERT-large model trained on CoLA dataset.

## Package dependencies:

- Currently we rely on the old huggingface package `pytorch_pretrained_bert`, this should be converted to `pytorch_transformers` in refactoring.
