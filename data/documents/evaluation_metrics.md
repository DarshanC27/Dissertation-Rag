# Evaluating Retrieval Quality

Evaluating a RAG system means measuring both retrieval quality and answer
quality. For a dissertation scanner the retrieval stage matters most, because a
correct answer can only be produced if the right passage is retrieved.

## Retrieval metrics

- **Precision@k:** the fraction of the top-k retrieved chunks that are relevant.
- **Recall@k:** the fraction of all relevant chunks that appear in the top-k.
- **Mean Reciprocal Rank (MRR):** rewards systems that place the first relevant
  chunk near the top of the list.

## Answer metrics

- **Groundedness:** whether every claim in the answer is supported by a
  retrieved passage. Extractive answers are grounded by construction.
- **Faithfulness:** whether the generated answer contradicts the source.
- **Answer relevance:** whether the answer actually addresses the question.

## Building a test set

Create a small set of question and expected-source pairs from your corpus. Run
the retriever and check whether the expected source appears in the top results.
Even a handful of questions catches regressions in chunking or vectorization.
