# Retrieval-Augmented Generation Overview

Retrieval-Augmented Generation (RAG) is an architecture that combines an
information retrieval component with a text generation component. Instead of
relying only on parameters learned during training, a RAG system first retrieves
relevant passages from an external corpus and then conditions its answer on
those passages. This grounding reduces hallucination and lets the system answer
questions about documents it never saw during training.

## Core stages

1. **Scanning and ingestion.** Source documents are read from disk, converted to
   plain text, and split into overlapping chunks. Chunking keeps passages small
   enough to be precise while overlap preserves context across boundaries.
2. **Indexing.** Each chunk is converted into a vector. This project uses TF-IDF
   vectors with unigrams and bigrams, which require no model download and run
   fully offline.
3. **Retrieval.** A user question is vectorized with the same representation and
   compared to every chunk using cosine similarity. The highest-scoring chunks
   are returned as context.
4. **Generation.** The retrieved context is turned into an answer. The default
   generator is extractive: it returns the most relevant passage verbatim. When
   an OpenAI API key is available, an abstractive answer can be synthesized
   instead.

## Why TF-IDF for a dissertation scanner

TF-IDF is a strong, transparent baseline for retrieval over a modest corpus. It
is deterministic, cheap to compute, and easy to reproduce in a development
environment, which makes it ideal for demonstrating an end-to-end RAG pipeline
without external dependencies or GPUs.
