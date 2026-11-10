# Chunking Strategies for Document Scanners

Chunking is the process of splitting a long document into smaller passages
before indexing. The chunk size and overlap have a large impact on retrieval
quality.

## Fixed-size character windows

The scanner in this project uses fixed-size character windows that respect word
boundaries. A chunk size of roughly 800 characters with 150 characters of
overlap works well for academic prose. Smaller chunks improve precision because
each passage covers a single idea, while overlap prevents important sentences
from being split awkwardly across two chunks.

## Trade-offs

- **Too small:** retrieval becomes noisy and context is lost.
- **Too large:** irrelevant text dilutes the similarity score and answers become
  vague.
- **Overlap:** increases index size but improves recall for facts that sit on a
  chunk boundary.

## Practical guidance

For a dissertation corpus of PDFs and Markdown notes, start with 800/150 and
tune based on the length of typical answers. Always normalize whitespace before
chunking so that page breaks and inconsistent spacing from PDF extraction do not
create spurious chunk boundaries.
