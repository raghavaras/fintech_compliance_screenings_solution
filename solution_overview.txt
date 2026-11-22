# Record Linkage for Sanctions Screening

## Overview

A deterministic record linkage system developed to screen a large client database against sanctions and watchlist entities.

The system was designed to operate at high volume on a single Linux server, matching approximately 40 million client records against a reference database of approximately 200,000 entities (Dow Jones format).

The approach relies entirely on deterministic matching techniques, combining character n-gram similarity and blocking strategies. It does not use probabilistic record linkage methods such as the Fellegi–Sunter model.

## Data

### Client database

The client database contains approximately 40 million records in a flat-file format:

`ID | Type (PP/PM) | Last Name | First Name | Year of Birth | Month of Birth | Day of Birth | Country of Birth`

It consists of two distinct populations:

- **Individuals (PP – Personnes Physiques):** Approximately 38 million records, with first name, last name, date of birth and country of birth.
- **Corporates (PM – Personnes Morales):** Approximately 2 million records, with an identifier and company name.

### Reference database

The reference database contains approximately 200,000 sanctions and watchlist entities in Dow Jones format.

The reference file is maintained as a single dataset and is not partitioned during processing.

## Methodology

### Matching strategy

The system uses deterministic record linkage based on character-level string similarity, with bigrams and trigrams as the primary similarity features.

Blocking techniques are applied before similarity computation to reduce the candidate comparison space and make large-scale matching computationally tractable.

Two separate matching pipelines are implemented to account for the different information available for individuals and corporate entities:

- **Individuals:** Matching based on first name, last name, date of birth and country of birth.
- **Corporates:** Matching based primarily on company name.

### Processing architecture

Given the size of the client database, processing is performed in chunks rather than loading the entire dataset into memory.

The default chunk size is approximately one million client records per file. Each chunk is processed against the complete reference database.

This architecture allows the system to operate within the memory and I/O constraints of a single Linux server, without requiring distributed computing infrastructure.

## Technical Environment

- **Language:** Python
- **Platform:** Single Linux server
- **Client database:** Approximately 40 million records
- **Reference database:** Approximately 200,000 entities
- **Processing:** Chunk-based, with a default chunk size of one million records
- **Matching:** Deterministic character n-gram similarity (bigrams and trigrams)
- **Candidate reduction:** Blocking

## Design Principles

The implementation follows three main principles:

1. **Scalability:** Make large-scale record linkage feasible on a single server through blocking and chunked processing.
2. **Determinism:** Use explicit string-similarity rules rather than probabilistic matching models.
3. **Separation of concerns:** Maintain distinct matching logic for individuals and corporate entities, reflecting their different data structures and available attributes.
