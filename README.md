# name_match

Fuzzy name matching between `customers` (40 files of ~1M rows) and `entities` (~200k rows).
Output: lastname and firstname distances (Jaccard, weighted Jaccard, TF-IDF cosine) plus the year gap, for candidate pairs. No combination, no decision.

Requires: Python 3, `numpy`, `pandas`, `scipy`.

## Input

CSV with a header.

- `customers`: `cust_id, lastname, firstname, year_of_birth`
- `entities`: `ent_id, lastname, firstname, year_of_birth`

## Usage

```bash
# 1. once, over all customer files + entities (global IDF)
python name_match.py idf --customers "cust_*.csv" --entities entities.csv --idf idf.pkl

# 2. per customer file (independent, run in parallel)
python name_match.py match --customers cust_01.csv --entities entities.csv \
       --idf idf.pkl --out out_01.csv
```

Options for `match`:

| option | default | meaning |
|---|---|---|
| `--block-cap` | 2000 | max entities per block before splitting on the next letter |
| `--max-prefix` | 4 | max prefix length for blocking |
| `--chunk` | 1000 | customer keys per sparse product |
| `--max-last-dist` | 0.8 | output-volume guard on lastname Jaccard distance (1.0 = off) |

## Method

1. **Normalize**: accents stripped, upper case, A-Z only, dead words removed (fallback: keep the name if only dead words). Year < 1900 or in the future = missing.
2. **Dedupe** on (lastname, firstname, year) keys; results are joined back to ids at the end.
3. **Grams**: padded trigrams per word (`^SMITH$`), plus the whole word for words of 3 letters or fewer.
4. **Blocking**: adaptive prefix trie on entity lastnames (frequent prefixes get longer), then year within ±5 (missing year = wildcard).
5. **Scores**: lastname on sparse matrix products, firstname on surviving pairs only.

## Output columns

`cust_id, ent_id, ln_jac, ln_wjac, ln_tfidf, fn_jac, fn_wjac, fn_tfidf, year_gap`

Distances: 0 = identical, 1 = nothing in common, empty = a name is missing. `year_gap` is empty if either year is unknown.

## Known limits

- A typo in letters 2 to 4 of the lastname, inside a large block, means a miss.
- Multi-word lastnames are blocked on the whole string.
- Only pairs sharing at least one lastname gram are output.
- The IDF file must be built from all files you intend to match.

## Later: Fellegi-Sunter (EM)

The six distances and the year gap can feed a Fellegi-Sunter model once binned into a few levels (for example exact, close, far, missing). Why a previous EM attempt may have given unreliable scores, to check here:

- **Conditional independence is violated**: lastname and firstname grams, and the three distances of one field, are strongly correlated. Use one distance per field, or model the dependence.
- **Biased pair sample**: blocking and the "shares a gram" rule remove most non-matches, so EM sees a distorted mix. The u-probabilities should be estimated from random pairs, not from candidates.
- **Frequency effects**: a match on SMITH carries less evidence than on a rare name. Use frequency-specific (term-based) u-probabilities, or keep the IDF-weighted score.
- **Unidentifiable fit**: EM can converge to a split that is not "match vs non-match" (for example "common names vs rare names"). Fix good starting values and sanity-check on a small labelled set.
- **Prior (match rate)**: with 40M x 200k pairs, the prior is tiny and sensitive to the blocking.

A few hundred hand-labelled pairs would help to validate any scoring.
