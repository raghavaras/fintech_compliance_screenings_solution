#!/usr/bin/env python3
"""
Name matching customers x entities: Jaccard, weighted Jaccard and TF-IDF cosine
distances on padded trigrams, for lastname and firstname.

Two steps:

  1) Global document frequencies (once, over ALL customer files + entities):
       python name_match.py idf --customers "cust_*.csv" --entities entities.csv --idf idf.pkl

  2) Matching, one customer file at a time (the 40 runs are independent,
     run them in parallel, e.g. with GNU parallel or xargs -P):
       python name_match.py match --customers cust_01.csv --entities entities.csv \
              --idf idf.pkl --out out_01.csv

Output: one row per (cust_id, ent_id) pair that shares at least one lastname gram,
is in the same block and satisfies the year window, with 6 distances (0 = identical,
1 = nothing in common, empty = missing name) and the year gap (empty = unknown).
"""

import argparse
import datetime
import glob
import pickle
import re
import unicodedata
from collections import Counter

import numpy as np
import pandas as pd
from scipy import sparse

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
MIN_YEAR = 1900
YEAR_WINDOW = 5
SHORT_WORD = 3          # words of <= 3 letters also get a whole-word token

DEAD_WORDS = frozenset([
    "A", "AL", "BIN", "D", "DA", "DAS", "DE", "DEL", "DELA", "DELLA",
    "DELLE", "DELLO", "DELOS", "DELS", "DES", "DI", "DO", "DOS", "DU",
    "EL", "LA", "LAS", "LE", "LES", "LO", "LOS", "L", "O", "OS", "ST",
    "SAINTE", "SAINT", "SAN", "SANTA", "SANTO", "TEN", "TER", "VAN",
    "VON", "Y",
])

# letters that NFKD does not decompose (applied after upper-casing)
SPECIAL = str.maketrans({
    "\u00c6": "AE", "\u0152": "OE", "\u00d8": "O", "\u0141": "L",
    "\u0110": "D", "\u00d0": "D", "\u00de": "TH", "\u1e9e": "SS",
})

NON_AZ = re.compile(r"[^A-Z]+")


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
def normalize_name(s):
    """Accents stripped, upper case, A-Z only, dead words removed (with fallback)."""
    if not isinstance(s, str) or s == "":
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.upper().translate(SPECIAL)
    s = NON_AZ.sub(" ", s).strip()
    kept = [w for w in s.split() if w not in DEAD_WORDS]
    if kept:
        return " ".join(kept)
    return s            # fallback: only dead words, keep the cleaned name


def normalize_series(series):
    """Normalize only the unique values, then map back (much faster)."""
    series = series.fillna("").astype(str)
    uniq = series.drop_duplicates()
    mapping = dict(zip(uniq, map(normalize_name, uniq)))
    return series.map(mapping)


def clean_year(series):
    """Year as int, -1 when missing or implausible (wildcard)."""
    y = pd.to_numeric(series, errors="coerce")
    ok = (y >= MIN_YEAR) & (y <= datetime.date.today().year)
    return y.where(ok, -1).astype("int64")


# --------------------------------------------------------------------------
# Grams
# --------------------------------------------------------------------------
def name_grams(name):
    """Set of padded trigrams of each word, plus whole-word token for short words."""
    grams = set()
    for w in name.split():
        p = "^" + w + "$"
        for i in range(len(p) - 2):
            grams.add(p[i:i + 3])
        if len(w) <= SHORT_WORD:
            grams.add("W:" + w)
    return grams


# --------------------------------------------------------------------------
# Step 1: global document frequencies
# --------------------------------------------------------------------------
def add_df(series, counter):
    """Add document frequencies of a normalized name column; return number of docs."""
    vc = series[series != ""].value_counts()
    for name, cnt in vc.items():
        cnt = int(cnt)
        for g in name_grams(name):
            counter[g] += cnt
    return int(vc.sum())


def cmd_idf(args):
    files = sorted(glob.glob(args.customers)) + [args.entities]
    counters = {"ln": Counter(), "fn": Counter()}
    totals = {"ln": 0, "fn": 0}
    for path in files:
        df = pd.read_csv(path, dtype=str, keep_default_na=False,
                         usecols=["lastname", "firstname"])
        totals["ln"] += add_df(normalize_series(df["lastname"]), counters["ln"])
        totals["fn"] += add_df(normalize_series(df["firstname"]), counters["fn"])
        print("idf pass:", path, flush=True)
    out = {f: (totals[f], dict(counters[f])) for f in ("ln", "fn")}
    with open(args.idf, "wb") as fh:
        pickle.dump(out, fh)


def load_idf(path):
    """Returns {field: (vocab dict gram->column, idf array)}."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    out = {}
    for f in ("ln", "fn"):
        n, df = data[f]
        grams = sorted(df)
        vocab = {g: i for i, g in enumerate(grams)}
        dfv = np.array([df[g] for g in grams], dtype=np.float64)
        out[f] = (vocab, np.log((n + 1.0) / (dfv + 1.0)) + 1.0)   # always > 0
    return out


# --------------------------------------------------------------------------
# Sparse representation of one name column
# --------------------------------------------------------------------------
def build_side(names, vocab, idf):
    """
    X  : binary gram matrix (rows = names)
    XW : X scaled by idf (weighted intersection / weighted Jaccard)
    T  : XW with L2-normalized rows (TF-IDF cosine)
    n  : number of grams per row;  s : sum of idf per row
    """
    indptr = [0]
    indices = []
    for nm in names:
        ids = sorted({vocab[g] for g in name_grams(nm) if g in vocab})
        indices.extend(ids)
        indptr.append(len(indices))
    data = np.ones(len(indices), dtype=np.float64)
    shape = (len(names), len(vocab))
    x = sparse.csr_matrix((data, np.array(indices, dtype=np.int32),
                           np.array(indptr, dtype=np.int64)), shape=shape)
    xw = (x @ sparse.diags(idf)).tocsr()
    norm = np.sqrt(np.asarray(xw.multiply(xw).sum(axis=1)).ravel())
    norm[norm == 0] = 1.0
    t = (sparse.diags(1.0 / norm) @ xw).tocsr()
    n = np.diff(x.indptr).astype(np.float64)
    s = np.asarray(x @ idf).ravel()
    return {"X": x, "XW": xw, "T": t, "n": n, "s": s}


def pair_scores(a, b, r, c):
    """Three distances for the pairs (row r of side a, row c of side b)."""
    na, nb = a["n"][r], b["n"][c]
    sa, sb = a["s"][r], b["s"][c]
    xb = b["X"][c]
    inter = np.asarray(a["X"][r].multiply(xb).sum(axis=1)).ravel()
    interw = np.asarray(a["XW"][r].multiply(xb).sum(axis=1)).ravel()
    cos = np.asarray(a["T"][r].multiply(b["T"][c]).sum(axis=1)).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        jac = 1.0 - inter / (na + nb - inter)
        wjac = 1.0 - interw / (sa + sb - interw)
    cosd = 1.0 - cos
    empty = (na == 0) | (nb == 0)
    jac[empty] = np.nan
    wjac[empty] = np.nan
    cosd[empty] = np.nan
    return (np.clip(jac, 0.0, 1.0) + 0.0, np.clip(wjac, 0.0, 1.0) + 0.0,
            np.clip(cosd, 0.0, 1.0) + 0.0)


# --------------------------------------------------------------------------
# Input
# --------------------------------------------------------------------------
def load_keys(path, id_col):
    """Unique (lastname, firstname, year) keys + map key -> id."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False,
                     usecols=[id_col, "lastname", "firstname", "year_of_birth"])
    df["ln"] = normalize_series(df["lastname"])
    df["fn"] = normalize_series(df["firstname"])
    df["yob"] = clean_year(df["year_of_birth"])
    df["key"] = df.groupby(["ln", "fn", "yob"], sort=False).ngroup()
    idmap = df[[id_col, "key"]]
    keys = df.drop_duplicates("key")[["key", "ln", "fn", "yob"]].reset_index(drop=True)
    return keys, idmap


# --------------------------------------------------------------------------
# Adaptive prefix blocking (trie with size cap) on the sorted entity lastnames
# --------------------------------------------------------------------------
def build_leaves(names, cap, maxlen):
    """
    names: sorted list of non-empty lastnames.
    Returns {prefix: (lo, hi)} = contiguous ranges of the sorted list.
    A block bigger than cap is split on the next letter. Names that equal the
    split prefix exactly are stored under the key prefix + '$'.
    """
    leaves = {}
    stack = [("", 0, len(names), 0)]
    while stack:
        prefix, lo, hi, depth = stack.pop()
        if hi - lo <= cap or depth >= maxlen:
            leaves[prefix] = (lo, hi)
            continue
        start = lo
        while start < hi and len(names[start]) == depth:
            start += 1
        if start > lo:
            leaves[prefix + "$"] = (lo, start)
        i = start
        while i < hi:
            ch = names[i][depth]
            j = i
            while j < hi and names[j][depth] == ch:
                j += 1
            stack.append((prefix + ch, i, j, depth + 1))
            i = j
    return leaves


def route_name(name, leaves, maxlen):
    """Entity range for a customer lastname, or None if no entity shares its prefix."""
    for l in range(min(len(name), maxlen), -1, -1):
        rng = leaves.get(name[:l])
        if rng is not None:
            return rng
        if l == len(name):
            rng = leaves.get(name + "$")
            if rng is not None:
                return rng
    return None


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
def score_chunk(ids, blo, et, side, cyr, eyr, max_dist):
    """Score a chunk of customer keys against one entity block."""
    prod = (side[("c", "ln")]["X"][ids] @ et).tocoo()
    if prod.nnz == 0:
        return None
    r = ids[prod.row]
    c = blo + prod.col
    yc, ye = cyr[r], eyr[c]
    ok = (yc < 0) | (ye < 0) | (np.abs(yc - ye) <= YEAR_WINDOW)
    r, c = r[ok], c[ok]
    if len(r) == 0:
        return None
    ln = pair_scores(side[("c", "ln")], side[("e", "ln")], r, c)
    keep = ln[0] <= max_dist            # NaN never kept
    r, c = r[keep], c[keep]
    if len(r) == 0:
        return None
    ln = [x[keep] for x in ln]
    fn = pair_scores(side[("c", "fn")], side[("e", "fn")], r, c)
    yc, ye = cyr[r], eyr[c]
    gap = np.where((yc < 0) | (ye < 0), np.nan, np.abs(yc - ye))
    return pd.DataFrame({
        "cpos": r, "epos": c,
        "ln_jac": ln[0], "ln_wjac": ln[1], "ln_tfidf": ln[2],
        "fn_jac": fn[0], "fn_wjac": fn[1], "fn_tfidf": fn[2],
        "year_gap": gap,
    })


def cmd_match(args):
    idfs = load_idf(args.idf)
    ck, cmap = load_keys(args.customers, "cust_id")
    ek, emap = load_keys(args.entities, "ent_id")
    ek = ek[ek["ln"] != ""].sort_values(["ln", "yob"]).reset_index(drop=True)

    side = {}
    for tag, keys in (("c", ck), ("e", ek)):
        for f in ("ln", "fn"):
            vocab, idf = idfs[f]
            side[(tag, f)] = build_side(keys[f].tolist(), vocab, idf)
    cyr = ck["yob"].to_numpy()
    eyr = ek["yob"].to_numpy()

    leaves = build_leaves(ek["ln"].tolist(), args.block_cap, args.max_prefix)

    # route each customer key to an entity block
    lo = np.full(len(ck), -1, dtype=np.int64)
    hi = np.full(len(ck), -1, dtype=np.int64)
    cache = {}
    for i, name in enumerate(ck["ln"].tolist()):
        if name == "":
            continue
        if name not in cache:
            cache[name] = route_name(name, leaves, args.max_prefix)
        rng = cache[name]
        if rng is not None:
            lo[i], hi[i] = rng
    blocks = pd.DataFrame({"i": np.arange(len(ck)), "lo": lo, "hi": hi})
    blocks = blocks[blocks["lo"] >= 0]

    parts = []
    for (blo, bhi), g in blocks.groupby(["lo", "hi"]):
        idx = g["i"].to_numpy()
        et = side[("e", "ln")]["X"][blo:bhi].T.tocsr()
        for start in range(0, len(idx), args.chunk):
            res = score_chunk(idx[start:start + args.chunk], blo, et, side,
                              cyr, eyr, args.max_last_dist)
            if res is not None:
                parts.append(res)

    if not parts:
        print("no pairs")
        return
    res = pd.concat(parts, ignore_index=True)
    res["key_c"] = ck["key"].to_numpy()[res["cpos"].to_numpy()]
    res["key_e"] = ek["key"].to_numpy()[res["epos"].to_numpy()]
    res = res.drop(columns=["cpos", "epos"])
    res = res.merge(cmap.rename(columns={"key": "key_c"}), on="key_c")
    res = res.merge(emap.rename(columns={"key": "key_e"}), on="key_e")
    cols = ["cust_id", "ent_id", "ln_jac", "ln_wjac", "ln_tfidf",
            "fn_jac", "fn_wjac", "fn_tfidf", "year_gap"]
    res[cols].to_csv(args.out, index=False, float_format="%.4f")
    print("pairs written:", len(res))


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("idf")
    p1.add_argument("--customers", required=True, help='glob, e.g. "cust_*.csv"')
    p1.add_argument("--entities", required=True)
    p1.add_argument("--idf", required=True)

    p2 = sub.add_parser("match")
    p2.add_argument("--customers", required=True)
    p2.add_argument("--entities", required=True)
    p2.add_argument("--idf", required=True)
    p2.add_argument("--out", required=True)
    p2.add_argument("--block-cap", type=int, default=2000,
                    help="max entities per block before splitting on next letter")
    p2.add_argument("--max-prefix", type=int, default=4)
    p2.add_argument("--chunk", type=int, default=1000,
                    help="customer keys per sparse product")
    p2.add_argument("--max-last-dist", type=float, default=0.8,
                    help="output-volume guard on lastname Jaccard distance; 1.0 = keep all")

    args = ap.parse_args()
    if args.cmd == "idf":
        cmd_idf(args)
    else:
        cmd_match(args)


if __name__ == "__main__":
    main()
