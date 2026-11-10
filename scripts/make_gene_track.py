#!/usr/bin/env python
"""Builds the bundled gene track (samplot/data/GRCh38_mane_select.bed.gz).

One line per MANE Select transcript, tabix indexed:

    chrom  start  end  symbol  transcript  strand  cds_start  cds_end  exon_starts  exon_ends

Coordinates are 0-based half-open, chromosomes have no "chr" prefix, exons are
in genomic order and cds_start == cds_end for a transcript with no CDS.

Inputs (both from pyvariantfilter/data):
    --exons   transcripts/GRCh38_latest_genomic_no_nw.csv.gz  RefSeq exon and
              CDS features (seq_id,source,type,start,end,score,strand,phase,
              mRNA,chromosome; 1-based, accession without version)
    --genes   gene_models/gene_models_abridged.csv  HGNC symbols with their
              MANE Select transcript (symbol, mane_select columns)
"""
import argparse
import csv
import gzip
import os
import sys

import pysam

CHROM_ORDER = [str(c) for c in range(1, 23)] + ["X", "Y", "MT"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--exons", required=True)
    parser.add_argument("--genes", required=True)
    parser.add_argument("--output", required=True, help="path of the .bed.gz to write")
    args = parser.parse_args()

    # accession without version -> (symbol, versioned accession)
    mane = {}
    with open(args.genes, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mane_select"]:
                mane[row["mane_select"].split(".")[0]] = (row["symbol"], row["mane_select"])

    # (accession, chromosome) -> features. Keyed by chromosome as well so the
    # X and Y copies of pseudoautosomal genes stay separate
    transcripts = {}
    with gzip.open(args.exons, "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["mRNA"] not in mane or row["chromosome"] not in CHROM_ORDER:
                continue
            transcript = transcripts.setdefault(
                (row["mRNA"], row["chromosome"]),
                {"strand": row["strand"], "exon": set(), "CDS": set()},
            )
            if row["type"] in ("exon", "CDS"):
                transcript[row["type"]].add((int(row["start"]) - 1, int(row["end"])))

    lines = []
    for (accession, chromosome), transcript in transcripts.items():
        exons = sorted(transcript["exon"])
        if not exons:
            continue
        symbol, versioned = mane[accession]
        start, end = exons[0][0], exons[-1][1]
        cds = sorted(transcript["CDS"])
        cds_start, cds_end = (cds[0][0], cds[-1][1]) if cds else (start, start)
        lines.append(
            (
                CHROM_ORDER.index(chromosome),
                start,
                end,
                "\t".join(
                    str(x)
                    for x in (
                        chromosome, start, end, symbol, versioned, transcript["strand"],
                        cds_start, cds_end,
                        ",".join(str(e[0]) for e in exons),
                        ",".join(str(e[1]) for e in exons),
                    )
                ),
            )
        )
    lines.sort()

    plain = args.output[: -len(".gz")] if args.output.endswith(".gz") else args.output
    with open(plain, "w") as out:
        for line in lines:
            out.write(line[3] + "\n")
    pysam.tabix_compress(plain, plain + ".gz", force=True)
    os.remove(plain)
    pysam.tabix_index(plain + ".gz", preset="bed", force=True)

    found = {key[0] for key in transcripts}
    print(
        "{} transcripts written for {} of {} MANE Select genes".format(
            len(lines), len(found), len(mane)
        ),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
