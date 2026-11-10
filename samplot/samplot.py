#!/usr/bin/env python
import argparse
import logging
import os
import random
import re
import sys
from argparse import SUPPRESS
from dataclasses import dataclass, fields
from typing import List, Optional

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.ticker as ticker
import numpy as np
import pysam
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredText


logger = logging.getLogger(__name__)

INTERCHROM_YAXIS = 5000


class SamplotError(Exception):
    """Raised when a plot cannot be made (bad input files, unsupported regions etc).

    The command line turns this into an error message and a non-zero exit;
    library callers can catch it.
    """


@dataclass
class PlotOptions:
    """Every setting of a plot, with its default.

    The fields are the long names of the `samplot plot` arguments (whose
    defaults come from here), so this is both what the command line parses
    into and what plot_region() builds from its keyword arguments.
    """

    # what to plot
    chrom: Optional[List[str]] = None
    start: Optional[List[int]] = None
    end: Optional[List[int]] = None
    bams: Optional[List[str]] = None
    sv_type: Optional[str] = None
    reference: Optional[str] = None
    titles: Optional[List[str]] = None
    start_ci: Optional[List[int]] = None
    end_ci: Optional[List[int]] = None
    # region shown
    window: Optional[int] = None
    zoom: int = 500000
    # reads
    z: int = 4
    max_depth: int = 1
    include_mqual: int = 1
    separate_mqual: int = 0
    long_read: int = 1000
    min_event_size: int = 20
    ignore_hp: bool = False
    coverage_only: bool = False
    discordant_only_min_size: int = 100000
    # coverage
    coverage_tracktype: str = "stack"
    max_coverage_points: int = 1000
    max_coverage: int = 0
    same_yaxis_scales: bool = False
    coverage_trend_min_size: int = 20000
    # annotation tracks
    transcript_file: Optional[str] = None
    transcript_filename: Optional[str] = None
    annotation_files: Optional[List[str]] = None
    annotation_filenames: Optional[List[str]] = None
    hide_annotation_labels: bool = False
    annotation_scalar: float = 0.3
    genes: Optional[str] = None
    gene_detail_max_size: int = 200000
    # appearance
    plot_height: Optional[int] = None
    plot_width: Optional[int] = None
    xaxis_label_fontsize: int = 6
    yaxis_label_fontsize: int = 6
    legend_fontsize: int = 6
    annotation_fontsize: int = 6
    marker_size: int = 3
    jitter: float = 0.0
    dpi: int = 150
    # output and housekeeping
    output_file: Optional[str] = None
    output_dir: str = "."
    print_args: bool = False
    json_only: bool = False
    debug: Optional[str] = None
    random_seed: int = 9999

    @classmethod
    def from_namespace(cls, namespace):
        """Builds options from parsed command line arguments"""
        return cls(
            **{
                f.name: getattr(namespace, f.name)
                for f in fields(cls)
                if hasattr(namespace, f.name)
            }
        )


# Palette. Dark ink for normal/deletion evidence, orange for duplications,
# blue for inversions (colour-blind checked hues), on a light grey panel with
# white gridlines.
INK = "#0b0b0b"
MUTED_INK = "#52514e"
ORANGE = "#eb6834"
BLUE = "#2a78d6"
GREEN = "#1baf7a"
PURPLE = "#8e5bd0"
GREY = "#8a8986"
PANEL_BACKGROUND = "#EAEAF2"

COLORS = {
    "Deletion/Normal": INK,
    "Deletion": INK,
    "Duplication": ORANGE,
    "Inversion": BLUE,
    "InterChrmInversion": BLUE,
    "InterChrm": INK,
    "Align": PURPLE,
    "Aligned long read": PURPLE,
    "Linked read": GREEN,
}

# rcParams applied (as a context, not globally) while a figure is built
STYLE_RC = {
    "font.size": 12,
    "font.family": "sans-serif",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.facecolor": PANEL_BACKGROUND,
    "axes.edgecolor": "white",
    "axes.grid": False,
    "axes.axisbelow": True,
    "grid.color": "white",
    "grid.linewidth": 0.8,
    "xtick.color": MUTED_INK,
    "ytick.color": MUTED_INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "agg.path.chunksize": 100000,
}

READ_TYPES_USED = {
    "Deletion/Normal": False,
    "Duplication": False,
    "Inversion": False,
    "Aligned long read": False,
    "Linked read": False,
    "Split-read": False,
    "Paired-end read": False,
}

# pysam.readthedocs.io/en/latest/api.html#pysam.AlignedSegment.cigartuples
CIGAR_MAP = {
    "M": 0,
    "I": 1,
    "D": 2,
    "N": 3,
    "S": 4,
    "H": 5,
    "P": 6,
    "=": 7,
    "X": 8,
    "B": 9,
}

def strip_chr(chrom):
    """
    safer way to replace chr string, to support non-human genomes
    """
    if chrom[:3] == "chr":
        chrom = chrom[3:]
    return chrom

# {{{class plan_step:
class plan_step:
    step_events = ["Align", "ANNOTATION"]

    def __init__(self, start_pos, end_pos, event, info=None):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.event = event
        self.info = info

    def __str__(self):
        if self.info:
            return (
                "Step("
                + str(self.start_pos)
                + ", "
                + str(self.end_pos)
                + ", "
                + self.event
                + ", "
                + str(self.info)
                + ")"
            )
        else:
            return (
                "Step("
                + str(self.start_pos)
                + ", "
                + str(self.end_pos)
                + ", "
                + self.event
                + ")"
            )

    def __repr__(self):
        return str(self)


# }}}

# {{{class genome_interval:
class genome_interval:
    def __init__(self, chrm, start, end):
        self.chrm = chrm
        self.start = start
        self.end = end

    def __str__(self):
        return "(" + self.chrm + "," + str(self.start) + "," + str(self.end) + ")"

    def __repr__(self):
        return str(self)

    def __eq__(self, gi2):
        return self.chrm == gi2.chrm and self.start == gi2.start and self.end == gi2.end

    """ return -1 if before, 0 if in, 1 if after """

    def intersect(self, gi):
        if strip_chr(gi.chrm) < strip_chr(self.chrm) or gi.end < self.start:
            return -1
        elif strip_chr(gi.chrm) > strip_chr(self.chrm) or gi.start > self.end:
            return 1
        else:
            return 0


# }}}

# {{{def get_range_hit(ranges, chrm, point):
def get_range_hit(ranges, chrm, point):
    for j in range(len(ranges)):
        r = ranges[j]
        if (
            strip_chr(r.chrm) == strip_chr(chrm)
            and r.start <= point
            and r.end >= point
        ):
            return j
    return None


# }}}

# {{{def map_genome_point_to_range_points(ranges, chrm, point):
def map_genome_point_to_range_points(ranges, chrm, point):
    range_hit = get_range_hit(ranges, chrm, point)

    if range_hit == None:
        return None
    p = 1.0 / len(ranges) * range_hit + (1.0 / len(ranges)) * (
        float(point - ranges[range_hit].start)
        / float(ranges[range_hit].end - ranges[range_hit].start)
    )

    return p


# }}}

# {{{def points_in_window(points):
def points_in_window(points):
    """Checks whether these points lie within the window of interest

    Points is a list of one start, one end coordinate (ints)
    """
    if (
        None in points
        or points[0] < -5
        or points[1] < -5
        or points[0] > 5
        or points[1] > 5
    ):
        return False
    return True


# }}}

# {{{ def get_tabix_iter(chrm, start, end, datafile):
def get_tabix_iter(chrm, start, end, datafile):
    """Gets an iterator from a tabix BED/GFF3 file

    Used to avoid chrX vs. X notation issues when extracting data from
    annotation files
    """
    try:
        tbx = pysam.TabixFile(datafile)
    except:
        tbx = pysam.TabixFile(datafile, index=datafile+".csi")


    itr = None
    try:
        itr = tbx.fetch(chrm, max(0, start - 1000), end + 1000)
    except ValueError:
        # try and account for chr/no chr prefix
        if chrm[:3] == "chr":
            chrm = chrm[3:]
        else:
            chrm = "chr" + chrm

        try:
            itr = tbx.fetch(chrm, max(0, start - 1000), end + 1000)
        except ValueError as e:
            logger.warning(
                "Could not fetch {}:{}-{} from {}".format(
                    chrm,
                    start,
                    end,
                    datafile
                )
            )
            print(e)
    return itr


# }}}

##Coverage methods
# {{{def add_coverage(bam_file, read, coverage, separate_mqual):
def add_coverage(read, coverage_matrix, offset, column):
    """Adds a read to the known coverage 

    Coverage from Pysam read is added to coverage_matrix.
    offset defines the start position of the current range
    column specifies which column to add to.
    """

    curr_pos = read.reference_start
    if not read.cigartuples:
        return

    for op, length in read.cigartuples:
        if op in [CIGAR_MAP["M"], CIGAR_MAP["="], CIGAR_MAP["X"]]:
            # reads can start before the range, so clip to it (a negative
            # slice start would wrap around to the end of the matrix)
            block_start = max(curr_pos - offset, 0)
            block_end = curr_pos + length - offset
            if block_end > block_start:
                coverage_matrix[block_start:block_end, column] += 1
            curr_pos += length
        elif op in [CIGAR_MAP["D"], CIGAR_MAP["N"]]:
            curr_pos += length
        # insertions, clips and padding do not consume the reference


# }}}

class CoverageCounter:
    """Counts per-base depth over a range from the reads added to it.

    Aligned blocks are recorded as start/end offsets and turned into depth
    with one cumulative sum at the end, which is much cheaper than
    incrementing an array slice for every read.
    """

    def __init__(self, start, length):
        self.start = start
        self.length = length
        self.starts = ([], [])
        self.ends = ([], [])

    def add(self, read, column):
        cigartuples = read.cigartuples
        if not cigartuples:
            return
        starts = self.starts[column]
        ends = self.ends[column]
        curr_pos = read.reference_start - self.start
        length = self.length
        for op, op_length in cigartuples:
            # M, = and X are aligned bases
            if op == 0 or op == 7 or op == 8:
                block_start = curr_pos if curr_pos > 0 else 0
                curr_pos += op_length
                block_end = curr_pos if curr_pos < length else length
                if block_end > block_start:
                    starts.append(block_start)
                    ends.append(block_end)
            # D and N consume the reference without covering it
            elif op == 2 or op == 3:
                curr_pos += op_length

    def to_array(self):
        """Depth as an array of shape (length, 2), one column per quality bin"""
        values = np.zeros((self.length, 2), dtype=np.int32)
        for column in (0, 1):
            if not self.starts[column]:
                continue
            steps = np.bincount(
                self.starts[column], minlength=self.length + 1
            ) - np.bincount(self.ends[column], minlength=self.length + 1)
            values[:, column] = np.cumsum(steps[: self.length])
        return values


class RangeCoverage:
    """Per-base coverage over one fetched range.

    values is an array of shape (end - start, 2): column 0 holds the high
    mapping quality depth and column 1 the low quality depth (all depth is in
    column 1 unless separate_mqual is used).
    """

    def __init__(self, start, values):
        self.start = start
        self.values = values

    @property
    def end(self):
        return self.start + len(self.values)

    def __repr__(self):
        return "RangeCoverage({}, {})".format(self.start, self.end)


def get_range_coverage(chrm_coverage, start, end):
    """Returns the (end - start + 1, 2) depth array for start..end inclusive.

    chrm_coverage is a list of RangeCoverage for one chromosome. Positions
    with no data are zero.
    """
    out = np.zeros((end - start + 1, 2), dtype=float)
    for rc in chrm_coverage or []:
        lo = max(start, rc.start)
        hi = min(end + 1, rc.end)
        if hi > lo:
            out[lo - start: hi - start] = rc.values[lo - rc.start: hi - rc.start]
    return out


def bin_coverage(values, max_points):
    """Averages per-base depth into at most max_points equal width bins.

    Returns (bin centres as offsets into values, binned values). Averaging,
    rather than taking every Nth base, is what keeps wide regions from
    looking spiky: each plotted point is the mean depth of everything it
    stands for.
    """
    n = len(values)
    n_bins = max(1, min(n, int(max_points)))
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    widths = np.diff(edges)
    binned = np.add.reduceat(values, edges[:-1], axis=0) / widths[:, None]
    centres = edges[:-1] + (widths - 1) / 2.0
    return centres, binned


def rolling_median(values, window):
    """Centred rolling median (the series is reflected at its ends)"""
    values = np.asarray(values, dtype=float)
    half = max(1, int(window) // 2)
    if len(values) <= half:
        return np.full(len(values), np.median(values) if len(values) else 0.0)
    padded = np.pad(values, half, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * half + 1)
    return np.median(windows, axis=1)


def nice_axis_top(value, intervals):
    """Smallest 'round' axis maximum >= value that divides into intervals
    equal steps, so both y axes can share the same gridlines"""
    if value <= 0:
        return float(intervals)
    step = value / float(intervals)
    magnitude = 10 ** np.floor(np.log10(step))
    for nice in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if nice * magnitude >= step * 0.999:
            step = nice * magnitude
            break
    return float(step * intervals)


def is_copy_number_type(sv_type):
    """True for variant types where depth is expected to change"""
    if not sv_type:
        return False
    sv_type = str(sv_type).upper()
    return any(t in sv_type for t in ("DEL", "DUP", "CNV", "LOSS", "GAIN"))


# {{{def plot_coverage(coverage,
def plot_coverage(coverage, ax, ranges, hp_count, max_coverage, options, sv=None):
    """Plots high and low quality coverage for the region

    User may specify a preference between stacked and superimposed 
    superimposed may cause unexpected behavior if low-quality depth is
    greater than high 

    Depth is averaged into at most options.max_coverage_points bins per range. For
    wide regions (>= options.coverage_trend_min_size bases plotted, 0 to disable) a
    rolling median trend line is drawn over the binned depth, and for copy
    number variants the flanking depth is marked with a dashed guide and the
    inside/flank depth ratio is written on the panel.
    """
    cover_x = []
    cover_y_lowqual = []
    cover_y_highqual = []
    range_slices = []
    in_sv = []

    same_chrom_sv = bool(sv) and strip_chr(sv[0].chrm) == strip_chr(sv[-1].chrm)
    points_per_range = max(1, int(options.max_coverage_points / len(ranges)))

    for i in range(len(ranges)):
        r = ranges[i]
        chrm_coverage = None
        for chrm in coverage:
            if strip_chr(chrm) == strip_chr(r.chrm):
                chrm_coverage = coverage[chrm]
        values = get_range_coverage(chrm_coverage, r.start, r.end)
        centres, binned = bin_coverage(values, points_per_range)
        positions = r.start + centres

        first = len(cover_x)
        span = float(max(1, r.end - r.start))
        cover_x.extend((i + (positions - r.start) / span) / len(ranges))
        cover_y_highqual.extend(binned[:, 0])
        cover_y_lowqual.extend(binned[:, 1])
        range_slices.append(slice(first, len(cover_x)))
        if same_chrom_sv and strip_chr(sv[0].chrm) == strip_chr(r.chrm):
            in_sv.extend((positions >= sv[0].start) & (positions <= sv[-1].end))
        else:
            in_sv.extend(np.zeros(len(positions), dtype=bool))

    cover_x = np.array(cover_x)
    cover_y_lowqual = np.array(cover_y_lowqual)
    cover_y_highqual = np.array(cover_y_highqual)
    cover_y_all = cover_y_lowqual + cover_y_highqual
    in_sv = np.array(in_sv, dtype=bool)

    plotted_bases = sum(r.end - r.start for r in ranges)
    draw_trend = (
        options.coverage_tracktype is not None
        and options.coverage_tracktype != "none"
        and options.coverage_trend_min_size
        and plotted_bases >= options.coverage_trend_min_size
        and len(cover_y_all) >= 50
    )

    trend = None
    if draw_trend:
        trend = np.zeros(len(cover_y_all))
        for sl in range_slices:
            n_points = sl.stop - sl.start
            # ~2% of the range: wide enough to iron out mappability/GC
            # spikes, and a median so depth steps at breakpoints stay sharp
            trend[sl] = rolling_median(cover_y_all[sl], max(5, n_points // 50))

    if max_coverage > 0:
        max_plot_depth = max_coverage
    else:
        # ignore isolated spikes when choosing the scale
        max_plot_depth = np.percentile(cover_y_all, 99.5)
        if cover_y_all.max() <= 1.5 * max_plot_depth:
            max_plot_depth = cover_y_all.max()
        if trend is not None:
            max_plot_depth = max(max_plot_depth, 1.25 * trend.max())
        # both y axes get the same number of evenly spaced ticks so they
        # share gridlines
        max_plot_depth = nice_axis_top(max_plot_depth, 5 if hp_count == 1 else 2)

    ax2 = ax.twinx()
    ax2.set_xlim([0, 1])

    if 0 == max_plot_depth:
        max_plot_depth = 0.01

    ax2.set_ylim([0, max(1, max_plot_depth)])
    bottom_fill = np.zeros(len(cover_y_all))
    # each range is filled on its own so nothing is drawn across the gap
    # between two ranges
    for sl in range_slices:
        x = cover_x[sl]
        if options.coverage_tracktype == "stack":
            ax2.fill_between(
                x, cover_y_highqual[sl], bottom_fill[sl],
                color=GREY, step="mid", alpha=0.55, linewidth=0,
            )
            ax2.fill_between(
                x, cover_y_all[sl], cover_y_highqual[sl],
                color=GREY, step="mid", alpha=0.3, linewidth=0,
            )
        elif options.coverage_tracktype == "superimpose":
            ax2.fill_between(
                x, cover_y_lowqual[sl], bottom_fill[sl],
                color=GREY, step="mid", alpha=0.3, linewidth=0,
            )
            ax2.fill_between(
                x, cover_y_highqual[sl], cover_y_lowqual[sl],
                color=GREY, step="mid", alpha=0.55, linewidth=0,
            )
        ## options.coverage_tracktype==None also allowed

        if trend is not None:
            ax2.plot(x, trend[sl], "-", color=INK, lw=0.9, alpha=0.85, zorder=4)

    # mark the depth of the flanks and say how the variant compares to it
    if (
        options.coverage_tracktype is not None
        and options.coverage_tracktype != "none"
        and hp_count == 1
        and is_copy_number_type(options.sv_type)
        and in_sv.any()
        and (~in_sv).any()
    ):
        flank_depth = np.median(cover_y_all[~in_sv])
        inside_depth = np.median(cover_y_all[in_sv])
        if flank_depth > 0:
            ax2.axhline(
                flank_depth, color="dimgrey", linestyle="--", linewidth=0.8,
                alpha=0.7, zorder=3,
            )
            ax2.add_artist(
                AnchoredText(
                    "depth in variant / flanks: {:.2f}x".format(inside_depth / flank_depth),
                    loc="upper right",
                    prop=dict(size=6, color=MUTED_INK),
                    borderpad=0.2,
                    pad=0.2,
                    frameon=False,
                )
            )

    # the coverage axis sits underneath the read axis and carries the panel
    # background and gridlines
    tick_count = 6 if hp_count == 1 else 3
    ax2.yaxis.set_major_locator(ticker.LinearLocator(tick_count))
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%g"))
    ax2.tick_params(axis="y", colors=GREY, labelsize=options.yaxis_label_fontsize)
    ax2.set_axisbelow(True)
    ax2.grid(True, axis="y")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(axis="x", length=0)
    ax2.tick_params(axis="y", length=0)
    ax2.set_zorder(ax.get_zorder() - 1)
    ax.patch.set_visible(False)

    # break the variant plot when we have multiple ranges
    for i in range(1, len(ranges)):
        ax2.axvline(x=float(i) / len(ranges), color="white", linewidth=5, zorder=5)

    return ax2


# }}}

##Pair End methods
# {{{class PairedEnd:
class PairedEnd:
    """container of paired-end read info

    Contains start(int), end(int), strand(bool True=forward), MI (int
    molecular identifier), HP (int haplotype)
    """

    def __init__(self, chrm, start, end, is_reverse, MI_tag, HP_tag):
        """Create PairedEnd instance

        Genomic interval is defined by start and end integers
        Strand is opposite of is_reverse
        Molecular identifier and Haplotype are integers if present, else
        False
        """
        self.pos = genome_interval(chrm, start, end)
        self.strand = not (is_reverse)
        # molecular identifier - linked reads only
        self.MI = None
        # haplotype - phased reads only
        self.HP = 0

        if MI_tag:
            self.MI = MI_tag
        if HP_tag:
            self.HP = HP_tag

    def __repr__(self):
        return "PairedEnd(%s,%s,%s,%s,%s,%s)" % (
            self.pos.chrm,
            self.pos.start,
            self.pos.end,
            self.strand,
            self.MI,
            self.HP,
        )


# }}}

# {{{ def add_pair_end(bam_file, read, pairs, linked_reads):
def add_pair_end(bam_file, read, pairs, linked_reads, ignore_hp):
    """adds a (mapped, primary, non-supplementary, and paired) read to the
    pairs list

    Pysam read is added as simpified PairedEnd instance to pairs
    Also added to linked_reads list if there is an associated MI tag
    """

    if read.is_unmapped:
        return
    if not (read.is_paired):
        return
    if read.is_secondary:
        return
    if read.is_supplementary:
        return

    MI_tag = False
    HP_tag = False

    if read.has_tag("MI"):
        MI_tag = int(read.get_tag("MI"))
    if not ignore_hp and read.has_tag("HP"):
        HP_tag = int(read.get_tag("HP"))

    pe = PairedEnd(
        bam_file.get_reference_name(read.reference_id),
        read.reference_start,
        read.reference_end,
        read.is_reverse,
        MI_tag,
        HP_tag,
    )

    if pe.HP not in pairs:
        pairs[pe.HP] = {}

    if read.query_name not in pairs[pe.HP]:
        pairs[pe.HP][read.query_name] = []

    if pe.MI:
        if pe.HP not in linked_reads:
            linked_reads[pe.HP] = {}

        if pe.MI not in linked_reads[pe.HP]:
            linked_reads[pe.HP][pe.MI] = [[], []]
        linked_reads[pe.HP][pe.MI][0].append(read.query_name)

    pairs[pe.HP][read.query_name].append(pe)
    pairs[pe.HP][read.query_name].sort(key=lambda x: x.pos.start)


# }}}

# {{{def sample_normal(max_depth, pairs, z):
def sample_normal(max_depth, pairs, z):
    """Downsamples paired-end reads 
    
    Selects max_depth reads
    Does not remove discordant pairs, those with insert distance greater
    than z stdevs from mean

    Returns downsampled pairs list
    """

    sampled_pairs = {}
    plus_minus_pairs = {}

    if max_depth == 0:
        return sampled_pairs

    for read_name in pairs:
        pair = pairs[read_name]
        if len(pair) != 2:
            continue
        if (pair[0].strand == True and pair[1].strand == False) or is_overlapping_pair(pair):
            plus_minus_pairs[read_name] = pair
        else:
            sampled_pairs[read_name] = pair

    if len(plus_minus_pairs) > max_depth:
        lens = np.array(
            [pair[1].pos.end - pair[0].pos.start for pair in plus_minus_pairs.values()]
        )
        mean = np.mean(lens)
        stdev = np.std(lens)

        inside_norm = {}

        for read_name in pairs:
            pair = pairs[read_name]
            if len(pair) != 2:
                continue
            if pair[1].pos.end - pair[0].pos.start >= mean + z * stdev:
                sampled_pairs[read_name] = pair
            else:
                inside_norm[read_name] = pair

        if len(inside_norm) > max_depth:
            for read_name in random.sample(list(inside_norm.keys()), max_depth):
                sampled_pairs[read_name] = inside_norm[read_name]
        else:
            for read_name in inside_norm:
                sampled_pairs[read_name] = inside_norm[read_name]
    else:
        for read_name in plus_minus_pairs:
            sampled_pairs[read_name] = plus_minus_pairs[read_name]

    return sampled_pairs


# }}}

# {{{def get_pairs_insert_sizes(pairs):
def get_pairs_insert_sizes(ranges, pairs):
    """Extracts the integer insert sizes for all pairs

    Return list of integer insert sizes
    """
    pair_insert_sizes = []

    for hp in pairs:
        for read_name in pairs[hp]:
            if len(pairs[hp][read_name]) == 2:
                size = get_pair_insert_size(ranges, pairs[hp][read_name])

                if size:
                    pair_insert_sizes.append(size)

    return pair_insert_sizes


# }}}

# {{{def get_pair_insert_size(ranges, pair):
def get_pair_insert_size(ranges, pair):
    """ Gives the outer distance
    """
    first = pair[0]
    second = pair[1]

    # make sure both sides are in range
    if (
        get_range_hit(ranges, first.pos.chrm, first.pos.start) != None
        or get_range_hit(ranges, first.pos.chrm, first.pos.end) != None
    ) and (
        get_range_hit(ranges, second.pos.chrm, second.pos.start) != None
        or get_range_hit(ranges, second.pos.chrm, second.pos.end) != None
    ):

        if first.pos.chrm == second.pos.chrm:
            return abs(second.pos.end - first.pos.start)
        else:
            return INTERCHROM_YAXIS
    else:
        return None


# }}}

# {{{ def get_pairs_plan(ranges, pairs, linked_plan=False):
def get_pairs_plan(ranges, pairs, linked_plan=False):
    steps = []
    max_event = 0

    insert_sizes = []

    for read_name in pairs:
        pair = pairs[read_name]

        plan = get_pair_plan(ranges, pair)

        if plan:
            insert_size, step = plan
            insert_sizes.append(insert_size)
            steps.append(step)

    if len(insert_sizes) > 0:
        max_event = max(insert_sizes)

    plan = [max_event, steps]

    return plan


# }}}

# {{{def get_pair_plan(ranges, pair, linked_plan=False):
def get_pair_plan(ranges, pair, linked_plan=False):
    if pair == None or len(pair) != 2:
        return None

    first = pair[0]
    second = pair[1]

    # see if they are part of a linked read
    if not linked_plan and (first.MI or second.MI):
        return None

    # make sure both ends are in the plotted region
    first_s_hit = get_range_hit(ranges, first.pos.chrm, first.pos.start)
    first_e_hit = get_range_hit(ranges, first.pos.chrm, first.pos.end)
    second_s_hit = get_range_hit(ranges, second.pos.chrm, second.pos.start)
    second_e_hit = get_range_hit(ranges, second.pos.chrm, second.pos.end)

    if (first_s_hit == None and first_e_hit == None) or (
        second_s_hit == None and second_e_hit == None
    ):
        return None

    insert_size = get_pair_insert_size(ranges, pair)

    first_hit = first_s_hit if first_s_hit != None else first_e_hit
    second_hit = second_e_hit if second_e_hit != None else second_s_hit

    start = genome_interval(
        first.pos.chrm,
        max(first.pos.start, ranges[first_hit].start),
        max(first.pos.start, ranges[first_hit].start),
    )

    end = genome_interval(
        second.pos.chrm,
        min(second.pos.end, ranges[second_hit].end),
        min(second.pos.end, ranges[second_hit].end),
    )

    step = plan_step(start, end, "PAIREND")

    event_type = get_pair_event_type(pair)
    step.info = {"TYPE": event_type, "INSERTSIZE": insert_size}

    return insert_size, step


# }}}

def is_overlapping_pair(pair):
    """True for a -/+ pair whose two reads overlap each other.

    When a fragment is shorter than the read length the reverse read can
    start at or before the forward read, so sorted by position the pair
    looks everted like duplication evidence. It is a normal short fragment:
    real everted pairs are separated by the size of the duplication.
    """
    first, second = pair[0], pair[1]
    return (
        first.strand == False
        and second.strand == True
        and first.pos.chrm == second.pos.chrm
        and second.pos.start <= first.pos.end
    )


# {{{def get_pair_event_type(pe_read):
def get_pair_event_type(pe_read):
    """Decide what type of event the read supports (del/normal, dup, inv)
    """
    event_by_strand = {
        (True, False): "Deletion/Normal",
        (False, True): "Duplication",
        (False, False): "Inversion",
        (True, True): "Inversion",
    }
    if is_overlapping_pair(pe_read):
        return "Deletion/Normal"
    event_type = event_by_strand[pe_read[0].strand, pe_read[1].strand]
    return event_type


# }}}

def jitter(value, bounds: float = 0.1) -> float:
    """
    Offset value by a random value within the defined bounds
    """
    assert 0.0 <= bounds < 1.0
    return value * (1 + bounds * random.uniform(-1, 1))


class ReadBatch:
    """Collects the lines for reads so each style is drawn in one go.

    One matplotlib artist per read is slow to build and to render when there
    are thousands of them; here every line with the same style goes into a
    single LineCollection (plus one marker-only line for the end markers).
    """

    def __init__(self):
        self.segments = {}

    def add(self, p, y, color, linestyle="-", lw=0.5, alpha=0.25, marker=None, zorder=2):
        style = (color, linestyle, lw, alpha, marker, zorder)
        self.segments.setdefault(style, []).append(((p[0], y[0]), (p[1], y[1])))

    def draw(self, ax, marker_size):
        for style, segments in self.segments.items():
            color, linestyle, lw, alpha, marker, zorder = style
            ax.add_collection(
                LineCollection(
                    segments,
                    colors=color,
                    linestyles=linestyle,
                    linewidths=lw,
                    alpha=alpha,
                    zorder=zorder,
                )
            )
            if marker:
                ax.plot(
                    [point[0] for segment in segments for point in segment],
                    [point[1] for segment in segments for point in segment],
                    linestyle="none",
                    marker=marker,
                    markersize=marker_size,
                    color=color,
                    alpha=alpha,
                    zorder=zorder,
                )
        self.segments = {}


# {{{def plot_pair_plan(ranges, step, ax):
def plot_pair_plan(ranges, step, ax, marker_size, jitter_bounds, batch=None):
    p = [
        map_genome_point_to_range_points(
            ranges, step.start_pos.chrm, step.start_pos.start
        ),
        map_genome_point_to_range_points(ranges, step.end_pos.chrm, step.end_pos.end),
    ]

    if None in p:
        return False

    # some points are far outside of the printable area, so we ignore them
    if not points_in_window(p):
        return False

    READ_TYPES_USED["Paired-end read"] = True

    y = step.info["INSERTSIZE"]

    # Offset y-values using jitter to avoid overlapping lines
    y = jitter(y, bounds=jitter_bounds)

    event_type = step.info["TYPE"]
    READ_TYPES_USED[event_type] = True
    color = COLORS[event_type]

    # plot the individual pair (drawn here unless the caller is batching)
    own_batch = batch is None
    if own_batch:
        batch = ReadBatch()
    batch.add(p, [y, y], color, linestyle="-", lw=0.5, marker="s", zorder=10)
    if own_batch:
        batch.draw(ax, marker_size)

    return True


# }}}

# {{{def plot_pairs(pairs,
def plot_pairs(
    pairs, ax, ranges, curr_min_insert_size, curr_max_insert_size, marker_size, jitter_bounds,
):
    """Plots all PairedEnd reads for the region
    """

    plan = get_pairs_plan(ranges, pairs)

    if not plan:
        [curr_min_insert_size, curr_max_insert_size]

    max_event, steps = plan

    batch = ReadBatch()
    for step in steps:
        plot_pair_plan(ranges, step, ax, marker_size, jitter_bounds, batch)
    batch.draw(ax, marker_size)

    if not curr_min_insert_size or curr_min_insert_size > max_event:
        curr_min_insert_size = max_event
    if not curr_max_insert_size or curr_max_insert_size < max_event:
        curr_max_insert_size = max_event

    return [curr_min_insert_size, curr_max_insert_size]


# }}}

##Split Read methods
# {{{class SplitRead:
class SplitRead:
    """container of split read info

    Contains start(int), end(int), strand(bool True=forward), query
    position (int), MI (int molecular identifier), HP (int haplotype)
    """

    def __init__(self, chrm, start, end, strand, query_pos, MI_tag=None, HP_tag=None):
        """Create SplitRead instance

        Genomic interval is defined by start, end, and query_pos integers
        Strand is opposite of is_reverse
        Molecular identifier and Haplotype are integers if present, else
        False
        """
        self.pos = genome_interval(chrm, start, end)
        self.strand = strand
        self.query_pos = query_pos
        # molecular identifier - linked reads only
        self.MI = None
        # haplotype - phased reads only
        self.HP = 0

        if MI_tag:
            self.MI = MI_tag
        if HP_tag:
            self.HP = HP_tag

    def __repr__(self):
        return "SplitRead(%s,%s,%s,%s,%s,%s,%s)" % (
            self.pos.chrm,
            self.pos.start,
            self.pos.end,
            self.strand,
            self.query_pos,
            self.MI,
            self.HP,
        )


# }}}

# {{{def calc_query_pos_from_cigar(cigar, strand):
def calc_query_pos_from_cigar(cigar, strand):
    """Uses the CIGAR string to determine the query position of a read

    The cigar arg is a string like the following: 86M65S
    The strand arg is a boolean, True for forward strand and False for
    reverse

    Returns pair of ints for query start, end positions
    """

    cigar_ops = [[int(op[0]), op[1]] for op in re.findall(r"(\d+)([A-Za-z])", cigar)]

    order_ops = cigar_ops
    if not strand:  # - strand
        order_ops = order_ops[::-1]

    qs_pos = 0
    qe_pos = 0
    q_len = 0

    for op_position in range(len(cigar_ops)):
        op_len = cigar_ops[op_position][0]
        op_type = cigar_ops[op_position][1]

        if op_position == 0 and (op_type == "H" or op_type == "S"):
            qs_pos += op_len
            qe_pos += op_len
            q_len += op_len
        elif op_type == "H" or op_type == "S":
            q_len += op_len
        elif op_type == "M" or op_type == "I" or op_type == "X":
            qe_pos += op_len
            q_len += op_len

    return qs_pos, qe_pos


# }}}

# {{{def add_split(read, splits, bam_file, linked_reads):
def add_split(read, splits, bam_file, linked_reads, ignore_hp):
    """adds a (primary, non-supplementary) read to the splits list

    Pysam read is added as simpified SplitRead instance to splits
    Also added to linked_reads list if there is an associated MI tag
    """
    if read.is_secondary:
        return
    if read.is_supplementary:
        return
    if not read.has_tag("SA"):
        return

    qs_pos, qe_pos = calc_query_pos_from_cigar(read.cigarstring, (not read.is_reverse))

    HP_tag = False
    MI_tag = False
    if read.has_tag("MI"):
        MI_tag = int(read.get_tag("MI"))

    if not ignore_hp and read.has_tag("HP"):
        HP_tag = int(read.get_tag("HP"))
    sr = SplitRead(
        bam_file.get_reference_name(read.reference_id),
        read.reference_start,
        read.reference_end,
        not (read.is_reverse),
        qs_pos,
        MI_tag,
        HP_tag,
    )

    if sr.MI:
        if sr.HP not in linked_reads:
            linked_reads[sr.HP] = {}
        if sr.MI not in linked_reads[sr.HP]:
            linked_reads[sr.HP][sr.MI] = [[], []]
        linked_reads[sr.HP][sr.MI][1].append(read.query_name)

    if sr.HP not in splits:
        splits[sr.HP] = {}

    splits[sr.HP][read.query_name] = [sr]

    for sa in read.get_tag("SA").split(";"):
        if len(sa) == 0:
            continue
        A = sa.split(",")
        chrm = A[0]
        pos = int(A[1])
        strand = A[2] == "+"
        cigar = A[3]
        #mapq and nm are never used, annotating this for code readability 
        mapq = int(A[4])
        nm = int(A[5])
        qs_pos, qe_pos = calc_query_pos_from_cigar(cigar, strand)
        splits[sr.HP][read.query_name].append(
            SplitRead(chrm, pos, pos + qe_pos, strand, qs_pos)
        )

    if len(splits[sr.HP][read.query_name]) == 1:
        del splits[sr.HP][read.query_name]
    else:
        splits[sr.HP][read.query_name].sort(key=lambda x: x.pos.start)


# }}}


# {{{def get_split_plan(ranges, split):
def get_split_plan(ranges, split, linked_plan=False):
    """
    There can be 2 or more alignments in a split. Plot only those that are in a
    range, and set the insert size to be the largest gap

    A split read acts like a long read, so we will covert the split read
    to a long read, then convert the long read plan back to a split read plan
    """

    alignments = []
    for s in split:
        # see if they are part of a linked read
        if not linked_plan and (s.MI):
            return None
        alignment = Alignment(s.pos.chrm, s.pos.start, s.pos.end, s.strand, s.query_pos)
        alignments.append(alignment)

    long_read = LongRead(alignments)
    long_reads = {}
    long_reads["convert"] = [long_read]
    plan = get_long_read_plan("convert", long_reads, ranges)

    if not plan:
        return None

    max_gap, lr_steps = plan

    if len(lr_steps) < 3:
        return None

    sr_steps = []

    # a split read will include 3 long read steps, align, event, align
    for i in range(0, len(lr_steps), 2):
        if i + 2 > len(lr_steps):
            break
        if (
            lr_steps[i].info["TYPE"] == "Align"
            and lr_steps[i + 1].info["TYPE"] != "Align"
            and lr_steps[i + 2].info["TYPE"] == "Align"
        ):
            start = genome_interval(
                lr_steps[i].end_pos.chrm,
                lr_steps[i].end_pos.end,
                lr_steps[i].end_pos.end,
            )
            end = genome_interval(
                lr_steps[i + 2].start_pos.chrm,
                lr_steps[i + 2].start_pos.start,
                lr_steps[i + 2].start_pos.start,
            )
            sr_steps.append(
                plan_step(
                    start,
                    end,
                    "SPLITREAD",
                    info={"TYPE": lr_steps[i + 1].info["TYPE"], "INSERTSIZE": max_gap},
                )
            )
    return max_gap, sr_steps


# }}}

# {{{def get_splits_plan(ranges, splits, linked_plan=False):
def get_splits_plan(ranges, splits, linked_plan=False):
    steps = []
    max_event = 0

    insert_sizes = []

    for read_name in splits:
        split = splits[read_name]

        plan = get_split_plan(ranges, split)

        if plan:
            insert_size, step = plan
            insert_sizes.append(insert_size)
            steps += step

    if len(insert_sizes) > 0:
        max_event = max(insert_sizes)

    plan = [max_event, steps]

    return plan


# }}}


# {{{def plot_split(split, y, ax, ranges):
def plot_split_plan(ranges, step, ax, marker_size, jitter_bounds, batch=None):
    p = [
        map_genome_point_to_range_points(
            ranges, step.start_pos.chrm, step.start_pos.start
        ),
        map_genome_point_to_range_points(ranges, step.end_pos.chrm, step.end_pos.end),
    ]

    if None in p:
        return False

    # some points are far outside of the printable area, so we ignore them
    if not points_in_window(p):
        return False

    READ_TYPES_USED["Split-read"] = True

    y = step.info["INSERTSIZE"]

    # Offset y-values using jitter to avoid overlapping lines
    y = jitter(y, bounds=jitter_bounds)

    event_type = step.info["TYPE"]
    READ_TYPES_USED[event_type] = True
    color = COLORS[event_type]

    own_batch = batch is None
    if own_batch:
        batch = ReadBatch()
    batch.add(p, [y, y], color, linestyle=":", lw=1, marker="o")
    if own_batch:
        batch.draw(ax, marker_size)


# }}}

# {{{def plot_splits(splits,
def plot_splits(
    splits, ax, ranges, curr_min_insert_size, curr_max_insert_size, marker_size, jitter_bounds,
):
    """Plots all SplitReads for the region
    """
    plan = get_splits_plan(ranges, splits)

    if not plan:
        [curr_min_insert_size, curr_max_insert_size]

    max_event, steps = plan

    batch = ReadBatch()
    for step in steps:
        plot_split_plan(ranges, step, ax, marker_size, jitter_bounds, batch)
    batch.draw(ax, marker_size)

    if not curr_min_insert_size or curr_min_insert_size > max_event:
        curr_min_insert_size = max_event
    if not curr_max_insert_size or curr_max_insert_size < max_event:
        curr_max_insert_size = max_event

    return [curr_min_insert_size, curr_max_insert_size]


# }}}

##Long Read methods
# {{{class Alignment:
class Alignment:
    """container of alignment info, from CIGAR string

    Contains start(int), end(int), strand(bool True=forward), query
    position (int)
    """

    def __init__(self, chrm, start, end, strand, query_position):
        """Create Alignment instance

        Genomic interval is defined by start, end, and query_pos integers
        Strand is bool (True for forward)
        """
        self.pos = genome_interval(chrm, start, end)
        self.strand = strand
        self.query_position = query_position

    def __str__(self):
        return ",".join(
            [
                str(x)
                for x in [
                    self.pos.chrm,
                    self.pos.start,
                    self.pos.end,
                    self.strand,
                    self.query_position,
                ]
            ]
        )

    def __repr__(self):
        return "Alignment(%s,%s,%s,%s,%s)" % (
            self.pos.chrm,
            self.pos.start,
            self.pos.end,
            self.strand,
            self.query_position,
        )


# }}}

# {{{class LongRead:
class LongRead:
    """container of LongRead info

    Contains start(int), end(int), list of Alignments
    """

    def __init__(self, alignments):
        """Create LongRead instance

        Genomic interval is defined by start, end integers
        List of Alignments set by parameter
        """
        self.alignments = alignments

    def __str__(self):
        return ",".join([str(x) for x in self.alignments])

    def __repr__(self):
        return "LongRead(" + str(self) + ")"


# }}}

# {{{def get_alignments_from_cigar(chrm,
def get_alignments_from_cigar(chrm, curr_pos, strand, cigartuples, reverse=False):
    """Breaks CIGAR string into individual Aignments

    Starting point within genome given by curr_pos and strand
    Set of CIGAR operations and lengths as pairs passed in as cigartuples
    Direction of alignment set to reverse with reverse boolean

    Return list of Alignments
    """
    alignments = []
    q_pos = 0
    if reverse:
        cigartuples = cigartuples[::-1]

    for op, length in cigartuples:
        if op in [CIGAR_MAP["M"], CIGAR_MAP["="], CIGAR_MAP["X"]]:
            alignments.append(
                Alignment(chrm, curr_pos, curr_pos + length, strand, q_pos)
            )
            curr_pos += length
            q_pos += length
        elif op == CIGAR_MAP["I"]:
            q_pos += length
        elif op == CIGAR_MAP["D"]:
            curr_pos += length
        elif op == CIGAR_MAP["N"]:
            curr_pos += length
        elif op == CIGAR_MAP["S"]:
            q_pos += length
    return alignments


# }}}

# {{{def get_cigartuples_from_string(cigarstring):
def get_cigartuples_from_string(cigarstring):
    """Extracts operations,lengths as tuples from cigar string"

    Returns list of tuples of [operation,length]
    """
    cigartuples = []
    for match in re.findall(r"(\d+)([A-Z]{1})", cigarstring):
        length = int(match[0])
        op = match[1]
        cigartuples.append((CIGAR_MAP[op], length))

    return cigartuples


# }}}

# {{{def merge_alignments(min_gap, alignments):
def merge_alignments(min_gap, alignments):
    """Combines previously identified alignments if close together
    Alignments are combined if within min_gap distance
    Returns list of Alignments
    """

    merged_alignments = []

    for alignment in alignments:
        if len(merged_alignments) == 0:
            merged_alignments.append(alignment)
        else:
            if (
                alignment.pos.chrm == merged_alignments[-1].pos.chrm
                and alignment.pos.start < merged_alignments[-1].pos.end + min_gap
            ):
                merged_alignments[-1].pos.end = alignment.pos.end
            else:
                merged_alignments.append(alignment)
    return merged_alignments


# }}}

# {{{def add_long_reads(bam_file, read, long_reads, min_event_size):
def add_long_reads(bam_file, read, long_reads, min_event_size, ignore_hp):
    """Adds a (primary, non-supplementary, long) read to the long_reads list

    Read added to long_reads if within the inteval defined by ranges
    Alignments belonging to the LongRead instance combined if within the
    min_event_size distance apart
    """
    if read.is_supplementary or read.is_secondary:
        return

    hp = 0

    if not ignore_hp and read.has_tag("HP"):
        hp = int(read.get_tag("HP"))

    alignments = get_alignments_from_cigar(
        bam_file.get_reference_name(read.reference_id),
        read.pos,
        not read.is_reverse,
        read.cigartuples,
    )

    min_gap = min_event_size
    merged_alignments = merge_alignments(min_gap, alignments)

    read_strand = not read.is_reverse

    if read.has_tag("SA"):
        for sa in read.get_tag("SA").split(";"):
            if len(sa) == 0:
                continue

            rname, pos, strand, cigar, mapq, nm = sa.split(",")

            sa_pos = int(pos)
            sa_strand = strand == "+"
            strand_match = read_strand != sa_strand
            sa_cigartuples = get_cigartuples_from_string(cigar)
            sa_alignments = get_alignments_from_cigar(
                rname, sa_pos, sa_strand, sa_cigartuples, reverse=strand_match
            )

            sa_merged_alignments = merge_alignments(min_gap, sa_alignments)

            if len(sa_merged_alignments) > 0:
                merged_alignments += sa_merged_alignments

    if hp not in long_reads:
        long_reads[hp] = {}

    if read.query_name not in long_reads[hp]:
        long_reads[hp][read.query_name] = []

    long_reads[hp][read.query_name].append(LongRead(merged_alignments))


# }}}

# {{{def add_align_step(alignment, steps, ranges):
def add_align_step(alignment, steps, ranges):
    # alignment can span ranges
    start_range_hit_i = get_range_hit(ranges, alignment.pos.chrm, alignment.pos.start)
    end_range_hit_i = get_range_hit(ranges, alignment.pos.chrm, alignment.pos.end)

    # neither end is in range, add nothing
    if start_range_hit_i == None and end_range_hit_i == None:
        return

    # start is not in range, use end hit
    if start_range_hit_i == None:
        start = genome_interval(
            alignment.pos.chrm,
            max(alignment.pos.start, ranges[end_range_hit_i].start),
            max(alignment.pos.start, ranges[end_range_hit_i].start),
        )
        end = genome_interval(
            alignment.pos.chrm,
            min(alignment.pos.end, ranges[end_range_hit_i].end),
            min(alignment.pos.end, ranges[end_range_hit_i].end),
        )
        steps.append(plan_step(start, end, "LONGREAD", info={"TYPE": "Align"}))
    # end is not in range, use start hit
    elif end_range_hit_i == None:
        start = genome_interval(
            alignment.pos.chrm,
            max(alignment.pos.start, ranges[start_range_hit_i].start),
            max(alignment.pos.start, ranges[start_range_hit_i].start),
        )
        end = genome_interval(
            alignment.pos.chrm,
            min(alignment.pos.end, ranges[start_range_hit_i].end),
            min(alignment.pos.end, ranges[start_range_hit_i].end),
        )
        steps.append(plan_step(start, end, "LONGREAD", info={"TYPE": "Align"}))
    # both are in the same range
    elif start_range_hit_i == end_range_hit_i:
        start = genome_interval(
            alignment.pos.chrm,
            max(alignment.pos.start, ranges[start_range_hit_i].start),
            max(alignment.pos.start, ranges[start_range_hit_i].start),
        )
        end = genome_interval(
            alignment.pos.chrm,
            min(alignment.pos.end, ranges[end_range_hit_i].end),
            min(alignment.pos.end, ranges[end_range_hit_i].end),
        )
        steps.append(plan_step(start, end, "LONGREAD", info={"TYPE": "Align"}))
    # in different ranges
    else:
        start_1 = genome_interval(
            alignment.pos.chrm,
            max(alignment.pos.start, ranges[start_range_hit_i].start),
            max(alignment.pos.start, ranges[start_range_hit_i].start),
        )
        end_1 = genome_interval(
            alignment.pos.chrm,
            ranges[start_range_hit_i].end,
            ranges[start_range_hit_i].end,
        )
        steps.append(plan_step(start_1, end_1, "LONGREAD", info={"TYPE": "Align"}))

        start_2 = genome_interval(
            alignment.pos.chrm,
            ranges[end_range_hit_i].start,
            ranges[end_range_hit_i].start,
        )
        end_2 = genome_interval(
            alignment.pos.chrm,
            min(alignment.pos.end, ranges[end_range_hit_i].end),
            min(alignment.pos.end, ranges[end_range_hit_i].end),
        )
        steps.append(plan_step(start_2, end_2, "LONGREAD", info={"TYPE": "Align"}))


# }}}

# {{{def get_long_read_plan(read_name, long_reads, ranges):
def get_long_read_plan(read_name, long_reads, ranges):
    """Create a plan to render a long read

    Plan consists of the largest event within the read 
        (used to determine the y-axis position of read)
        and the alignment types for plotting each Alignment within 
        LongRead.alignments Align, Duplication, Deletion, Inversion,
        Inversion,
        InterChrmInversion, InterChrm

    Returns plan
    """

    alignments = []

    # only keep alignments that intersect a range
    seen = {}

    if read_name not in long_reads:
        raise SamplotError("Read name {} not in list of long reads".format(read_name))

    for long_read in long_reads[read_name]:
        for alignment in long_read.alignments:
            if alignment.query_position in seen:
                continue
            seen[alignment.query_position] = 1
            # check to see if any part of this alignment overlaps a plot
            # range
            in_range = False
            for r in ranges:
                if r.intersect(alignment.pos) == 0:
                    in_range = True
            if in_range:
                alignments.append(alignment)

    if len(alignments) <= 0:
        return None
    alignments.sort(key=lambda x: x.query_position)

    # we set the primary strand to be the one with the longest alignment
    # this will affect which alignment is inverted. There are clearly edge
    # cases here that we will need to address as we get more examples
    # of inversions
    longest_alignment = 0
    longest_alignment_i = -1
    for i in range(len(alignments)):
        l = alignments[i].pos.end - alignments[i].pos.start
        if longest_alignment < l:
            longest_alignment = l
            longest_alignment_i = i
    primary_strand = alignments[longest_alignment_i].strand

    steps = []
    # long aglinments may spill over the edges, so we will clip that starts
    curr = alignments[0]

    add_align_step(curr, steps, ranges)

    for i in range(1, len(alignments)):
        last = alignments[i - 1]
        curr = alignments[i]

        # figure out what the event is

        # INTER CHROM
        if curr.pos.chrm != last.pos.chrm:
            if curr.strand != last.strand:
                start = genome_interval(last.pos.chrm, last.pos.end, last.pos.end)

                end = genome_interval(curr.pos.chrm, curr.pos.end, curr.pos.end)

                info = {"TYPE": "InterChrmInversion"}
                steps.append(plan_step(start, end, "LONGREAD", info=info))
            else:
                start = genome_interval(last.pos.chrm, last.pos.end, last.pos.end)
                end = genome_interval(curr.pos.chrm, curr.pos.start, curr.pos.start)
                info = {"TYPE": "InterChrm"}
                steps.append(plan_step(start, end, "LONGREAD", info=info))

            add_align_step(curr, steps, ranges)
        # Inversion
        elif curr.strand != last.strand:
            # it is possible that we have a complex even that
            # is an inverted Duplication
            if curr.pos.start < last.pos.end:
                start = genome_interval(last.pos.chrm, last.pos.end, last.pos.end)
                end = genome_interval(curr.pos.chrm, curr.pos.start, curr.pos.start)
                info = {"TYPE": "Deletion"}
                steps.append(plan_step(start, end, "LONGREAD", info=info))
            if curr.strand != primary_strand:
                # last (primary) | curr
                # +++++++++++++++|-------
                #               ^.......^
                #             end           end

                # last (primary) | curr
                # ---------------|+++++++
                #               ^.......^
                #             end           end

                start = genome_interval(last.pos.chrm, last.pos.end, last.pos.end)
                end = genome_interval(curr.pos.chrm, curr.pos.end, curr.pos.end)
                info = {"TYPE": "Inversion"}
                steps.append(plan_step(start, end, "LONGREAD", info=info))
            else:
                if curr.pos.start < last.pos.end:
                    start = genome_interval(last.pos.chrm, last.pos.end, last.pos.end)
                    end = genome_interval(curr.pos.chrm, curr.pos.start, curr.pos.start)
                    info = {"TYPE": "Duplication"}
                    steps.append(plan_step(start, end, "LONGREAD", info=info))

                # last   | curr (primary)
                # +++++++|-------------
                # ^.......^
                # start   start

                # last   | curr (primary)
                # -------|+++++++++++++++
                # ^.......^
                # start   start

                start = genome_interval(last.pos.chrm, last.pos.start, last.pos.start)
                end = genome_interval(curr.pos.chrm, curr.pos.start, curr.pos.start)
                info = {"TYPE": "Inversion"}
                steps.append(plan_step(start, end, "LONGREAD", info=info))

            add_align_step(curr, steps, ranges)
        # Duplication
        elif curr.pos.start < last.pos.end:
            start = genome_interval(last.pos.chrm, last.pos.end, last.pos.end)
            end = genome_interval(curr.pos.chrm, curr.pos.start, curr.pos.start)
            info = {"TYPE": "Duplication"}
            steps.append(plan_step(start, end, "LONGREAD", info=info))
            add_align_step(curr, steps, ranges)
        # Deletion
        else:
            start = genome_interval(last.pos.chrm, last.pos.end, last.pos.end)
            end = genome_interval(curr.pos.chrm, curr.pos.start, curr.pos.start)
            info = {"TYPE": "Deletion"}
            # steps.append(plan_step(start, end, 'LONGREAD', info=info))
            steps.append(plan_step(start, end, "LONGREAD", info={"TYPE": "Deletion"}))
            add_align_step(curr, steps, ranges)

        # if either end is in a range, then add its gap to the list

    max_gap = None

    chrms = set([s.start_pos.chrm for s in steps] + [s.end_pos.chrm for s in steps])

    # set interchrm dist to 5000
    if len(chrms) > 1:
        max_gap = INTERCHROM_YAXIS
    else:
        step_sizes = [
            abs(step.end_pos.end - step.start_pos.start)
            for step in steps
            if step.info["TYPE"] != "Align"
            and get_range_hit(ranges, step.start_pos.chrm, step.start_pos.start) != None
            and get_range_hit(ranges, step.end_pos.chrm, step.end_pos.end) != None
        ]

        max_gap = max(step_sizes) if len(step_sizes) > 0 else 0

    plan = [max_gap, steps]

    return plan


# }}}


##Variant methods
def sv_color(sv_type):
    """Colour for the variant bar and breakpoint markers, matching the
    colour of the read evidence that supports that kind of variant"""
    sv_type = str(sv_type or "").upper()
    if "DUP" in sv_type or "GAIN" in sv_type:
        return ORANGE
    if "INV" in sv_type:
        return BLUE
    return MUTED_INK


def format_size(size):
    """Human readable size of a variant in bp/kb/Mb"""
    if size > 1000000:
        return "{0:0.2f} Mb".format(size / 1000000.0)
    if size > 1000:
        return "{0:0.2f} kb".format(size / 1000.0)
    return "{} bp".format(int(size))


def get_sv_title(sv, sv_type):
    """Title for the plot, eg 'DEL · chr4:115,928,730-115,931,875 · 3.15 kb'"""
    if sv[0].chrm == sv[-1].chrm:
        sv_size = float(sv[0].end) - float(sv[0].start)
        if len(sv) > 1:
            sv_size = abs(int(float(sv[0].end) - float(sv[-1].start)))
        return "{} \u00b7 {}:{:,}-{:,} \u00b7 {}".format(
            sv_type, sv[0].chrm, sv[0].start, sv[-1].end, format_size(sv_size)
        )
    return "{} \u00b7 {}:{:,} / {}:{:,}".format(
        sv_type, sv[0].chrm, sv[0].start, sv[-1].chrm, sv[-1].end
    )


def get_breakpoint_points(sv, ranges):
    """Plot x positions (0-1) of the two breakpoints, None if not in view"""
    return [
        map_genome_point_to_range_points(ranges, sv[0].chrm, sv[0].start),
        map_genome_point_to_range_points(ranges, sv[-1].chrm, sv[-1].end),
    ]


def clean_axis(ax):
    """An annotation style axis: white, no frame, ticks or gridlines"""
    ax.set_xlim([0, 1])
    ax.set_facecolor("white")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_variant(sv, sv_type, ax, ranges):
    """Plots the variant bar at the top of the image

    """

    r = get_breakpoint_points(sv, ranges)
    color = sv_color(sv_type)

    if None not in r:
        ax.plot(r, [0, 0], "-", color=color, lw=7, solid_capstyle="butt", alpha=0.75)

    clean_axis(ax)
    ax.set_ylim([-1, 1])
    ax.set_title(
        get_sv_title(sv, sv_type), fontsize=10, loc="left", color=INK, pad=4
    )


# }}}

# {{{def plot_confidence_interval(chrm, breakpoint,ci, ax, ranges):
def plot_confidence_interval(chrm, breakpoint, ci, ax, ranges):
    """Plots a confidence interval on the variant bar
    """

    r = [
        map_genome_point_to_range_points(ranges, chrm, breakpoint - int(ci[0])),
        map_genome_point_to_range_points(ranges, chrm, breakpoint + int(ci[1])),
    ]
    if None in r:
        # confidence intervals are invalid
        return

    ax.plot(r, [0, 0], "-", color=INK, lw=0.6, alpha=1)
    ax.axvline(r[0], color=INK, lw=0.6, alpha=1, ymin=0.25, ymax=0.75)
    ax.axvline(r[1], color=INK, lw=0.6, alpha=1, ymin=0.25, ymax=0.75)


# }}}

# {{{def create_variant_plot(grid,
def create_variant_plot(grid, ax_i, sv, sv_type, ranges, start_ci, end_ci):
    """Plots the pieces of the variant bar at the top, including bar and
    confidence intervals 
    """
    ax = grid.figure.add_subplot(grid[ax_i])
    plot_variant(sv, sv_type, ax, ranges)
    ax_i += 1
    # plot confidence intervals if provided
    if start_ci and start_ci != None:
        plot_confidence_interval(sv[0].chrm, sv[0].start, start_ci, ax, ranges)
    if end_ci and end_ci != None:
        plot_confidence_interval(sv[-1].chrm, sv[-1].end, end_ci, ax, ranges)

    # break the variant plot when we have multiple ranges
    for i in range(1, len(ranges)):
        ax.axvline(x=float(i) / len(ranges), color="white", linewidth=5)
        ax.text(
            float(i) / len(ranges),
            0,
            "//",
            fontsize=7,
            color=MUTED_INK,
            horizontalalignment="center",
            verticalalignment="center",
        )

    return ax_i


# }}}

# Linked Reads methods
# {{{ def get_linked_plan(ranges, pairs, splits, linked_reads, gem_name):
def get_linked_plan(ranges, pairs, splits, linked_reads, gem_name):
    insert_sizes = []

    gem_poss = [[] for i in range(len(ranges))]

    linked_pair_steps = []
    # collect all the pairs in a gem
    for name in linked_reads[gem_name][0]:
        if name in pairs and len(pairs[name]) == 2:
            pair = pairs[name]
            plan = get_pair_plan(ranges, pair, linked_plan=True)
            if plan:
                insert_size, step = plan
                insert_sizes.append(insert_size)
                linked_pair_steps.append(step)

    # collect all the splits in a gem
    linked_split_steps = []
    for name in linked_reads[gem_name][1]:
        if name in splits:
            split = splits[name]
            plan = get_split_plan(ranges, split, linked_plan=True)
            if plan:
                insert_size, steps = plan
                insert_sizes.append(insert_size)
                linked_split_steps += steps

    if len(linked_split_steps) == 0 and len(linked_pair_steps) == 0:
        return None

    for step in linked_split_steps + linked_pair_steps:
        poss = [
            (step.start_pos.chrm, step.start_pos.start),
            (step.start_pos.chrm, step.start_pos.end),
            (step.end_pos.chrm, step.end_pos.start),
            (step.end_pos.chrm, step.end_pos.end),
        ]
        for pos in poss:
            hit = get_range_hit(ranges, pos[0], pos[1])
            if hit > -1:
                gem_poss[hit].append(pos[1])

    max_event_size = max(insert_sizes)

    gem_steps = []

    for i in range(len(ranges)):
        if len(gem_poss[i]) == 0:
            continue
        start = genome_interval(ranges[i].chrm, min(gem_poss[i]), min(gem_poss[i]))
        end = genome_interval(ranges[i].chrm, max(gem_poss[i]), max(gem_poss[i]))
        gem_steps.append(plan_step(start, end, "LINKED"))

    # if the gem extends beyond the range, then push the end pos to the
    # end/begining of the range
    if len(gem_steps) > 1:
        gem_steps[0].end_pos.start = ranges[0].end
        gem_steps[0].end_pos.end = ranges[0].end

        gem_steps[1].start_pos.start = ranges[1].start
        gem_steps[1].start_pos.end = ranges[1].start

    info = {
        "INSERTSIZE": max_event_size,
        "PAIR_STEPS": linked_pair_steps,
        "SPLIT_STEPS": linked_split_steps,
    }

    gem_steps[0].info = info

    return max(insert_sizes), gem_steps


# }}}

# {{{ def plot_linked_reads(pairs,
def plot_linked_reads(
    pairs,
    splits,
    linked_reads,
    ax,
    ranges,
    curr_min_insert_size,
    curr_max_insert_size,
    marker_size,
    jitter_bounds,
):
    """Plots all LinkedReads for the region
    """
    batch = ReadBatch()
    for linked_read in linked_reads:
        plan = get_linked_plan(ranges, pairs, splits, linked_reads, linked_read)

        if not plan:
            continue

        insert_size, steps = plan

        insert_size = jitter(insert_size, bounds=jitter_bounds)

        if not curr_min_insert_size or curr_min_insert_size > insert_size:
            curr_min_insert_size = insert_size
        if not curr_max_insert_size or curr_max_insert_size < insert_size:
            curr_max_insert_size = insert_size

        for step in steps:
            p = [
                map_genome_point_to_range_points(
                    ranges, step.start_pos.chrm, step.start_pos.start
                ),
                map_genome_point_to_range_points(
                    ranges, step.end_pos.chrm, step.end_pos.end
                ),
            ]
            # ignore points outside window
            if not points_in_window(p):
                continue

            READ_TYPES_USED["Linked read"] = True

            batch.add(
                p, [insert_size, insert_size], COLORS["Linked read"], lw=0.25, alpha=0.75
            )

        for pair_step in steps[0].info["PAIR_STEPS"]:
            pair_step.info["INSERTSIZE"] = insert_size
            plot_pair_plan(ranges, pair_step, ax, marker_size, jitter_bounds, batch)

        for split_step in steps[0].info["SPLIT_STEPS"]:
            split_step.info["INSERTSIZE"] = insert_size
            plot_split_plan(ranges, split_step, ax, marker_size, jitter_bounds, batch)

    batch.draw(ax, marker_size)
    return [curr_min_insert_size, curr_max_insert_size]


# }}}

# {{{def plot_long_reads(long_reads,
def plot_long_reads(long_reads, ax, ranges, curr_min_insert_size, curr_max_insert_size, jitter_bounds):
    """Plots all LongReads for the region
    """

    Path = mpath.Path

    colors = COLORS
    batch = ReadBatch()

    for read_name in long_reads:
        long_read_plan = get_long_read_plan(read_name, long_reads, ranges)

        if long_read_plan is None:
            continue
        max_gap = long_read_plan[0]
        steps = long_read_plan[1]
        for step in steps:

            p = [
                map_genome_point_to_range_points(
                    ranges, step.start_pos.chrm, step.start_pos.start
                ),
                map_genome_point_to_range_points(
                    ranges, step.end_pos.chrm, step.end_pos.end
                ),
            ]

            # some points are far outside of the printable area, so we
            # ignore them
            if not points_in_window(p):
                continue

            READ_TYPES_USED["Aligned long read"] = True

            event_type = step.info["TYPE"]
            READ_TYPES_USED[event_type] = True

            if event_type == "Align":
                batch.add(p, [max_gap, max_gap], colors[event_type], lw=1)

                curr_max_insert_size = max(curr_max_insert_size, max_gap)
            else:
                x1 = p[0]
                x2 = p[1]
                # get offset to bend the line up
                max_gap_offset = max(jitter(max_gap * 1.1, bounds=jitter_bounds), max_gap)
                pp = mpatches.PathPatch(
                    Path(
                        [
                            (x1, max_gap),
                            (x1, max_gap_offset),
                            (x2, max_gap_offset),
                            (x2, max_gap),
                        ],
                        [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4],
                    ),
                    fc="none",
                    color=colors[event_type],
                    alpha=0.25,
                    lw=1,
                    ls=":",
                )
                ax.add_patch(pp)

                # add some room for the bend line
                curr_max_insert_size = max(curr_max_insert_size, max_gap_offset)

    batch.draw(ax, 0)
    return [curr_min_insert_size, curr_max_insert_size]


# }}}

##Setup
# {{{def pair(arg):
def pair(arg):
    """Defines behavior for ArgParse pairs 

    Pairs must be comma-separated list of two items
    """
    try:
        parsed_arg = [int(x) for x in arg.split(",")]
        if len(parsed_arg) == 2:
            return parsed_arg
        else:
            logger.error("Invalid number of pair values")
            sys.exit(1)
    except Exception as e:
        logger.error("Invalid pair values")
        print(e, file=sys.stderr)
        sys.exit(1)


# }}}

# {{{def print_arguments(options):
def print_arguments(options):
    """Prints out the arguments to samplot as a json object

    Used as metadata for PlotCritic
    """
    if options.print_args or options.json_only:
        import json

        args_filename = os.path.splitext(options.output_file)[0] + ".json"
        args_info = {
            "titles": options.titles if options.titles else "None",
            "reference": options.reference if options.reference else "None",
            "bams": options.bams,
            "output_file": options.output_file,
            "start": options.start,
            "end": options.end,
            "chrom": options.chrom,
            "window": options.window,
            "max_depth": options.max_depth if options.max_depth else "None",
            "sv_type": options.sv_type,
            "transcript_file": options.transcript_file
            if options.transcript_file
            else "None",
        }
        with open(args_filename, "w") as outfile:
            json.dump(args_info, outfile)


# }}}


# {{{def setup_arguments():
def add_plot(parent_parser):
    """Defines the allowed arguments for plot function
    """
    parser = parent_parser.add_parser(
        "plot",
        help="Plot an image of a genome region from "
        + "CRAM/SAM alignments, "
        + "optimized for structural variant call review",
    )

    parser.add_argument(
        "-n",
        "--titles",
        help="Space-delimited list of plot titles. "
        + "Use quote marks to include spaces "
        + '(i.e. "plot 1" "plot 2")',
        type=str,
        nargs="+",
        required=False,
    )

    parser.add_argument(
        "-r",
        "--reference",
        help="Reference file for CRAM, required if " + "CRAM files used",
        type=str,
        required=False,
    )

    parser.add_argument(
        "-z",
        "--z",
        type=int,
        help="Number of stdevs from the mean (default 4)",
        required=False,
    )

    def bam_file(bam):
        if not os.path.isfile(bam):
            parser.error("alignment file {} does not exist or is not a valid file".format(bam))
        options = ["sam", "bam", "cram"]
        idx_options = ["sai", "bai", "crai", "csi"]
        fields = os.path.splitext(bam)
        ext = fields[1][1:].lower()
        if ext not in options:
            parser.error("alignment file {} is not in SAM/BAM/CRAM format".format(bam))
        idx_type = idx_options[options.index(ext)]
        #try the type-specific index name
        picard_bam = os.path.splitext(bam)[0]
        if (not os.path.isfile(bam + "." + idx_type) and 
                not os.path.isfile(picard_bam + "." + idx_type)):
            idx_type = idx_options[3]
            #try the csi index name
            if not os.path.isfile(bam + "." + idx_type):
                parser.error("alignment file {} has no index".format(bam))
        return bam


    parser.add_argument(
        "-b",
        "--bams",
        type=bam_file,
        nargs="+",
        help="Space-delimited list of BAM/CRAM file names",
        required=True,
    )

    parser.add_argument(
        "-o", 
        "--output_file", 
        type=str, 
        help="Output file name/type. "
        +"Defaults to {type}_{chrom}_{start}_{end}.png",
        required=False,
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory name. Defaults to working dir. "
        +"Ignored if --output_file is set",
        required=False,
    )

    parser.add_argument(
        "-s",
        "--start",
        type=int,
        help="Start position of region/variant (add multiple for translocation/BND events)",
        action="append",
        required=True,
    )

    parser.add_argument(
        "-e",
        "--end",
        type=int,
        help="End position of region/variant (add multiple for translocation/BND events)",
        action="append",
        required=True,
    )

    parser.add_argument(
        "-c",
        "--chrom", type=str,
        help="Chromosome (add multiple for translocation/BND events)",
        action="append",
        required=True
    )

    parser.add_argument(
        "-w",
        "--window",
        type=int,
        help="Window size (count of bases to include " + "in view), default(0.5 * len)",
        required=False,
    )

    parser.add_argument(
        "-d",
        "--max_depth",
        type=int,
        help="Max number of normal pairs to plot",
        required=False,
    )

    parser.add_argument(
        "-t",
        "--sv_type",
        type=str,
        help="SV type. If omitted, plot is created " + "without variant bar",
        required=False,
    )
    
    def gff_file(transcript_file):
        if not os.path.isfile(transcript_file):
            parser.error("transcript file {} does not exist or is not a valid file".format(transcript_file))
        options = ["gff", "gff3"]
        fields = os.path.splitext(transcript_file)
        ext = fields[1][1:]
        if ext == "gz":
            ext = os.path.splitext(fields[0])[1][1:]
        ext = ext.lower()
        if ext not in options:
            parser.error("transcript file {} is not in GFF3 format".format(transcript_file))

        idx_file = transcript_file + ".tbi"
        if not os.path.isfile(idx_file):
            idx_file = transcript_file + ".csi"
            if not os.path.isfile(idx_file):
                parser.error("transcript file {} is missing .tbi/.csi index file".format(transcript_file))
        return transcript_file

    parser.add_argument(
        "-T", "--transcript_file",
        help="GFF3 of transcripts",
        required=False,
        type=gff_file,
    )

    parser.add_argument(
        "--transcript_filename",
        help="Name for transcript track",
        required=False,
        type=str,
    )
    
    parser.add_argument(
        "--max_coverage_points",
        help="number of points to plot in coverage axis (the region is averaged into this many bins)",
        required=False,
        type=int,
    )

    def bed_file(annotation_file):
        if not os.path.isfile(annotation_file):
            parser.error("annotation file {} does not exist or is not a valid file".format(annotation_file))
        fields = os.path.splitext(annotation_file)
        ext = fields[1][1:]
        if ext == "gz":
            ext = os.path.splitext(fields[0])[1][1:]
        ext = ext.lower()
        if ext != "bed":
            parser.error("annotation file {} is not in BED format".format(annotation_file))

        idx_file = annotation_file + ".tbi"
        if not os.path.isfile(idx_file):
            idx_file = annotation_file + ".csi"
            if not os.path.isfile(idx_file):
                parser.error("annotation file {} is missing .tbi index file".format(annotation_file))
        return annotation_file

    parser.add_argument(
        "-A",
        "--annotation_files",
        type=bed_file,
        nargs="+",
        help="Space-delimited list of bed.gz tabixed "
        + "files of annotations (such as repeats, "
        + "mappability, etc.)",
        required=False,
    )
    
    parser.add_argument(
        "--annotation_filenames",
        type=str,
        nargs="+",
        help="Space-delimited list of names for the tracks in --annotation_files",
        required=False,
    )

    parser.add_argument(
        "--coverage_tracktype",
        type=str,
        help="type of track to use for low MAPQ " + "coverage plot.",
        choices=["stack", "superimpose", "none"],
        required=False,
    )

    parser.add_argument(
        "-a",
        "--print_args",
        action="store_true",
        help="Print commandline arguments to a json file, useful with PlotCritic",
        required=False,
    )

    parser.add_argument(
        "-H", "--plot_height", type=int, help="Plot height", required=False
    )

    parser.add_argument(
        "-W", "--plot_width", type=int, help="Plot width", required=False
    )

    parser.add_argument(
        "-q",
        "--include_mqual",
        type=int,
        help="Min mapping quality of reads to be included in plot (default 1)",
        required=False,
    )

    parser.add_argument(
        "--separate_mqual",
        type=int,
        help="coverage from reads with MAPQ <= separate_mqual "
        + "plotted in lighter grey. To disable, "
        + "pass in negative value",
        required=False,
    )

    parser.add_argument(
        "-j",
        "--json_only",
        action="store_true",
        help="Create only the json file, not the " + "image plot",
        required=False,
    )

    parser.add_argument(
        "--start_ci",
        help="confidence intervals of SV first "
        + "breakpoint (distance from the "
        + "breakpoint). Must be a "
        + "comma-separated pair of ints (i.e. 20,40)",
        type=pair,
        required=False,
    )

    parser.add_argument(
        "--end_ci",
        help="confidence intervals of SV end "
        + "breakpoint (distance from the "
        + "breakpoint). Must be a "
        + "comma-separated pair of ints (i.e. 20,40)",
        type=pair,
        required=False,
    )

    parser.add_argument(
        "--long_read",
        type=int,
        help="Min length of a read to be treated as a " + "long-read (default 1000)",
        required=False,
    )

    parser.add_argument(
        "--ignore_hp",
        action="store_true",
        help="Choose to ignore HP tag in alignment files",
        required=False,
    )
    parser.add_argument(
        "--min_event_size",
        type=int,
        help="Min size of an event in long-read " + "CIGAR to include (default 20)",
        required=False,
    )

    parser.add_argument(
        "--xaxis_label_fontsize",
        type=int,
        help="Font size for X-axis labels (default 6)",
        required=False,
    )

    parser.add_argument(
        "--yaxis_label_fontsize",
        type=int,
        help="Font size for Y-axis labels (default 6)",
        required=False,
    )

    parser.add_argument(
        "--legend_fontsize",
        type=int,
        help="Font size for legend labels (default 6)",
        required=False,
    )

    parser.add_argument(
        "--annotation_fontsize",
        type=int,
        help="Font size for annotation labels (default 6)",
        required=False,
    )

    parser.add_argument(
        "--hide_annotation_labels",
        action="store_true",
        help="Hide the label (fourth column text) "
        + "from annotation files, useful for regions "
        + "with many annotations",
        required=False,
    )

    parser.add_argument(
        "--coverage_only",
        action="store_true",
        help="Hide all reads and show only coverage",
        required=False,
    )

    parser.add_argument(
        "--max_coverage",
        type=int,
        help="apply a maximum coverage cutoff. Unlimited by default",
    )

    parser.add_argument(
        "--same_yaxis_scales",
        action="store_true",
        help="Set the scales of the Y axes to the " + "max of all",
        required=False,
    )

    parser.add_argument(
        "--marker_size",
        type=int,
        help="Size of marks on pairs and splits (default 3)",
        required=False,
    )
    parser.add_argument(
        "--jitter",
        type=float,
        nargs="?",
        const=0.08,
        help="Add uniform random noise to insert sizes. This can be helpful "
             "to resolve overlapping entries. Either a custom value (<1.0) is "
             "supplied or %(const)s will be used."
    )
    parser.add_argument(
        "--dpi",
        type=int,
        help="Dots per inches (pixel count)",
        required=False,
    )
    parser.add_argument(
        "--annotation_scalar",
        type=float,
        help="scaling factor for the optional annotation/trascript tracks",
        required=False,
    )
    parser.add_argument(
        "--zoom",
        type=int,
        help="Only show +- zoom amount around breakpoints, "
            +"much faster for large regions. "
            +"Ignored if region smaller than --zoom (default 500000)",
        required=False,
    )

    parser.add_argument(
        "--debug",
        type=str,
        help="Print debug statements",
        required=False
    )
    parser.add_argument(
        "--genes",
        type=str,
        help="Draw a gene track. Give a genome build with a bundled MANE "
        + "Select track (GRCh38) or the path of a tabix indexed file in the "
        + "same format (see scripts/make_gene_track.py). Exons, coding "
        + "regions and exon numbers are drawn on small plots and one bar "
        + "per gene on large ones",
        required=False,
    )
    parser.add_argument(
        "--gene_detail_max_size",
        type=int,
        help="The gene track shows exons when the plotted region is at most "
        + "this many bases, and whole genes above it",
        required=False,
    )
    parser.add_argument(
        "--discordant_only_min_size",
        type=int,
        help="When the plotted region is at least this many bases, properly "
        + "paired reads are only counted towards coverage: just discordant "
        + "pairs and split reads are kept and drawn, which is much faster "
        + "and less cluttered for large variants. 0 to always keep "
        + "normal pairs",
        required=False,
    )
    parser.add_argument(
        "--coverage_trend_min_size",
        type=int,
        help="Draw a smoothed (rolling median) coverage trend line, which makes "
        + "copy number breakpoints easier to see, when the plotted region is at "
        + "least this many bases. 0 to never draw it",
        required=False,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        help=SUPPRESS,
    )
    # every default lives on PlotOptions
    defaults = PlotOptions()
    parser.set_defaults(
        func=plot,
        **{
            action.dest: getattr(defaults, action.dest)
            for action in parser._actions
            if hasattr(defaults, action.dest)
        }
    )
    return parser


# }}}

def open_alignment_file(path, reference=None):
    """Opens a BAM or CRAM, raising SamplotError if it can't be read"""
    try:
        if not reference:
            return pysam.AlignmentFile(path, "rb")
        return pysam.AlignmentFile(path, "rc", reference_filename=reference)
    except Exception as err:
        raise SamplotError(
            "Could not open alignment file {}: {}. Please make sure that it is "
            "sorted and indexed before trying again".format(path, err)
        ) from err


# {{{def estimate_fragment_len(bam)
def estimate_fragment_len(bam, reference):
    bam_file = open_alignment_file(bam, reference)

    frag_lens = []

    for i, read in enumerate(bam_file):
        if i >= 10000:
            break
        frag_lens.append(abs(read.tlen))
    if len(frag_lens) >= 5000:
        return np.median(frag_lens)
    else:
        logger.warning(
            "Insufficient reads for fragment length estimate.\nContinuing with unmodified window size"
        )
        return 0


# {{{def set_plot_dimensions(sv,
def set_plot_dimensions(sv, options):
    """Chooses appropriate dimensions for the plot

    Includes the number of samples, whether a variant type is included, and
    any annotations in height. Includes the start, end, and window argument
    in width If height and width are chosen by user, these are used instead

    Return plot height, width, and window as integers
    """

    plot_height = 5
    plot_width = 8
    if options.plot_height:
        plot_height = options.plot_height
    else:
        num_subplots = len(options.bams)
        if options.annotation_files:
            num_subplots += 0.3 * len(options.annotation_files)
        if options.transcript_file:
            num_subplots += 0.3
        plot_height = 2 + num_subplots

    if options.plot_width:
        plot_width = options.plot_width

    window = 0
    ranges = []
    if options.window:
        window = options.window

    """ 
    Several things determine the window size. 
    1) SV is not given, window = 0
    1) SV is given
        1) it is directly set
        2) it is not directly set
           2.1) single interval SV
           2.2) options.zoom set
           2.3) 2-interval SV
    """
    # if an SV type is given, then expand the window around its bounds
    if options.sv_type:
        # if the sv has one interval then set the window proportional
        # to sv size and set one range
        if len(sv) == 1:
            if options.window:
                window = options.window
            else:
                window = int((sv[0].end - sv[0].start) / 2)
                frag_len = estimate_fragment_len(options.bams[0], options.reference)

                if (0 < frag_len) and (window < 1.5 * frag_len):
                    old_window = window
                    window = int(1.5 * frag_len)
                    logger.warning(
                        "Window size is under 1.5x the estimated fragment length "
                        + "and will be resized to {}. Rerun with -w {} to override".format(
                            window, old_window
                        )
                    )

            ranges = [
                genome_interval(
                    sv[0].chrm, max(0, sv[0].start - window), sv[0].end + window
                )
            ]

            # if region is larger than options.zoom, set window to options.zoom and set two ranges
            if window >= options.zoom:
                window = options.zoom
                ranges = [
                    genome_interval(
                        sv[0].chrm,
                        max(0, sv[0].start - window),
                        sv[0].start + window,
                    ),
                    genome_interval(
                        sv[0].chrm, max(0, sv[0].end - window), sv[0].end + window
                    ),
                ]
        elif len(sv) == 2:
            if options.window:
                window = options.window
            elif options.zoom:
                window = options.zoom
            else:
                window = 1000

            ranges = [
                genome_interval(
                    sv[0].chrm, max(0, sv[0].start - window), sv[0].start + window
                ),
                genome_interval(
                    sv[1].chrm, max(0, sv[1].end - window), sv[1].end + window
                ),
            ]
        else:
            raise SamplotError("{} genome splits are not supported".format(str(len(sv))))
    else:
        ranges = [genome_interval(sv[0].chrm, sv[0].start, sv[0].end)]

    return plot_height, plot_width, window, ranges


# }}}

# {{{def get_read_data(ranges,
def get_read_data(ranges, options):
    """Reads alignment files to extract reads for the region

    Uses these options: bams and reference (needed for CRAM); include_mqual
    (reads below it are ignored); separate_mqual; coverage_only (reads only
    count towards coverage); long_read (reads at least this long are treated
    as long reads); min_event_size; ignore_hp; max_depth and z (normal pairs
    are downsampled to max_depth, pairs with an insert size over z standard
    deviations from the mean are discordant and always kept);
    same_yaxis_scales; and discordant_only_min_size (when the ranges cover at
    least this many bases, properly paired reads only count towards coverage
    and just discordant pairs and split reads are kept).

    Returns the read data and the coverage to scale every sample to (0
    unless same_yaxis_scales).
    """

    plotted_bases = sum(r.end - r.start for r in ranges)
    discordant_only = bool(
        options.discordant_only_min_size
        and plotted_bases >= options.discordant_only_min_size
    )

    all_pairs = []
    all_splits = []
    all_coverages = []
    all_long_reads = []
    all_linked_reads = []

    max_coverage = 0

    for bam_file_name in options.bams:
        bam_file = open_alignment_file(bam_file_name, options.reference)

        pairs = {}
        splits = {}
        long_reads = {}
        # coverage[haplotype][chrom] is a list of RangeCoverage, one per range
        coverage = {}
        linked_reads = {}

        for r in ranges:
            # Define range boundries
            range_start = max(0, r.start - 1000)
            range_end = r.end + 1000

            try:
                bam_iter = bam_file.fetch(r.chrm, range_start, range_end)
            except ValueError:
                chrm = r.chrm
                if chrm[:3] == "chr":
                    chrm = chrm[3:]
                else:
                    chrm = "chr" + chrm
                bam_iter = bam_file.fetch(chrm, range_start, range_end)

            chrm = strip_chr(r.chrm)

            # Per-base coverage over the range for each haplotype, allocated the
            # first time a haplotype is seen. If using separate_mqual the first
            # column will hold the coverage for high quality reads and the
            # second column low quality reads. Otherwise all coverage will be
            # in the second column.
            range_len = range_end - range_start
            range_hp_coverage = {}
            include_mqual = options.include_mqual
            separate_mqual = options.separate_mqual
            coverage_only = options.coverage_only
            long_read_length = options.long_read
            ignore_hp = options.ignore_hp

            for read in bam_iter:
                if (
                    read.is_qcfail
                    or read.is_unmapped
                    or read.is_duplicate
                    or int(read.mapping_quality) < include_mqual
                ):
                    continue

                if not coverage_only:
                    if read.query_length >= long_read_length:
                        add_long_reads(
                            bam_file, read, long_reads, options.min_event_size, ignore_hp
                        )
                    elif not (
                        discordant_only
                        and read.is_proper_pair
                        and not read.has_tag("SA")
                        and not read.has_tag("MI")
                    ):
                        add_pair_end(bam_file, read, pairs, linked_reads, ignore_hp)
                        add_split(read, splits, bam_file, linked_reads, ignore_hp)

                # Add read coverage to the specified haplotype and column
                hp = 0 if ignore_hp or not read.has_tag("HP") else read.get_tag("HP")
                column = 0 if separate_mqual and (read.mapping_quality > separate_mqual) else 1
                if hp not in range_hp_coverage:
                    range_hp_coverage[hp] = CoverageCounter(range_start, range_len)
                range_hp_coverage[hp].add(read, column)

            # Keep the coverage of each haplotype that has any
            for hp, counter in range_hp_coverage.items():
                range_coverage = counter.to_array()
                if not range_coverage.any():
                    continue
                if hp not in coverage:
                    coverage[hp] = {}
                coverage[hp].setdefault(chrm, []).append(
                    RangeCoverage(range_start, range_coverage)
                )

        if (
            len(pairs) == 0
            and len(splits) == 0
            and len(long_reads) == 0
            and len(linked_reads) == 0
        ):
            if not coverage_only:
                logger.warning(
                    "No data returned from fetch in regions {} from {}".format(
                        " ".join([str(r) for r in ranges]),
                        bam_file
                    )
                )

        # Update max_coverage
        for hp in coverage:
            for chrm in coverage[hp]:
                for range_coverage in coverage[hp][chrm]:
                    curr_max = np.percentile(range_coverage.values.sum(axis=1), 99.5)
                    if curr_max > max_coverage:
                        max_coverage = curr_max

        all_coverages.append(coverage)
        all_pairs.append(pairs)
        all_splits.append(splits)
        all_long_reads.append(long_reads)
        all_linked_reads.append(linked_reads)

    read_data = {
        "all_pairs": all_pairs,
        "all_splits": all_splits,
        "all_coverages": all_coverages,
        "all_long_reads": all_long_reads,
        "all_linked_reads": all_linked_reads,
    }

    # Sample +/- pairs in the normal insert size range. Not needed when only
    # discordant pairs were kept (and the insert size statistics that
    # downsampling uses to spot discordant pairs would be wrong)
    if options.max_depth and not discordant_only:
        read_data["all_pairs"] = downsample_pairs(
            options.max_depth, options.z, read_data["all_pairs"]
        )
    if not options.same_yaxis_scales:
        max_coverage = 0
    return read_data, max_coverage


# }}}

# {{{def downsample_pairs(max_depth, z_score, all_pairs):
def downsample_pairs(max_depth, z_score, all_pairs):
    """Downsamples to keep only max_depth normal pairs from all PairedEnd
    reads 
    """
    for bam_i in range(len(all_pairs)):
        for hp_i in all_pairs[bam_i]:
            all_pairs[bam_i][hp_i] = sample_normal(
                max_depth, all_pairs[bam_i][hp_i], z_score
            )
    return all_pairs


# }}}

# {{{def set_haplotypes(curr_coverage):
def set_haplotypes(curr_coverage):
    """Creates a list to manage counting haplotypes for subplots
    """
    hps = sorted(curr_coverage.keys(), reverse=True)
    # if there are multiple haplotypes, must have 0,1,2
    if len(hps) > 1 or (len(hps) == 1 and hps[0] != 0):
        if 0 not in hps:
            hps.append(0)
        if 1 not in hps:
            hps.append(1)
        if 2 not in hps:
            hps.append(2)
    elif 0 not in hps:
        hps.append(0)
    hps.sort(reverse=True)
    return hps


# }}}

def get_xaxis_ticks(ranges, ticks_per_plot=8):
    """Picks round genomic positions for the x axis ticks

    Returns the tick positions in plot coordinates (0-1) and their labels.
    """
    if len(ranges) > 2:
        raise SamplotError("Ranges greater than 2 are not supported")

    ticks = []
    labels = []
    per_range = max(2, int(ticks_per_plot / len(ranges)))
    locator = ticker.MaxNLocator(nbins=per_range, steps=[1, 2, 2.5, 5, 10], integer=True)
    for r in ranges:
        for value in locator.tick_values(r.start, r.end):
            if value < r.start or value > r.end:
                continue
            ticks.append(map_genome_point_to_range_points(ranges, r.chrm, value))
            labels.append("{:,}".format(int(value)))
    return ticks, labels


def format_insert_size(value, pos=None):
    """Tick label for the insert size axis"""
    if value >= 1000000:
        return "{:.3g}M".format(value / 1000000.0)
    if value >= 10000:
        return "{:.3g}k".format(value / 1000.0)
    return "{:.0f}".format(value)


def sv_spans_ranges(sv, ranges):
    """True if the plotted ranges lie entirely inside the variant (neither
    breakpoint is in view)"""
    if strip_chr(sv[0].chrm) != strip_chr(sv[-1].chrm):
        return False
    return all(
        strip_chr(r.chrm) == strip_chr(sv[0].chrm)
        and r.start >= sv[0].start
        and r.end <= sv[-1].end
        for r in ranges
    )


# {{{def plot_samples(ranges,
def plot_samples(ranges, read_data, grid, ax_i, number_of_axes, options, max_coverage, sv=None):

    """Plots all samples
    """
    max_insert_size = 0
    fig = grid.figure
    x_ticks, x_tick_labels = get_xaxis_ticks(ranges)
    breakpoints = get_breakpoint_points(sv, ranges) if sv and options.sv_type else []

    # If jitter > 0.08 is use we need to shift the ylim a bit to not hide any entires.
    ylim_margin = max(1.02 + options.jitter, 1.10)
    for i in range(len(options.bams)):
        hps = set_haplotypes(read_data["all_coverages"][i])
        inner_axs = gridspec.GridSpecFromSubplotSpec(
            len(hps), 1, subplot_spec=grid[ax_i], wspace=0.0, hspace=0.5
        )
        axs = {}
        for j in range(len(hps)):
            axs[j] = fig.add_subplot(inner_axs[hps[j]])

        curr_min_insert_size = None
        curr_max_insert_size = 0

        cover_axs = {}
        for hp in hps:
            curr_ax = axs[hp]

            curr_splits = []
            if hp in read_data["all_splits"][i]:
                curr_splits = read_data["all_splits"][i][hp]

            curr_linked_reads = []
            if hp in read_data["all_linked_reads"][i]:
                curr_linked_reads = read_data["all_linked_reads"][i][hp]

            curr_long_reads = []
            if hp in read_data["all_long_reads"][i]:
                curr_long_reads = read_data["all_long_reads"][i][hp]

            curr_pairs = []
            if hp in read_data["all_pairs"][i]:
                curr_pairs = read_data["all_pairs"][i][hp]

            curr_coverage = {}
            if hp in read_data["all_coverages"][i]:
                curr_coverage = read_data["all_coverages"][i][hp]

            cover_ax = plot_coverage(
                curr_coverage, curr_ax, ranges, len(hps), max_coverage, options, sv=sv
            )

            if len(curr_linked_reads) > 0:
                curr_min_insert_size, curr_max_insert_size = plot_linked_reads(
                    curr_pairs,
                    curr_splits,
                    curr_linked_reads,
                    curr_ax,
                    ranges,
                    curr_min_insert_size,
                    curr_max_insert_size,
                    options.marker_size,
                    options.jitter
                )
            elif len(curr_long_reads) > 0:
                curr_min_insert_size, curr_max_insert_size = plot_long_reads(
                    curr_long_reads,
                    curr_ax,
                    ranges,
                    curr_min_insert_size,
                    curr_max_insert_size,
                    options.jitter
                )
            else:
                curr_min_insert_size, curr_max_insert_size = plot_pairs(
                    curr_pairs,
                    curr_ax,
                    ranges,
                    curr_min_insert_size,
                    curr_max_insert_size,
                    options.marker_size,
                    options.jitter
                )

                curr_min_insert_size, curr_max_insert_size = plot_splits(
                    curr_splits,
                    curr_ax,
                    ranges,
                    curr_min_insert_size,
                    curr_max_insert_size,
                    options.marker_size,
                    options.jitter,
                )

            cover_axs[hp] = cover_ax
            if curr_max_insert_size and (curr_max_insert_size > max_insert_size):
                max_insert_size = curr_max_insert_size

        # {{{ set axis parameters
        # set the axis title to be either one passed in or filename
        curr_ax = axs[hps[0]]
        if options.titles and len(options.titles) == len(options.bams):
            curr_ax.set_title(options.titles[i], fontsize=8, loc="left", color=INK)
        else:
            curr_ax.set_title(os.path.basename(options.bams[i]), fontsize=8, loc="left", color=INK)

        if len(axs) > 1:
            for j in axs:
                curr_ax = axs[j]
                fp = dict(size=7, color=MUTED_INK)
                text = "HP: "
                if j == 0:
                    text += "Undef"
                else:
                    text += str(j)
                at = AnchoredText(
                    text, loc=2, prop=fp, borderpad=0, pad=0, frameon=False
                )
                curr_ax.add_artist(at)

        for j in hps:
            curr_ax = axs[j]
            curr_ax.set_xlim([0, 1])
            # round the top of the axis up so the tick labels are round numbers
            y_intervals = 5 if len(hps) == 1 else 2
            if options.same_yaxis_scales:
                y_top = max(1, max_insert_size * ylim_margin)
            else:
                y_top = max(1, curr_max_insert_size * ylim_margin)
            curr_ax.set_ylim([0, nice_axis_top(y_top, y_intervals)])
            for spine in curr_ax.spines.values():
                spine.set_visible(False)
            curr_ax.tick_params(axis="y", labelsize=options.yaxis_label_fontsize)
            # if there's one hp, 6 ticks fit. Otherwise, do 3
            tick_count = 6 if len(hps) == 1 else 3
            curr_ax.yaxis.set_major_locator(ticker.LinearLocator(tick_count))
            curr_ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_insert_size))

            curr_ax.tick_params(axis="both", length=0)
            curr_ax.set_xticks(x_ticks)
            curr_ax.set_xticklabels([])
            if options.coverage_only:
                curr_ax.yaxis.set_visible(False)

            # mark the breakpoints down every panel, and lightly shade the
            # variant between them
            in_view = [b for b in breakpoints if b is not None]
            for b in in_view:
                curr_ax.axvline(
                    b, color=sv_color(options.sv_type), linestyle="--", linewidth=0.8,
                    alpha=0.8, zorder=1,
                )
            if breakpoints:
                span = [
                    0.0 if breakpoints[0] is None else breakpoints[0],
                    1.0 if breakpoints[1] is None else breakpoints[1],
                ]
                if span[1] > span[0] and (in_view or sv_spans_ranges(sv, ranges)):
                    curr_ax.axvspan(
                        span[0], span[1], color=sv_color(options.sv_type), alpha=0.07,
                        linewidth=0, zorder=0,
                    )

        last_sample_num = number_of_axes - 1
        if options.annotation_files:
            last_sample_num -= len(options.annotation_files)
        if options.transcript_file:
            last_sample_num -= 1
        if options.genes:
            last_sample_num -= 1

        if ax_i == last_sample_num:
            curr_ax = axs[hps[-1]]
            curr_ax.set_xticklabels(x_tick_labels, fontsize=options.xaxis_label_fontsize)
            chrms = list(dict.fromkeys(x.chrm for x in ranges))
            curr_ax.set_xlabel(
                "Chromosomal position on " + "/".join(chrms), fontsize=8
            )

        curr_ax = axs[hps[int(len(hps) / 2)]]
        curr_ax.set_ylabel("Insert size", fontsize=8)
        cover_ax = cover_axs[hps[int(len(hps) / 2)]]
        cover_ax.set_ylabel("Coverage", fontsize=8, color=GREY)
        # }}}

        ax_i += 1
    return ax_i


# }}}

# {{{def plot_legend(fig, legend_fontsize):
def plot_legend(fig, legend_fontsize, marker_size):
    """Plots the figure legend
    """
    marker_colors = []
    marker_labels = []
    read_colors = {
        read_type: COLORS[read_type]
        for read_type in (
            "Deletion/Normal",
            "Duplication",
            "Inversion",
            "Aligned long read",
            "Linked read",
        )
    }

    for read_type in READ_TYPES_USED:
        if read_type in read_colors:
            color = read_colors[read_type]
            flag = READ_TYPES_USED[read_type]
            if flag:
                marker_colors.append(color)
                marker_labels.append(read_type)
    legend_elements = []

    for color in marker_colors:
        legend_elements += [
            Line2D([0, 0], [0, 1], color=color, linestyle="-", lw=1)
        ]
    if READ_TYPES_USED["Split-read"]:
        marker_labels.append("Split read")
        legend_elements += [
            Line2D(
                [0, 0],
                [0, 1],
                markerfacecolor="None",
                markeredgecolor=GREY,
                color=GREY,
                marker="o",
                markersize=marker_size,
                linestyle=":",
                lw=1,
            )
        ]

    if READ_TYPES_USED["Paired-end read"]:
        marker_labels.append("Paired-end read")
        legend_elements += [
            Line2D(
                [0, 0],
                [0, 1],
                markerfacecolor="None",
                markeredgecolor=GREY,
                color=GREY,
                marker="s",
                markersize=marker_size,
                linestyle="-",
                lw=1,
            )
        ]

    if not legend_elements:
        return
    fig.legend(
        legend_elements,
        marker_labels,
        loc="upper right",
        fontsize=legend_fontsize,
        frameon=False,
        labelcolor=MUTED_INK,
        # a compact block that sits beside the title, clear of the panels
        ncol=3,
        columnspacing=1.2,
        handlelength=1.8,
        borderaxespad=0.3,
    )


# }}}

# {{{def create_gridspec(bams, transcript_file, annotation_files, sv_type ):
def create_gridspec(options, read_data, fig=None, gene_inches=0):
    """Helper function for creation of a correctly-sized GridSpec instance
    """
    # give one axis to display each sample
    num_ax = len(options.bams)

    # add another if we are displaying the SV
    if options.sv_type:
        num_ax += 1

    # add another if a annotation file is given
    if options.transcript_file:
        num_ax += 1

    if options.annotation_files:
        num_ax += len(options.annotation_files)

    # set the relative sizes for each
    ratios = []
    if options.sv_type:
        ratios = [1]

    for i in range(len(options.bams)):
        ratios.append(len(read_data["all_coverages"][i]) * 3)
        if len(read_data["all_coverages"]) > 0:
            ratios[-1] = 9

    if options.annotation_files:
        ratios += [options.annotation_scalar] * len(options.annotation_files)
    if options.transcript_file:
        ratios.append(options.annotation_scalar * 3)

    # the gene track goes under the samples and annotations (a sample is a
    # ratio of 9 and about an inch tall)
    if gene_inches:
        num_ax += 1
        gene_ratio = 9 * gene_inches
        if options.transcript_file:
            ratios.insert(len(ratios) - 1, gene_ratio)
        else:
            ratios.append(gene_ratio)

    if fig is None:
        fig = Figure()
    return gridspec.GridSpec(num_ax, 1, height_ratios=ratios, figure=fig), num_ax


# }}}

##Annotations/Transcript methods
# {{{def get_plot_annotation_plan(ranges, annotation_file):
def get_plot_annotation_plan(ranges, annotation_file):
    annotation_plan = []
    for r in ranges:
        itr = get_tabix_iter(r.chrm, r.start, r.end, annotation_file)
        if not (itr):
            continue
        for row in itr:
            A = row.rstrip().split()
            A[0] = strip_chr(A[0])
            chrm = A[0]
            start = int(A[1])
            end = int(A[2])

            interval = genome_interval(chrm, start, end)

            # check to see if any part of this alignment overlaps a plot
            # range
            in_range = False
            for r in ranges:
                if r.intersect(interval) == 0:
                    in_range = True
            if in_range:
                step = plan_step(
                    genome_interval(chrm, start, start),
                    genome_interval(chrm, end, end),
                    "ANNOTATION",
                )
                if len(A) > 3:
                    try:
                        v = float(A[3])
                        step.event = "FLOAT_ANNOTATION"
                        step.info = v
                    except ValueError:
                        step.event = "STRING_ANNOTATION"
                        step.info = A[3]

                annotation_plan.append(step)
    return annotation_plan


# }}}

# {{{def plot_annotations(annotation_files, chrom, start, end,
def plot_annotations(options, ranges, grid, ax_i):
    """Plots annotation information from region 
    """
    annotation_files = options.annotation_files
    annotation_filenames = options.annotation_filenames
    hide_annotation_labels = options.hide_annotation_labels
    annotation_fontsize = options.annotation_fontsize
    if not annotation_filenames:
        annotation_filenames = []
        for annotation_file in annotation_files:
            annotation_filenames.append(os.path.basename(annotation_file))

    for i,annotation_file in enumerate(annotation_files):
        annotation_plan = get_plot_annotation_plan(ranges, annotation_file)
        annotation_filename = annotation_filenames[i]

        if len(annotation_plan) == 0:
            continue
        ax = grid.figure.add_subplot(grid[ax_i])
        ax_i += 1

        for step in annotation_plan:
            p = [
                map_genome_point_to_range_points(
                    ranges, step.start_pos.chrm, step.start_pos.start
                ),
                map_genome_point_to_range_points(
                    ranges, step.end_pos.chrm, step.end_pos.end
                ),
            ]
            # if an annotation lies outside the window, its coordinate will be None, so we trim to the window
            if p[0] is None:
                p[0] = 0
            if p[1] is None:
                p[1] = 1

            if step.event == "ANNOTATION":
                ax.plot(p, [0, 0], "-", color=MUTED_INK, lw=5, solid_capstyle="butt")
            elif step.event == "FLOAT_ANNOTATION":
                ax.plot(p, [0, 0], "-", color=str(step.info), lw=15, solid_capstyle="butt")
            elif step.event == "STRING_ANNOTATION":
                ax.plot(p, [0, 0], "-", color=MUTED_INK, lw=15, solid_capstyle="butt")
                if step.info and not hide_annotation_labels:
                    ax.text(
                        p[0],
                        0.06,
                        step.info,
                        color=INK,
                        fontsize=annotation_fontsize,
                    )
            else:
                raise SamplotError("Unsupported annotation type: {}".format(step.event))

        # set axis parameters
        clean_axis(ax)
        ax.set_title(annotation_filename, fontsize=8, loc="left", color=MUTED_INK)


# }}}

# {{{def get_interval_range_plan_start_end(ranges, interval):
def get_interval_range_plan_start_end(ranges, interval):

    # transcript can span ranges
    start_range_hit_i = get_range_hit(ranges, interval.chrm, interval.start)
    end_range_hit_i = get_range_hit(ranges, interval.chrm, interval.end)

    if start_range_hit_i is None and end_range_hit_i is None:
        for i, range_item in enumerate(ranges):
            if (
                (strip_chr(range_item.chrm) == strip_chr(interval.chrm))
                and (interval.start <= range_item.start <= interval.end)
                and (interval.start <= range_item.end <= interval.end)
            ):
                start_range_hit_i = i
                end_range_hit_i = i

    start = None
    end = None
    # neither end is in range, add nothing
    if start_range_hit_i == None and end_range_hit_i == None:
        return None, None
    # start is in, end is not
    elif end_range_hit_i == None:
        start = genome_interval(
            interval.chrm,
            max(interval.start, ranges[start_range_hit_i].start),
            max(interval.start, ranges[start_range_hit_i].start),
        )
        end = genome_interval(
            interval.chrm, ranges[start_range_hit_i].end, ranges[start_range_hit_i].end
        )
    # end is in, start is not
    elif start_range_hit_i == None:
        start = genome_interval(
            interval.chrm, ranges[end_range_hit_i].start, ranges[end_range_hit_i].start
        )
        end = genome_interval(
            interval.chrm,
            min(interval.end, ranges[end_range_hit_i].end),
            min(interval.end, ranges[end_range_hit_i].end),
        )
    # in same range or in different ranges
    else:
        start = genome_interval(
            interval.chrm,
            max(interval.start, ranges[start_range_hit_i].start),
            max(interval.start, ranges[start_range_hit_i].start),
        )
        end = genome_interval(
            interval.chrm,
            min(interval.end, ranges[end_range_hit_i].end),
            min(interval.end, ranges[end_range_hit_i].end),
        )
    return start, end


# }}}

# {{{def get_transcript_plan(ranges, transcript_file):
def get_transcript_plan(ranges, transcript_file):
    genes = {}
    transcripts = {}
    cdss = {}

    for r in ranges:
        itr = get_tabix_iter(r.chrm, r.start, r.end, transcript_file)
        if not itr:
            continue
        for row in itr:
            gene_annotation = row.rstrip().split()

            if gene_annotation[2] == "gene":
                info = dict(
                    [list(val.split("=")) for val in gene_annotation[8].split(";")]
                )

                info["strand"] = gene_annotation[6] == "+"

                if "Name" not in info:
                    continue

                genes[info["Name"]] = [
                    genome_interval(
                        gene_annotation[0],
                        int(gene_annotation[3]),
                        int(gene_annotation[4]),
                    ),
                    info,
                ]
            elif gene_annotation[2] in ["transcript", "mRNA"]:
                info = dict(
                    [list(val.split("=")) for val in gene_annotation[8].split(";")]
                )
                info["strand"] = gene_annotation[6] == "+"

                if info["Parent"] not in transcripts:
                    transcripts[info["Parent"]] = {}
                transcripts[info["Parent"]][info["ID"]] = [
                    genome_interval(
                        gene_annotation[0],
                        int(gene_annotation[3]),
                        int(gene_annotation[4]),
                    ),
                    info,
                ]
            elif gene_annotation[2] == "CDS":
                info = dict(
                    [list(val.split("=")) for val in gene_annotation[8].split(";")]
                )
                info["strand"] = gene_annotation[6] == "+"

                if info["Parent"] not in cdss:
                    cdss[info["Parent"]] = {}

                if info["ID"] not in cdss[info["Parent"]]:
                    cdss[info["Parent"]][info["ID"]] = []

                cdss[info["Parent"]][info["ID"]].append(
                    genome_interval(
                        gene_annotation[0],
                        int(gene_annotation[3]),
                        int(gene_annotation[4]),
                    )
                )
    transcript_plan = []
    for gene in genes:
        gene_id = genes[gene][1]["ID"]
        if gene_id not in transcripts:
            continue
        for transcript in transcripts[gene_id]:
            interval, info = transcripts[gene_id][transcript]
            start, end = get_interval_range_plan_start_end(ranges, interval)

            if not start or not end:
                continue

            step = plan_step(start, end, "TRANSCRIPT")
            step.info = {"Name": None, "Strand": None, "Exons": None}
            step.info["Name"] = info["Name"]
            step.info["Strand"] = info["strand"]

            exons = []
            if transcript in cdss:
                for cds in cdss[transcript]:
                    for exon in cdss[transcript][cds]:
                        start, end = get_interval_range_plan_start_end(ranges, exon)
                        if start and end:
                            exons.append(plan_step(start, end, "EXON"))
            if len(exons) > 0:
                step.info["Exons"] = exons

            transcript_plan.append(step)
    return transcript_plan


# }}}

# {{{ def plot_transcript(transcript_file, chrom, start, end,
def plot_transcript(options, ranges, grid):
    """Plots a transcript file annotation
    """
    transcript_file = options.transcript_file
    transcript_filename = options.transcript_filename
    annotation_fontsize = options.annotation_fontsize
    if not transcript_filename:
        transcript_filename = os.path.basename(transcript_file)
    transcript_idx = 0
    transcript_idx_max = 0
    currect_transcript_end = 0
    ax = grid.figure.add_subplot(grid[-1])

    transcript_plan = get_transcript_plan(ranges, transcript_file)

    for step in transcript_plan:
        p = [
            map_genome_point_to_range_points(
                ranges, step.start_pos.chrm, step.start_pos.start
            ),
            map_genome_point_to_range_points(
                ranges, step.end_pos.chrm, step.end_pos.end
            ),
        ]
        # if an annotation lies outside the window, its coordinate will be None, so we trim to the window
        if p[0] is None:
            p[0] = 0
        if p[1] is None:
            p[1] = 0

        # Reset transcript index outside of current stack
        if p[0] > currect_transcript_end:
            transcript_idx = 0

        currect_transcript_end = max(p[1], currect_transcript_end)

        ax.plot(
            p, [transcript_idx, transcript_idx], "-", color=BLUE, lw=0.8,
            solid_capstyle="butt",
        )

        # Print arrows throughout gene to show direction.
        nr_arrows = 2 + int((p[1]-p[0])/0.02)
        arrow_locs = np.linspace(p[0], p[1], nr_arrows)
        arrowprops = dict(arrowstyle="->", color=BLUE, lw=0.5,
                          mutation_aspect=2, mutation_scale=3)

        if step.info["Strand"]:
            # Add left-facing arrows
            for arrow_loc in arrow_locs[1:]:
                ax.annotate(
                    "",
                    xy=(arrow_loc, transcript_idx),
                    xytext=(p[0], transcript_idx),
                    arrowprops=arrowprops,
                    annotation_clip=True,
                )
        else:
            # Add right-facing arrows
            for arrow_loc in arrow_locs[:-1]:
                ax.annotate(
                    "",
                    xy=(arrow_loc, transcript_idx),
                    xytext=(p[1], transcript_idx),
                    arrowprops=arrowprops,
                    annotation_clip=True,
                )

        if step.info["Exons"]:
            for exon in step.info["Exons"]:
                p_exon = [
                    map_genome_point_to_range_points(
                        ranges, exon.start_pos.chrm, exon.start_pos.start
                    ),
                    map_genome_point_to_range_points(
                        ranges, exon.end_pos.chrm, exon.end_pos.end
                    ),
                ]
                if not points_in_window(p_exon):
                    continue

                ax.plot(
                    p_exon,
                    [transcript_idx, transcript_idx],
                    "-",
                    color=BLUE,
                    solid_capstyle="butt",
                    lw=4,
                )

        ax.text(
            sum(p)/2,
            transcript_idx + 0.1,
            step.info["Name"],
            color=INK,
            fontsize=annotation_fontsize,
            ha="center"
        )

        transcript_idx += 1
        transcript_idx_max = max(transcript_idx, transcript_idx_max)

    # set axis parameters
    clean_axis(ax)
    ax.set_ylim([transcript_idx_max * -0.1, 0.01+(transcript_idx_max * 1.01)])
    ax.set_title(transcript_filename, fontsize=8, loc="left", color=MUTED_INK)


# }}}


##Gene track methods
BUNDLED_GENE_TRACKS = {"GRCh38": "GRCh38_mane_select.bed.gz"}

# inches of figure height per row of the gene track, and for its title
GENE_ROW_INCHES = {True: 0.36, False: 0.24}
GENE_TRACK_TITLE_INCHES = 0.3
GENE_TRACK_MAX_ROWS = 6


class Gene:
    """One transcript from a gene track file (0-based half-open coordinates)"""

    def __init__(self, chrm, start, end, symbol, transcript, strand, cds_start, cds_end, exons):
        self.chrm = chrm
        self.start = start
        self.end = end
        self.symbol = symbol
        self.transcript = transcript
        self.strand = strand
        self.cds_start = cds_start
        self.cds_end = cds_end
        self.exons = exons

    @classmethod
    def from_line(cls, line):
        f = line.rstrip("\n").split("\t")
        exons = list(
            zip(
                [int(x) for x in f[8].split(",") if x],
                [int(x) for x in f[9].split(",") if x],
            )
        )
        return cls(
            strip_chr(f[0]), int(f[1]), int(f[2]), f[3], f[4], f[5], int(f[6]), int(f[7]), exons
        )

    def exon_numbers(self):
        """Exon numbers in transcript order, for exons in genomic order"""
        n = len(self.exons)
        return list(range(1, n + 1)) if self.strand != "-" else list(range(n, 0, -1))

    def __repr__(self):
        return "Gene({}, {}:{}-{})".format(self.symbol, self.chrm, self.start, self.end)


def get_gene_track_path(genes):
    """Path of a bundled gene track (by genome build) or of a custom file"""
    if genes in BUNDLED_GENE_TRACKS:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", BUNDLED_GENE_TRACKS[genes]
        )
    if os.path.isfile(genes):
        return genes
    raise SamplotError(
        "genes must be one of {} or the path of a gene track file, not {}".format(
            ", ".join(BUNDLED_GENE_TRACKS), genes
        )
    )


def get_genes(ranges, gene_track_path):
    """The genes overlapping the plotted ranges, in genomic order"""
    try:
        tbx = pysam.TabixFile(gene_track_path)
    except Exception as err:
        raise SamplotError(
            "Could not open gene track {}: {}".format(gene_track_path, err)
        ) from err

    genes = {}
    for r in ranges:
        chrm = strip_chr(r.chrm)
        for name in (chrm, "chr" + chrm):
            if name in tbx.contigs:
                for line in tbx.fetch(name, max(0, r.start), r.end):
                    gene = Gene.from_line(line)
                    genes[(gene.chrm, gene.start, gene.transcript)] = gene
    tbx.close()
    return [genes[key] for key in sorted(genes, key=lambda k: (k[0], k[1]))]


def map_interval_to_range_points(ranges, range_i, start, end):
    """Plot x positions (0-1) of the part of start-end inside one range, or
    None if they don't overlap"""
    r = ranges[range_i]
    lo = max(start, r.start)
    hi = min(end, r.end)
    if hi < lo:
        return None
    span = float(max(1, r.end - r.start))
    return [
        (range_i + (lo - r.start) / span) / len(ranges),
        (range_i + (hi - r.start) / span) / len(ranges),
    ]


def plan_gene_track(ranges, genes, detail, plot_width, fontsize):
    """Lays the genes out in rows so that neither they nor their labels overlap.

    Returns (rows, unlabelled): rows is a list of lists of items, each a dict
    with the gene, the range it is drawn in, its x extent and its label;
    unlabelled are items there was no room to label, which are drawn as bare
    bars on a row of their own at the bottom.
    """
    # rough width of one character as a fraction of the axis width
    char_width = (0.62 * fontsize / 72.0) / (plot_width * 0.86)

    items = []
    for gene in genes:
        for range_i, r in enumerate(ranges):
            if strip_chr(r.chrm) != gene.chrm:
                continue
            x = map_interval_to_range_points(ranges, range_i, gene.start, gene.end)
            if x is None:
                continue
            label = gene.symbol
            if detail:
                label += " " + gene.transcript
            label = (label + " ›") if gene.strand != "-" else ("‹ " + label)
            # keep the label inside the plot
            label_width = len(label) * char_width
            label_x = min(x[0], 1.0 - label_width)
            if range_i > 0:
                # clear of the gap drawn between two ranges
                label_x = max(label_x, float(range_i) / len(ranges) + char_width)
            items.append(
                {
                    "gene": gene,
                    "range_i": range_i,
                    "x": x,
                    "label": label,
                    "label_x": max(0.0, label_x),
                    "extent": [min(x[0], label_x), max(x[1], label_x + label_width)],
                }
            )

    items.sort(key=lambda item: item["extent"][0])
    rows = []
    row_ends = []
    unlabelled = []
    gap = 2 * char_width
    for item in items:
        for row_i in range(len(rows)):
            # a gene running on from one range into the next stays on its row
            same_gene = rows[row_i][-1]["gene"] is item["gene"]
            if item["extent"][0] >= row_ends[row_i] + (0 if same_gene else gap):
                rows[row_i].append(item)
                row_ends[row_i] = item["extent"][1]
                break
        else:
            if len(rows) < GENE_TRACK_MAX_ROWS:
                rows.append([item])
                row_ends.append(item["extent"][1])
            else:
                unlabelled.append(item)
    return rows, unlabelled


def gene_track_inches(rows, unlabelled, detail):
    """Figure height the gene track needs"""
    n_rows = max(1, len(rows)) + (1 if unlabelled else 0)
    return GENE_TRACK_TITLE_INCHES + GENE_ROW_INCHES[detail] * n_rows


def draw_gene(ax, ranges, item, y, detail, fontsize, label=True):
    """Draws one gene on its row (rows are one y unit tall, y is the bottom)"""
    gene = item["gene"]
    x0, x1 = item["x"]

    if not detail:
        ax.add_patch(
            mpatches.Rectangle(
                (x0, y + 0.2), max(x1 - x0, 0.0015), 0.2, facecolor=BLUE,
                edgecolor="none",
            )
        )
        text_y = y + 0.5
    else:
        mid = y + 0.42
        ax.plot([x0, x1], [mid, mid], "-", color=BLUE, lw=0.6, solid_capstyle="butt")
        # direction of transcription along the introns
        n_arrows = int((x1 - x0) / 0.03)
        if n_arrows > 0:
            ax.plot(
                np.linspace(x0, x1, n_arrows + 2)[1:-1],
                [mid] * n_arrows,
                linestyle="none",
                marker=">" if gene.strand != "-" else "<",
                markersize=1.8,
                color=BLUE,
            )
        last_number_x = None
        numbers = gene.exon_numbers()
        for (exon_start, exon_end), number in zip(gene.exons, numbers):
            exon_x = map_interval_to_range_points(ranges, item["range_i"], exon_start, exon_end)
            if exon_x is None:
                continue
            # thin box for the whole exon, thick for its coding part
            ax.add_patch(
                mpatches.Rectangle(
                    (exon_x[0], mid - 0.08), max(exon_x[1] - exon_x[0], 0.0008), 0.16,
                    facecolor=BLUE, edgecolor="none", zorder=3,
                )
            )
            coding = [max(exon_start, gene.cds_start), min(exon_end, gene.cds_end)]
            if coding[1] > coding[0]:
                cds_x = map_interval_to_range_points(ranges, item["range_i"], coding[0], coding[1])
                if cds_x is not None:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (cds_x[0], mid - 0.17), max(cds_x[1] - cds_x[0], 0.0008), 0.34,
                            facecolor=BLUE, edgecolor="none", zorder=3,
                        )
                    )
            # exon numbers, where there is room for them
            centre = (exon_x[0] + exon_x[1]) / 2.0
            if last_number_x is None or abs(centre - last_number_x) > 0.018:
                ax.text(
                    centre, y + 0.2, str(number), fontsize=fontsize * 0.7,
                    color=MUTED_INK, ha="center", va="top",
                )
                last_number_x = centre
        text_y = y + 0.66

    if label:
        ax.text(
            item["label_x"], text_y, item["label"], fontsize=fontsize, color=INK,
            ha="left", va="bottom",
        )


def plot_gene_track(ax, ranges, rows, unlabelled, detail, title, fontsize, sv=None, sv_type=None):
    """Draws the planned gene track"""
    # genes there was no room to label share a row of their own at the bottom
    extra_row = 1 if unlabelled else 0
    n_rows = max(1, len(rows)) + extra_row
    clean_axis(ax)
    ax.set_ylim([0, n_rows])
    ax.set_title(title, fontsize=8, loc="left", color=MUTED_INK)

    # the first row is drawn at the top
    for row_i, row in enumerate(rows):
        for item in row:
            draw_gene(ax, ranges, item, n_rows - 1 - row_i, detail, fontsize)
    for item in unlabelled:
        draw_gene(ax, ranges, item, 0, False, fontsize, label=False)

    if not rows:
        ax.text(
            0.5, 0.5, "No genes in view", ha="center", va="center",
            fontsize=fontsize, color=MUTED_INK,
        )
    elif unlabelled:
        ax.text(
            0.0, 0.5, "{} more genes, not labelled".format(len(unlabelled)),
            ha="left", va="bottom", fontsize=fontsize, color=MUTED_INK,
        )

    # carry the breakpoints down through the genes
    if sv and sv_type:
        for b in get_breakpoint_points(sv, ranges):
            if b is not None:
                ax.axvline(
                    b, color=sv_color(sv_type), linestyle="--", linewidth=0.8,
                    alpha=0.8, zorder=1,
                )

    # break the track when we have multiple ranges
    for i in range(1, len(ranges)):
        ax.axvline(x=float(i) / len(ranges), color="white", linewidth=5, zorder=5)


########################################################################
# main block
########################################################################
def create_figure(options):
    """Builds the samplot figure described by options and returns it.

    options is a PlotOptions. Nothing is written to
    disk; this is the shared core of the command line and of plot_region().
    Raises SamplotError if the plot can't be made.
    """
    random.seed(options.random_seed)
    for read_type in READ_TYPES_USED:
        READ_TYPES_USED[read_type] = False

    if (options.annotation_files 
            and options.annotation_filenames 
            and len(options.annotation_files) != len(options.annotation_filenames)):
        raise SamplotError("annotation filenames do not match annotation files")

    for bam in options.bams:
        if ".cram" in bam:
            if not options.reference:
                raise SamplotError(
                    "Missing argument reference (-r/--reference) required for CRAM"
                )

    if not (len(options.chrom) == len(options.start) == len(options.end)):
        raise SamplotError(
            "The number of chromosomes ({}), starts ({}), and ends ({}) do not match.".format(
                len(options.chrom),
                len(options.start),
                len(options.end)
            )
        )

    sv = [
        genome_interval(strip_chr(chrom), start, end)
        for chrom, start, end in zip(options.chrom, options.start, options.end)
    ]
    # set up plot
    plot_height, plot_width, window, ranges = set_plot_dimensions(sv, options)

    # plan the gene track, which sets how tall it needs to be
    gene_inches = 0
    if options.genes:
        gene_detail = (
            sum(r.end - r.start for r in ranges) <= options.gene_detail_max_size
        )
        gene_track_path = get_gene_track_path(options.genes)
        gene_rows, genes_unlabelled = plan_gene_track(
            ranges,
            get_genes(ranges, gene_track_path),
            gene_detail,
            plot_width,
            options.annotation_fontsize,
        )
        gene_inches = gene_track_inches(gene_rows, genes_unlabelled, gene_detail)
        if not options.plot_height:
            plot_height += gene_inches

    # read alignment data
    read_data, max_coverage = get_read_data(ranges, options)
    if options.max_coverage:
        max_coverage = options.max_coverage

    # The figure is built without pyplot, so no global matplotlib state
    # (backend, current figure, rcParams) is touched: safe to call from an
    # app or a notebook
    with matplotlib.rc_context(STYLE_RC):
        fig = Figure(figsize=(plot_width, plot_height))

        # set up grid organizer
        grid, num_ax = create_gridspec(
            options, read_data, fig=fig, gene_inches=gene_inches
        )
        current_axis_idx = 0

        # plot variant on top
        if options.sv_type:
            current_axis_idx = create_variant_plot(
                grid,
                current_axis_idx,
                sv,
                options.sv_type,
                ranges,
                options.start_ci,
                options.end_ci,
            )

        # Plot each sample
        current_axis_idx = plot_samples(
            ranges, read_data, grid, current_axis_idx, num_ax, options, max_coverage, sv=sv
        )
        plot_legend(fig, options.legend_fontsize, options.marker_size)

        # Plot annotation files
        if options.annotation_files:
            plot_annotations(options, ranges, grid, current_axis_idx)

        # Plot the gene track
        if options.genes:
            gene_ax_i = num_ax - (2 if options.transcript_file else 1)
            if options.genes in BUNDLED_GENE_TRACKS:
                gene_title = "{} MANE Select transcripts".format(options.genes)
            else:
                gene_title = os.path.basename(gene_track_path)
            plot_gene_track(
                fig.add_subplot(grid[gene_ax_i]),
                ranges,
                gene_rows,
                genes_unlabelled,
                gene_detail,
                gene_title,
                options.annotation_fontsize,
                sv=sv,
                sv_type=options.sv_type,
            )

        # Plot sorted/bgziped/tabixed transcript file
        if options.transcript_file:
            plot_transcript(options, ranges, grid)

        fig.tight_layout(pad=0.8, h_pad=0.1, w_pad=0.1)

    return fig


def default_options():
    """A PlotOptions holding the default of every setting"""
    return PlotOptions()


def plot_region(chrom, start, end, bams, sv_type=None, output_file=None, **kwargs):
    """Plots a region or structural variant and returns the matplotlib Figure.

    This is the library equivalent of `samplot plot`:

        import samplot
        fig = samplot.plot_region("chr4", 115928730, 115931875,
                                  bams=["proband.cram", "mum.cram"],
                                  sv_type="DEL", reference="genome.fa",
                                  titles=["Proband", "Mother"])
        fig.savefig("del.png", dpi=150)

    chrom, start, end: the variant or region. Pass two-item lists for a
        variant whose breakpoints are on different chromosomes.
    bams: list of BAM/CRAM paths (CRAM needs reference=...).
    sv_type: eg "DEL", "DUP", "INV", "BND". If None the region is plotted
        as given with no variant bar or padding.
    output_file: if given the figure is also saved there (format from the
        extension, resolution from dpi=...).
    kwargs: any other field of PlotOptions (the long names of the
        `samplot plot` options), eg window=1000, coverage_only=True,
        transcript_file="genes.gff3.gz", max_coverage=60.

    No global matplotlib state is used, so this can be called from a web app
    or notebook. To get image bytes rather than a file:

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150)

    Raises SamplotError if the plot can't be made and TypeError for an
    unknown option.
    """

    def as_list(value):
        return list(value) if isinstance(value, (list, tuple)) else [value]

    options = PlotOptions(**kwargs)
    options.chrom = [str(c) for c in as_list(chrom)]
    options.start = [int(s) for s in as_list(start)]
    options.end = [int(e) for e in as_list(end)]
    options.bams = [str(b) for b in as_list(bams)]
    options.sv_type = sv_type
    options.output_file = output_file
    for key in ("titles", "annotation_files", "annotation_filenames"):
        if getattr(options, key) is not None:
            setattr(options, key, as_list(getattr(options, key)))

    fig = create_figure(options)
    if output_file:
        fig.savefig(output_file, dpi=options.dpi)
    return fig


def plot(parser, options, extra_args=None):
    """
    Command line entry point for `samplot plot`.

    To support translocations, the SVs are specified as an array of 
    genome_interval. For now we let that array be size 1 or 2.
    """
    options = PlotOptions.from_namespace(options)
    if options.debug:
        logger.setLevel(logging.DEBUG)
    
    if options.print_args or options.json_only:
        print_arguments(options)
        if options.json_only:
            sys.exit(0)

    if options.output_file:
        output_file = options.output_file
    else:
        if not os.path.isdir(options.output_dir):
            os.mkdir(options.output_dir)
        name_fields = [
            options.sv_type,
            "-".join(options.chrom),
            "-".join([str(s) for s in options.start]),
            "-".join([str(e) for e in options.end]),
        ]
        if options.sv_type:
            output_file = os.path.join(options.output_dir, "_".join(name_fields))
        else:
            output_file = os.path.join(options.output_dir, "_".join(name_fields[1:]))

    try:
        fig = create_figure(options)
    except SamplotError as e:
        logger.error(str(e))
        sys.exit(1)

    try:
        fig.savefig(output_file, dpi=options.dpi)
    except Exception as e:
        logger.error(
            "Failed to save figure {}".format(output_file)
        )
        print(e)
# }}}
