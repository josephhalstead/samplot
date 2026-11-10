"""samplot: plots of sequencing evidence for structural and copy number variants.

Use from the command line (`samplot plot`, `samplot vcf`) or as a library:

    import samplot
    fig = samplot.plot_region("chr4", 115928730, 115931875,
                              bams=["sample.bam"], sv_type="DEL")
    fig.savefig("del.png")
"""
# upstream (ryanlayer/samplot) has no 1.4.0: the +awgl local version marks
# this fork so the installed build is identifiable (samplot --version)
__version__ = "1.4.0+awgl"

from .samplot import (
    PlotOptions,
    SamplotError,
    create_figure,
    default_options,
    plot_region,
)

__all__ = [
    "PlotOptions",
    "SamplotError",
    "create_figure",
    "default_options",
    "plot_region",
    "__version__",
]
