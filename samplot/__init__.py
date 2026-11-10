"""samplot: plots of sequencing evidence for structural and copy number variants.

Use from the command line (`samplot plot`, `samplot vcf`) or as a library:

    import samplot
    fig = samplot.plot_region("chr4", 115928730, 115931875,
                              bams=["sample.bam"], sv_type="DEL")
    fig.savefig("del.png")
"""
__version__ = "1.4.0"

from .samplot import SamplotError, create_figure, default_options, plot_region

__all__ = [
    "SamplotError",
    "create_figure",
    "default_options",
    "plot_region",
    "__version__",
]
