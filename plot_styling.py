import matplotlib as mpl
import seaborn as sns


def set_plot_styling():
    sns.set_theme()
    sns.set_context("paper")
    mpl.style.use(["seaborn-white", "seaborn-paper"])
    style = {
        "pgf.rcfonts": False,
        "pgf.texsystem": "pdflatex",
        "text.usetex": False,
        "font.family": "serif",
        "font.serif": "Times",
        "axes.linewidth": 0.75,
    }
    mpl.rcParams.update(style)
