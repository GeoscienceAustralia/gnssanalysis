import logging as _logging
import os as _os

import matplotlib.dates as _mdates
import matplotlib.units as _munits
import numpy as _np
import pandas as _pd
import plotext as plx
from matplotlib import cm as _cm

from . import gn_datetime as _gn_datetime


def plot_vec(df: _pd.DataFrame, axes: list, axes_idx: list, legend: bool = True):
    """Function to plot columnar (single-lvl column names) dataframe to axes list
    according to axes index provided (needed in case of non-alphabetic order of column names and axes desired).
    Example snippet below:

    fig = plt.figure(figsize=(10,5)
    gs = fig.add_gridspec(3, hspace=0.2)
    ax = gs.subplots(sharex=True, sharey=False)
    plot_vec(axes=ax,df=a,axes_idx=[1,2,0])
    fig.savefig('blah.png')
    """
    converter = _mdates.DateConverter()
    _munits.registry[_np.datetime64] = converter
    hours = (df.index.values[-1] - df.index.values[0]) // 3600
    locator = _mdates.HourLocator(interval=(hours // 24 + 1) * 3)
    formatter = _mdates.ConciseDateFormatter(locator, show_offset=True)  # offset with date at the bottom
    formatter.zero_formats[3] = "%d-%b"  # the 00:00 label formatter

    df.index = _gn_datetime.j20002datetime(df.index.values)  # converting J2000 seconds to datetime64
    components = df.columns.levels[0]
    sv_list = df.columns.levels[1]

    styl_list = ["solid", "dotted", "dashed", "dashdot"]

    cmap = _cm.gist_rainbow  # hsv in the original plotting script
    for i in range(len(axes_idx)):
        df_c = df[components[i]]
        for j in range(len(sv_list)):
            axes[axes_idx[i]].plot(
                df_c[sv_list[j]],
                label=sv_list[j] if (i == 0 and legend) else "",
                ls=styl_list[j % 4],
                color=cmap(j / len(sv_list)),
            )

        axes[axes_idx[i]].set_ylabel(f"{components[i]} (cm)")

    axes[0].xaxis.set_major_locator(locator)
    axes[0].xaxis.set_major_formatter(formatter)


def diff2plot(diff, kind=None, title="Unnamed plot"):
    """Function to plot graph to the terminal. Can be scatter or bar plot (Initial test functionality)
    Works only with plotext 4.2, not above"""
    # try:
    #     import plotext as plt
    # except ModuleNotFoundError:
    #     # Error handling
    #     pass
    plx.clear_plot()
    diff = diff.round(2)
    if kind == "bar":
        # expects a series with index being string names
        mask0 = (diff.values != 0) & ~_np.isnan(diff.values)
        if mask0.any():
            plx.bar(diff.index.values[mask0].tolist(), diff.values[mask0].tolist(), orientation="h", width=0.3)
            plx.vertical_line(coordinate=0)
            plx.plotsize(100, diff.shape[0] * 2 + 3)
        else:
            return None
    else:
        mask0 = ((diff.values != 0) & ~_np.isnan(diff.values)).any(axis=0)
        if mask0.any():
            cols = diff.columns[mask0]
            x1 = _gn_datetime.j20002datetime(diff.index.values).astype(str)
            plx.datetime._datetime_form = "%Y-%m-%dT%H:%M:%S" # change parsed format to numpy-like
            for i in range(cols.shape[0]):
                plx.scatter_date(x1, diff[cols[i]].to_list(), color=i, marker="hd", label=diff.columns[i])
                plx.plotsize(100, 30 + 3)
        else:
            return None

    plx.title(title)
    plx.limit_size(limit_xsize=False, limit_ysize=False)
    plx.show()


def id_df2html(id_df, outdir=None, verbose=False):
    import plotly.express as px

    fig = px.scatter_geo(
        id_df,
        lon="LON",
        lat="LAT",
        title=id_df.attrs["infomsg"],
        size="SIZE",
        color="PATH",
        size_max=18,
        hover_name="CODE",  # column added to hover information
        hover_data=["PT", "DOMES"],
        projection="natural earth",
    )
    filename = (
        "gather_map" if len(id_df.attrs["info"]) > 1 else _os.path.basename(list(id_df.attrs["info"])[0])
    ) + ".html"
    save_path = _os.path.join(_os.path.curdir if outdir is None else outdir, filename)
    if verbose:
        _logging.info(msg=f"saving html to {_os.path.abspath(save_path)}")
    fig.write_html(save_path)


def racplot(rac_unstack, output=None):
    import matplotlib.pyplot as plt

    extended_plot = rac_unstack.attrs["hlm_mode"] is not None
    fig = plt.figure(figsize=(10, 5 + 5 * extended_plot), dpi=100)
    gs = fig.add_gridspec(3 + 3 * extended_plot, hspace=0.2)
    ax = gs.subplots(sharex=True, sharey=False)
    plot_vec(axes=ax, df=rac_unstack * 100, axes_idx=[1, 2, 0])  # meter to cm

    if extended_plot:  # append hlm residuals plot if transformation has been selected
        line = plt.Line2D([0, 1], [0.49, 0.49], transform=fig.transFigure, color="black", ls="--")
        plt.text(
            0.015,
            0.485,
            va="top",
            s=f"{rac_unstack.attrs['sp3_b']} - {rac_unstack.attrs['sp3_b']}"
            + f" (HLM in {rac_unstack.attrs['hlm_mode']})\nResiduals shown are in ECI frame",
            rotation=90,
            transform=fig.transFigure,
            fontfamily="monospace",
        )
        fig.add_artist(line)

        plot_vec(axes=ax, df=rac_unstack.attrs["hlm"][1].RES.unstack() * 100000, axes_idx=[3, 4, 5], legend=False)

        hlm = rac_unstack.attrs["hlm"][0][0].reshape(-1).tolist()
        hlm_txt = (
            "HLM coeffiecients:\n\n"
            + f"Tx {hlm[0]:13.5e}\nTy {hlm[1]:13.5e}\nTz {hlm[2]:13.5e}\n"
            + f"Rx {hlm[0]:13.5e}\nRy {hlm[1]:13.5e}\nRz {hlm[2]:13.5e}\n"
            + f"μ  {hlm[0]:13.5e}"
        )
        plt.text(0.815, 0.485, s=hlm_txt, va="top", transform=fig.transFigure, fontfamily="monospace")

    fig.suptitle(
        rac_unstack.attrs["sp3_a"]
        + " - "
        + rac_unstack.attrs["sp3_b"]
        + (f" (HLM in {rac_unstack.attrs['hlm_mode']})" if extended_plot else ""),
        y=0.92,
    )
    fig.patch.set_facecolor("w")  # white background (non-transparent)
    fig.legend(bbox_to_anchor=(0.955, 0.89), ncol=2, fontsize=8)
    plt.subplots_adjust(right=0.8)

    fig.savefig(
        f"{rac_unstack.attrs['sp3_a'].split('.')[0]}-{rac_unstack.attrs['sp3_b'].split('.')[0]}.pdf"
        if output is None
        else output
    )
