"""ITRF2014+ postseismic deformation file"""
from .. import gn_io as _gn_io


def _get_psd_df(psd_snx_path):
    """Monument (PT) is taken to account for maximum consistency
    0-level columns are VAL and STD"""
    psd_df = (
        _gn_io.sinex._get_snx_vector(path_or_bytes=psd_snx_path, stypes=["EST"], format="raw")
        .droplevel("SOLN")
        .droplevel(1, axis=1)
    )
    # monument is always A in psd file, cumcount is used as index if n
    # parameters of the same type are present for the same event
    psd_df["key"] = psd_df.groupby(["CODE_PT", "REF_EPOCH", "TYPE"]).cumcount()
    psd_df = psd_df.set_index(["key"], append=True).unstack(0)
    lvl_0 = psd_df.columns.droplevel(1).values
    lvl_1_2 = psd_df.columns.droplevel(0).to_series().str.split("_", expand=True).values.T
    psd_df.columns = [lvl_0, lvl_1_2[0], lvl_1_2[1]]
    return psd_df
