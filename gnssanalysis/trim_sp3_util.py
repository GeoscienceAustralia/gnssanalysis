# SP3 file trimming / editing utility. Intended for test purposes only, including creation of SP3 test data for unit
# tests, avoiding the need to store excessively large files in the repo.

import argparse
from datetime import timedelta
from typing import Optional
from gnssanalysis.filenames import convert_nominal_span, determine_properties_from_filename
from gnssanalysis.gn_io.sp3 import (
    filter_by_svs,
    read_sp3,
    trim_to_first_n_epochs,
    write_sp3,
    remove_offline_sats,
    trim_df,
)
import logging

logger = logging.getLogger(__name__)


#### Configuration ####

# Constrain to x SVs, specific SV names, both, or neither
trim_to_sv_names: Optional[list[str]] = None  # ["G02", "G03", "G19"]
trim_to_sv_count: Optional[int] = None  # 1
trim_to_sat_letter: Optional[str] = None  # "E"

# How many epochs to include in the trimmed file (offset from start)
trim_to_num_epochs: Optional[int] = None  # 3

# Trim off this time onwards, leaving only the start of the file
trim_to_first_n_time: Optional[timedelta] = None

drop_offline_sats: bool = False

#### End configuration ####


def trim_sp3(src_path: str, dest_path: str) -> None:
    # Default to hardcoded paths, unless called with paths.

    filename = src_path.rsplit("/")[-1]
    print(f"Filename is: {filename}")

    # Determine sample rate (needed for trimming)
    # Raw data would be: determine_sp3_name_props() - that retrieves in seconds. But we want to be more generally applicable, so not just SP3 here ideally.
    sample_rate_raw: timedelta | None = convert_nominal_span(
        determine_properties_from_filename(filename)["sampling_rate"], non_timed_span_output="none"
    )
    if sample_rate_raw is None:
        print("Warning: failed to determine sample rate, may be a non-timed unit i.e. 'U'")
    else:
        # sample_rate: timedelta = sample_rate_raw
        print(f"sample_rate is: {sample_rate_raw}")

    # Load
    print("Loading SP3 into DataFrame (Pos data only, strict mode, warn only)...")
    sp3_df = read_sp3(
        src_path,
        # check_header_vs_filename_vs_content_discrepancies=True,
        # continue_on_discrepancies=True,
    )
    print("Read done.")

    # Trim to first x epochs
    if trim_to_num_epochs is not None:
        print(f"Trimming to first {trim_to_num_epochs} epochs")
        sp3_df = trim_to_first_n_epochs(sp3_df=sp3_df, epoch_count=trim_to_num_epochs, sp3_filename=filename)

    elif trim_to_first_n_time is not None:  # These two are mutually exclusive
        print(f"Trimming to first: {trim_to_first_n_time} (timedelta)")
        sp3_df = trim_df(sp3_df, keep_first_delta_amount=trim_to_first_n_time)

    # Filter to chosen SVs or number of SVs...
    print(
        "Applying SV filters (max count: "
        f"{trim_to_sv_count}, limit to names: {trim_to_sv_names}, limit to constellation: {trim_to_sat_letter})..."
    )
    sp3_df = filter_by_svs(
        sp3_df,
        filter_by_count=trim_to_sv_count,
        filter_by_name=trim_to_sv_names,
        filter_to_sat_letter=trim_to_sat_letter,
    )

    # Drop offline sats if requested
    if drop_offline_sats:
        print(f"Dropping offline sats (and updating header accordingly)...")
        sp3_df = remove_offline_sats(sp3_df)

    # Write out
    print(
        "Writing out new SP3 file... "
        'CAUTION: please check output header for consistency. It is based on metadata in .attrs["HEADER"], not the '
        "contents of the dataframe, and may not have been updated for all changes."
    )
    write_sp3(sp3_df, dest_path)

    # Test if we can successfully read that file...
    print("Testing re-read of the output file (strict mode, warn only)...")
    re_read = read_sp3(
        dest_path,
        pOnly=False,
        # check_header_vs_filename_vs_content_discrepancies=True,
        # continue_on_discrepancies=True
    )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Trim SP3 files for testing purposes")
    parser.add_argument(
        "-i",
        "--inpath",
        type=str,
        help="Input filepath",
        # nargs="+",
        required=True,
    )
    parser.add_argument("-o", "--outpath", type=str, help="Output filepath", required=True)  # default=None
    return parser.parse_args()


if __name__ == "__main__":
    # Arg parse example based on snx2map.py
    parsed_args = parse_arguments()
    print(f"Parsed args: in path: '{parsed_args.inpath}', out path: '{parsed_args.outpath}'")
    trim_sp3(src_path=parsed_args.inpath, dest_path=parsed_args.outpath)
