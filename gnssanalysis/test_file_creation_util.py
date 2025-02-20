from datetime import timedelta
from typing import Optional
from gnssanalysis.filenames import convert_nominal_span, determine_properties_from_filename
from gnssanalysis.gn_io.sp3 import filter_by_svs, read_sp3, trim_to_first_n_epochs, write_sp3, remove_offline_sats
import logging

logger = logging.getLogger(__name__)


#### Configuration ####

src_path = "IGS0DEMULT_20243181800_02D_05M_ORB.SP3"
dest_path = "IGS0DEMULT_20243181800_02D_05M_ORB.SP3-trimmed"

# Constrain to x SVs, specific SV names, both, or neither
trim_to_sv_names: Optional[list[str]] = ["G02", "G03", "G19"]
trim_to_sv_count: Optional[int] = None  # 1
trim_to_sat_letter: Optional[str] = None  # "E"

# How many epochs to include in the trimmed file (offset from start)
trim_to_num_epochs: Optional[int] = None  # 3

drop_offline_sats: bool = False

#### End configuration ####


filename = src_path.rsplit("/")[-1]
print(f"Filename is: {filename}")

# Determine sample rate (needed for trimming)
# Raw data would be: determine_sp3_name_props() - that retrieves in seconds. But we want to be more generally applicable, so not just SP3 here ideally.
sample_rate: timedelta = convert_nominal_span(determine_properties_from_filename(filename)["sampling_rate"])
print(f"sample_rate is: {sample_rate}")

# Load
print("Loading SP3 into DataFrame (Pos data only, strict mode, warn only)...")
sp3_df = read_sp3(
    src_path,
    check_header_vs_filename_vs_content_discrepancies=True,
    continue_on_discrepancies=True,
)
print("Read done.")

# Trim to first x epochs
if trim_to_num_epochs:
    print(f"Trimming to first {trim_to_num_epochs} epochs")
    sp3_df = trim_to_first_n_epochs(sp3_df=sp3_df, epoch_count=trim_to_num_epochs, sp3_filename=filename)

# Filter to chosen SVs or number of SVs...
print(
    "Applying SV filters (max count: "
    f"{trim_to_sv_count}, limit to names: {trim_to_sv_names}, limit to constellation: {trim_to_sat_letter})..."
)
sp3_df = filter_by_svs(
    sp3_df, filter_by_count=trim_to_sv_count, filter_by_name=trim_to_sv_names, filter_to_sat_letter=trim_to_sat_letter
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
    dest_path, pOnly=False, check_header_vs_filename_vs_content_discrepancies=True, continue_on_discrepancies=True
)
