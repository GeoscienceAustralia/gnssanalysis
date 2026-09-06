"""Truncated IONEX snippets for unit tests.

The padded 73-longitude band follows ESA GIM products such as
ESA0OPSFIN_20240010000_01D_02H_GIM.INX: 16I5 records with the last line
holding 9 values and trailing spaces to 80 columns.
"""

# First latitude band of ESA0OPSFIN_20240010000_01D_02H_GIM.INX (87.5 N).
_ESA_LAT87_VALUES = (
    46,
    46,
    46,
    45,
    45,
    45,
    45,
    44,
    44,
    43,
    42,
    41,
    41,
    40,
    39,
    38,
    37,
    36,
    35,
    34,
    33,
    32,
    32,
    31,
    30,
    30,
    29,
    29,
    29,
    28,
    28,
    28,
    28,
    28,
    29,
    29,
    29,
    30,
    30,
    30,
    31,
    31,
    32,
    33,
    33,
    34,
    34,
    35,
    36,
    36,
    37,
    38,
    38,
    39,
    40,
    40,
    41,
    41,
    42,
    42,
    43,
    43,
    44,
    44,
    44,
    45,
    45,
    45,
    45,
    46,
    46,
    46,
    46,
)


def _record(body: str, label: str) -> str:
    return f"{body:<60}{label:<20}"


def _i5_lines(values, pad_to=80):
    lines = []
    for start in range(0, len(values), 16):
        chunk = values[start : start + 16]
        line = "".join(f"{value:5d}" for value in chunk)
        lines.append(line.ljust(pad_to) if pad_to else line)
    return lines


def _minimal_ionex(lat1, lat2, dlat, lon1, lon2, dlon, exponent, values, pad_data_lines=True):
    data_pad = 80 if pad_data_lines else 0
    header = [
        _record("     1.0            IONOSPHERE MAPS     GPS", "IONEX VERSION / TYPE"),
        _record(f"{exponent:6d}", "EXPONENT"),
        _record(f"{lat1:8.1f}{lat2:6.1f}{dlat:6.1f}", "LAT1 / LAT2 / DLAT"),
        _record(f"{lon1:8.1f}{lon2:6.1f}{dlon:6.1f}", "LON1 / LON2 / DLON"),
        _record("", "END OF HEADER"),
        _record("     1", "START OF TEC MAP"),
        _record("  2024     1     1     0     0     0", "EPOCH OF CURRENT MAP"),
        _record(f"{lat1:8.1f}{lon1:6.1f}{lon2:6.1f}{dlon:6.1f} 450.0", "LAT/LON1/LON2/DLON/H"),
    ]
    return ("\n".join(header + _i5_lines(values, pad_to=data_pad) + [_record("     1", "END OF TEC MAP")]) + "\n").encode(
        "ascii"
    )


# 1 latitude x 73 longitudes, last 16I5 line padded with spaces (ESA GIM layout).
ionex_esa_padded_lat_band = _minimal_ionex(
    lat1=87.5,
    lat2=87.5,
    dlat=-2.5,
    lon1=-180.0,
    lon2=180.0,
    dlon=5.0,
    exponent=-1,
    values=_ESA_LAT87_VALUES,
    pad_data_lines=True,
)

# Same grid without right-padding on short data lines.
ionex_esa_unpadded_lat_band = _minimal_ionex(
    lat1=87.5,
    lat2=87.5,
    dlat=-2.5,
    lon1=-180.0,
    lon2=180.0,
    dlon=5.0,
    exponent=-1,
    values=_ESA_LAT87_VALUES,
    pad_data_lines=False,
)

# Tiny 1 x 3 map for a simple numeric check.
ionex_tiny_unpadded = _minimal_ionex(
    lat1=0.0,
    lat2=0.0,
    dlat=2.5,
    lon1=0.0,
    lon2=10.0,
    dlon=5.0,
    exponent=-1,
    values=(10, 20, 30),
    pad_data_lines=False,
)
