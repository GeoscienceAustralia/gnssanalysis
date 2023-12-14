# gnssanalysis
The package encompasses various GNSS-related functionality such as efficient reading and writing GNSS files (e.g. SINEX, SP3, CLK, IONEX and many others), advanced analysis and comparison, various coordinate transformations including geodetic frame rotations, predictions and combinations.
Package  Solver.

## Install

```bash
pip install gnssanalysis
```



### File formats supported
- BLQ
- BSX/BIA
- CLK
- ERP
- IONEX
- NANU
- RINEX
- SINEX (including discontinuity, post-seismic file formats)
- SP3
- TROP
- GINAN proprietary formats: PEA partials, POD output, STEC and TRACE

# Standalone utilities
There is a set of standalone utilities installed together with the module module which are build on top of gnssanalysis.

## diffutil
A utility originally created for automated testing of the computed GNSS files relative to the known good solution files.
The simplest use case for the `diffutil` is to call it with two GNSS files specified after `-i`:
```bash
diffutil -i file1 file2
```
`diffutil` parses files' extensions and automatically calls a command needed, e.g. `sp3` for `file.sp3`. It is also possible to specify the command manually in case file extensions are non-standard or missing.

```bash
diffutil -i file1 file2 sp3 # sp3 is a command inside diffutil
```

## snxmap
Reads any number of sinex files given, specifically the SITE/ID block and creates an interactive html map with all the stations plotted. Every file will get a unique color marker, with decreasing size for each additional marker, constructing "lollipops" at common stations. This allows seeing intersections of stations within files to be easily seen.
The sinex files may also be compressed (either .Z or .gz)

How to use:
```bash
snxmap snxfile1 snxfile2.Z snxfile3.gz
```
`-o path_to_output`

## sp3merge
Merges any number of sp3 files together creating sp3 file of any length. Could also accept clk files to populate merged sp3 with clock offset values.

How to use:
```bash
sp3merge -s file1.sp3 file2.sp3.Z file3.sp3.gz -c file1.clk file2.clk.Z file3.clk.gz
```
## log2snx
A utility to parse collection of igs-format station files and create a sinex file with required station information - location, station hardware etc. 

How to use:
```bash
log2snx -l "~/logfiles/*/*log"
```


## trace2mongo (needs update)
Converts tracefile to the mongo database that is compatible with Ginan's mongo output for EDA

## gnss-filename
Determines appropriate filename for GNSS based on the content


## orbq
Compares two sp3 files and outputs statistics to the terminal window.

How to use:
```bash
orbq -i IGS2R03FIN_20191990000_01D_05M_ORB.SP3 TUG0R03FIN_20191990000_01D_05M_ORB.SP3
```
Result:
```
PRN     R_RMS   A_RMS   C_RMS   dX_RMS  dY_RMS  dZ_RMS  1D_MEAN 3D_RMS
E01     0.01429 0.00629 0.0077  0.01116 0.00894 0.00993 0.01001 0.01741
E02     0.01931 0.01063 0.01009 0.01564 0.01332 0.01288 0.01395 0.02424
E03     0.0207  0.01334 0.00935 0.0139  0.01909 0.01168 0.01489 0.02634
...
=======================================================================
        R_RMS   A_RMS   C_RMS   dX_RMS  dY_RMS  dZ_RMS  1D_MEAN 3D_RMS
AVG     0.01155 0.01536 0.01346 0.01504 0.01443 0.01277 0.01408 0.02485
STD     0.01045 0.02978 0.01767 0.02065 0.02253 0.01809 0.02017 0.03522
RMS     0.01553 0.03334 0.02212 0.02543 0.02663 0.02205 0.02449 0.04292

```

## clkq
Compares two clk files and outputs statistics to the terminal window.
How to use:
```bash
clkq -i IGS2R03FIN_20191990000_01D_30S_CLK.CLK COD0R03FIN_20191990000_01D_30S_CLK.CLK
```
Result:
```
INFO:root:filtering using 10.00 hard cutoff on the detrended data. [99.99% data left]
INFO:root:filtering using 3.00*sigma cut on the detrended data. [99.99% data left]
INFO:root:clkq
CODE    AVG     STD     RMS
E01     3.6127  0.0469  3.613
E02     3.5929  0.0469  3.5932
E03     3.6221  0.042   3.6224
...
==============================
GNSS    AVG     STD     RMS
E       3.6075  0.0595  3.608
G       -0.0621 0.0663  0.0909
R       6.1658  0.3282  6.1746
```
### Reject satellites
You are able to provide a regex to reject satellites, e.g.:
```bash
clkq -i IGS2R03FIN_20191990000_01D_30S_CLK.CLK COD0R03FIN_20191990000_01D_30S_CLK.CLK --reject "G18"
```
Result:
```
NFO:root:Excluding satellites based on regex expression: 'G18'
INFO:root:Removed the following satellites from first file: '['G18']'
INFO:root:Removed the following satellites from second file: '['G18']'
INFO:root:filtering using 10.00 hard cutoff on the detrended data. [100.00% data left]
INFO:root:filtering using 3.00*sigma cut on the detrended data. [100.00% data left]
INFO:root:clkq
CODE    AVG     STD     RMS
E01     3.6127  0.0469  3.613
E02     3.5929  0.0469  3.5932
...
```
### Normalisation Procedure
And you are also able to provide a normalisation parameter to choose how the data is normalised (daily, epoch), e.g.:
```bash
clkq -i IGS2R03FIN_20191990000_01D_30S_CLK.CLK COD0R03FIN_20191990000_01D_30S_CLK.CLK --norm "daily"
```
Result:
```
INFO:root::_clk_compare:using ['daily'] clk normalization
INFO:root:---removing common mode from clk 1---
INFO:root:Using daily offsets for common mode removal
INFO:root:---removing common mode from clk 2---
INFO:root:Using daily offsets for common mode removal
INFO:root:filtering using 10.00 hard cutoff on the detrended data. [99.99% data left]
INFO:root:filtering using 3.00*sigma cut on the detrended data. [99.99% data left]
INFO:root:clkq
CODE    AVG     STD     RMS
E01     -0.1216 0.0469  0.1303
E02     -0.1214 0.0469  0.1302
E03     -0.1216 0.042   0.1286
...
```
These can also be stacked:
```bash
clkq -i IGS2R03FIN_20191990000_01D_30S_CLK.CLK COD0R03FIN_20191990000_01D_30S_CLK.CLK --norm "daily"
```
Result:
```
INFO:root::_clk_compare:using ['daily', 'epoch'] clk normalization
INFO:root:---removing common mode from clk 1---
INFO:root:Using daily offsets for common mode removal
INFO:root:Using epoch normalization (mean gnss) offsets for common mode removal
INFO:root:---removing common mode from clk 2---
INFO:root:Using daily offsets for common mode removal
INFO:root:Using epoch normalization (mean gnss) offsets for common mode removal
INFO:root:filtering using 10.00 hard cutoff on the detrended data. [99.99% data left]
INFO:root:filtering using 3.00*sigma cut on the detrended data. [99.99% data left]
INFO:root:clkq
CODE    AVG     STD     RMS
E01     -0.0    0.0149  0.0149
E02     -0.0    0.0191  0.0191
E03     -0.0    0.0076  0.0076
...
```

# Some usage examples

## Combination of sinex solutions files
Combination of with a frame file projected to a midday of a date of interest
Usage examples:

- Daily comnination with frame_of_day centered at midday
```python
from gnssanalysis import gn_combi
daily_comb_neq = gn_combi.addneq(snx_filelist=_glob.glob('/data/cddis/2160/[!sio][!mig]*0.snx.Z'),frame_of_day=frame_of_day)
```

- Weekly combination with frame_of_day centered at week's center:
```python
weekly_comb_neq = gn_combi.addneq(snx_filelist=_glob.glob('/data/cddis/2160/[!sio][!mig]*.snx.Z'),frame_of_day=frame_of_day)
```

 The frame of day could be generated using a respective function from `gn_frame` module:
```python
from gnssanalysis import gn_frame, gn_datetime

frame_datetime = gn_datetime.gpsweeksec2datetime(2160,43200)

frame_of_day = gn_frame.get_frame_of_day(date_or_j2000=frame_datetime, itrf_path_or_df = '/data/cddis/itrf2014/ITRF2014-IGS-TRF.SNX.gz',discon_path_or_df='/data/cddis/itrf2014/ITRF2014-soln-gnss.snx',psd_path_or_df='/data/cddis/itrf2014/ITRF2014-psd-gnss.snx')
 ```