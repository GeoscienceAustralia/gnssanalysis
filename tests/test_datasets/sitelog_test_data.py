# Central record of IGS site log test data sets to be shared across unit tests

# first dataset is a truncated version of file abmf_20240710.log

abmf_site_log_v1 = bytes(
    """
     ABMF Site Information Form (site log)
     International GNSS Service
     See Instructions at:
       https://files.igs.org/pub/station/general/sitelog_instr.txt

0.   Form

     Prepared by (full name)  : RGP TEAM
     Date Prepared            : 2024-07-10
     Report Type              : UPDATE
     If Update:
      Previous Site Log       : (ssss_ccyymmdd.log)
      Modified/Added Sections : (n.n,n.n,...)


1.   Site Identification of the GNSS Monument

     Site Name                : Aeroport du Raizet -LES ABYMES - Météo France
     Four Character ID        : ABMF
     Monument Inscription     : NONE
     IERS DOMES Number        : 97103M001
     CDP Number               : NONE
     Monument Description     : INOX TRIANGULAR PLATE ON TOP OF METALLIC PILAR
       Height of the Monument : 2.0 m
       Monument Foundation    : ROOF
       Foundation Depth       : 4.0 m
     Marker Description       : TOP AND CENTRE OF THE TRIANGULAR PLATE
     Date Installed           : 2008-07-15T00:00Z
     Geologic Characteristic  : 
       Bedrock Type           : 
       Bedrock Condition      : 
       Fracture Spacing       : 11-50 cm
       Fault zones nearby     : 
         Distance/activity    : 
     Additional Information   : 


2.   Site Location Information

     City or Town             : Les Abymes
     State or Province        : Guadeloupe (971)
     Country                  : Guadeloupe
     Tectonic Plate           : CARIBBEAN
     Approximate Position (ITRF)
       X coordinate (m)       : 2919786.0
       Y coordinate (m)       : -5383745.0
       Z coordinate (m)       : 1774604.0
       Latitude (N is +)      : +161544.30
       Longitude (E is +)     : -0613139.11
       Elevation (m,ellips.)  : -25.0
     Additional Information   : 


3.   GNSS Receiver Information

3.1 Receiver Type            : LEICA GR25
     Satellite System         : GPS+GLO+GAL+BDS+SBAS
     Serial Number            : 1830399
     Firmware Version         : 4.31
     Elevation Cutoff Setting : 3 deg
     Date Installed           : 2019-03-13T17:00Z
     Date Removed             : 2019-04-15T12:00Z
     Temperature Stabiliz.    : none
     Additional Information   : L2C disabled

3.2 Receiver Type            : SEPT POLARX5
     Satellite System         : GPS+GLO+GAL+BDS+SBAS
     Serial Number            : 3013312
     Firmware Version         : 5.2.0
     Elevation Cutoff Setting : 0 deg
     Date Installed           : 2019-04-15T12:00Z
     Date Removed             : 2019-10-01T16:00Z
     Temperature Stabiliz.    : none
     Additional Information   : L2C disabled

3.3 Receiver Type            : SEPT POLARX5
     Satellite System         : GPS+GLO+GAL+BDS+SBAS
     Serial Number            : 3013312
     Firmware Version         : 5.3.0
     Elevation Cutoff Setting : 0 deg
     Date Installed           : 2019-10-01T16:00Z
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Temperature Stabiliz.    : none
     Additional Information   : L2C disabled

3.x  Receiver Type            : (A20, from rcvr_ant.tab; see instructions)
     Satellite System         : (GPS+GLO+GAL+BDS+QZSS+SBAS)
     Serial Number            : (A20, but note the first A5 is used in SINEX)
     Firmware Version         : (A11)
     Elevation Cutoff Setting : (deg)
     Date Installed           : (CCYY-MM-DDThh:mmZ)
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Temperature Stabiliz.    : (none or tolerance in degrees C)
     Additional Information   : (multiple lines)


4.   GNSS Antenna Information

4.1  Antenna Type             : AERAT2775_43    SPKE
     Serial Number            : 5546
     Antenna Reference Point  : TOP
     Marker->ARP Up Ecc. (m)  : 000.0500
     Marker->ARP North Ecc(m) : 000.0000
     Marker->ARP East Ecc(m)  : 000.0000
     Alignment from True N    : 0 deg
     Antenna Radome Type      : SPKE
     Radome Serial Number     : NONE
     Antenna Cable Type       : 
     Antenna Cable Length     : 30.0 m
     Date Installed           : 2008-07-15T00:00Z
     Date Removed             : 2009-10-15T20:00Z
     Additional Information   : 

4.2  Antenna Type             : TRM55971.00     NONE
     Serial Number            : 1440911917
     Antenna Reference Point  : BAM
     Marker->ARP Up Ecc. (m)  : 000.0000
     Marker->ARP North Ecc(m) : 000.0000
     Marker->ARP East Ecc(m)  : 000.0000
     Alignment from True N    : 0 deg
     Antenna Radome Type      : NONE
     Radome Serial Number     : 
     Antenna Cable Type       : 
     Antenna Cable Length     : 30.0 m
     Date Installed           : 2009-10-15T20:00Z
     Date Removed             : 2012-01-24T12:00Z
     Additional Information   : 

4.3  Antenna Type             : TRM57971.00     NONE
     Serial Number            : 1441112501
     Antenna Reference Point  : BAM
     Marker->ARP Up Ecc. (m)  : 000.0000
     Marker->ARP North Ecc(m) : 000.0000
     Marker->ARP East Ecc(m)  : 000.0000
     Alignment from True N    : 0 deg
     Antenna Radome Type      : NONE
     Radome Serial Number     : 
     Antenna Cable Type       : 
     Antenna Cable Length     : 30.0 m
     Date Installed           : 2012-01-24T12:00Z
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Additional Information   : 

4.x  Antenna Type             : (A20, from rcvr_ant.tab; see instructions)
     Serial Number            : (A*, but note the first A5 is used in SINEX)
     Antenna Reference Point  : (BPA/BCR/XXX from "antenna.gra"; see instr.)
     Marker->ARP Up Ecc. (m)  : (F8.4)
     Marker->ARP North Ecc(m) : (F8.4)
     Marker->ARP East Ecc(m)  : (F8.4)
     Alignment from True N    : (deg; + is clockwise/east)
     Antenna Radome Type      : (A4 from rcvr_ant.tab; see instructions)
     Radome Serial Number     : 
     Antenna Cable Type       : (vendor & type number)
     Antenna Cable Length     : (m)
     Date Installed           : (CCYY-MM-DDThh:mmZ)
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Additional Information   : (multiple lines)
     """,
    "utf-8",
)

abmf_site_log_v2 = bytes(
    """
     ABMF00GLP Site Information Form (site log v2.0)
     International GNSS Service
     See Instructions at:
       https://files.igs.org/pub/station/general/sitelog_instr_v2.0.txt

0.   Form

     Prepared by (full name)  : RGP TEAM
     Date Prepared            : 2024-07-10
     Report Type              : UPDATE
     If Update:
      Previous Site Log       : (ssssmrccc_ccyymmdd.log)
      Modified/Added Sections : (n.n,n.n,...)


1.   Site Identification of the GNSS Monument

     Site Name                : Aeroport du Raizet -LES ABYMES - Météo France
     Nine Character ID        : ABMF00GLP
     Monument Inscription     : NONE
     IERS DOMES Number        : 97103M001
     CDP Number               : NONE
     Monument Description     : INOX TRIANGULAR PLATE ON TOP OF METALLIC PILAR
       Height of the Monument : 2.0 m
       Monument Foundation    : ROOF
       Foundation Depth       : 4.0 m
     Marker Description       : TOP AND CENTRE OF THE TRIANGULAR PLATE
     Date Installed           : 2008-07-15T00:00Z
     Geologic Characteristic  : 
       Bedrock Type           : 
       Bedrock Condition      : 
       Fracture Spacing       : 11-50 cm
       Fault zones nearby     : 
         Distance/activity    : 
     Additional Information   : 


2.   Site Location Information

     City or Town             : Les Abymes
     State or Province        : Guadeloupe (971)
     Country or Region        : GLP
     Tectonic Plate           : CARIBBEAN
     Approximate Position (ITRF)
       X coordinate (m)       : 2919786.0
       Y coordinate (m)       : -5383745.0
       Z coordinate (m)       : 1774604.0
       Latitude (N is +)      : +161544.30
       Longitude (E is +)     : -0613139.11
       Elevation (m,ellips.)  : -25.0
     Additional Information   : 


3.   GNSS Receiver Information

3.1 Receiver Type            : LEICA GR25
     Satellite System         : GPS+GLO+GAL+BDS+SBAS
     Serial Number            : 1830399
     Firmware Version         : 4.31
     Elevation Cutoff Setting : 3 deg
     Date Installed           : 2019-03-13T17:00Z
     Date Removed             : 2019-04-15T12:00Z
     Temperature Stabiliz.    : none
     Additional Information   : L2C disabled

3.2 Receiver Type            : SEPT POLARX5
     Satellite System         : GPS+GLO+GAL+BDS+SBAS
     Serial Number            : 3013312
     Firmware Version         : 5.2.0
     Elevation Cutoff Setting : 0 deg
     Date Installed           : 2019-04-15T12:00Z
     Date Removed             : 2019-10-01T16:00Z
     Temperature Stabiliz.    : none
     Additional Information   : L2C disabled

3.3 Receiver Type            : SEPT POLARX5
     Satellite System         : GPS+GLO+GAL+BDS+SBAS
     Serial Number            : 3013312
     Firmware Version         : 5.3.0
     Elevation Cutoff Setting : 0 deg
     Date Installed           : 2019-10-01T16:00Z
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Temperature Stabiliz.    : none
     Additional Information   : L2C disabled

3.x  Receiver Type            : (A20, from rcvr_ant.tab; see instructions)
     Satellite System         : (GPS+GLO+GAL+BDS+QZSS+SBAS)
     Serial Number            : (A20, but note the first A5 is used in SINEX)
     Firmware Version         : (A11)
     Elevation Cutoff Setting : (deg)
     Date Installed           : (CCYY-MM-DDThh:mmZ)
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Temperature Stabiliz.    : (none or tolerance in degrees C)
     Additional Information   : (multiple lines)


4.   GNSS Antenna Information

4.1  Antenna Type             : AERAT2775_43    SPKE
     Serial Number            : 5546
     Antenna Reference Point  : TOP
     Marker->ARP Up Ecc. (m)  : 000.0500
     Marker->ARP North Ecc(m) : 000.0000
     Marker->ARP East Ecc(m)  : 000.0000
     Alignment from True N    : 0 deg
     Antenna Radome Type      : SPKE
     Radome Serial Number     : NONE
     Antenna Cable Type       : 
     Antenna Cable Length     : 30.0 m
     Date Installed           : 2008-07-15T00:00Z
     Date Removed             : 2009-10-15T20:00Z
     Additional Information   : 

4.2  Antenna Type             : TRM55971.00     NONE
     Serial Number            : 1440911917
     Antenna Reference Point  : BAM
     Marker->ARP Up Ecc. (m)  : 000.0000
     Marker->ARP North Ecc(m) : 000.0000
     Marker->ARP East Ecc(m)  : 000.0000
     Alignment from True N    : 0 deg
     Antenna Radome Type      : NONE
     Radome Serial Number     : 
     Antenna Cable Type       : 
     Antenna Cable Length     : 30.0 m
     Date Installed           : 2009-10-15T20:00Z
     Date Removed             : 2012-01-24T12:00Z
     Additional Information   : 

4.3  Antenna Type             : TRM57971.00     NONE
     Serial Number            : 1441112501
     Antenna Reference Point  : BAM
     Marker->ARP Up Ecc. (m)  : 000.0000
     Marker->ARP North Ecc(m) : 000.0000
     Marker->ARP East Ecc(m)  : 000.0000
     Alignment from True N    : 0 deg
     Antenna Radome Type      : NONE
     Radome Serial Number     : 
     Antenna Cable Type       : 
     Antenna Cable Length     : 30.0 m
     Date Installed           : 2012-01-24T12:00Z
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Additional Information   : 

4.x  Antenna Type             : (A20, from rcvr_ant.tab; see instructions)
     Serial Number            : (A*, but note the first A5 is used in SINEX)
     Antenna Reference Point  : (BPA/BCR/XXX from "antenna.gra"; see instr.)
     Marker->ARP Up Ecc. (m)  : (F8.4)
     Marker->ARP North Ecc(m) : (F8.4)
     Marker->ARP East Ecc(m)  : (F8.4)
     Alignment from True N    : (deg; + is clockwise/east)
     Antenna Radome Type      : (A4 from rcvr_ant.tab; see instructions)
     Radome Serial Number     : 
     Antenna Cable Type       : (vendor & type number)
     Antenna Cable Length     : (m)
     Date Installed           : (CCYY-MM-DDThh:mmZ)
     Date Removed             : (CCYY-MM-DDThh:mmZ)
     Additional Information   : (multiple lines)
    """,
    "utf-8",
)
