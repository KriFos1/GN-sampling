--------------------------------------------------------------------------------------
RUNSPEC
--------------------------------------------------------------------------------------
DIMENS
   50   50    1  /

METRIC

OIL

WATER

GAS

DISGAS


TABDIMS
    1       1      20      200       1      200       1    /

WELLDIMS
    5     1        5       5  /


START
  1 'JAN' 2000 /


NSTACK
 10 /



--------------------------------------------------------------------------------------
GRID
--------------------------------------------------------------------------------------
INCLUDE
 '../TRUE_MODEL/include/TRUE_MODEL.COORD' /

INCLUDE
 '../TRUE_MODEL/include/TRUE_MODEL.ZCORN' /

PORO
2500*0.2
/

<%
import numpy as np
%>
PERMX
% for p in log_permx:
% if p > 10:
${"%.9f" %(np.exp(10))}
% elif p < -5:
${"%.9f" %(np.exp(-5))}
% else:
${"%.9f" %(np.exp(p))}
% endif
% endfor
/

COPY
 PERMX PERMY /
 PERMX PERMZ /
/

MULTIPLY
 PERMZ 0.001 /
/

INIT

NEWTRAN


--------------------------------------------------------------------------------------
PROPS
--------------------------------------------------------------------------------------

ROCK
  300 1.450E-05 /

INCLUDE
 '../TRUE_MODEL/include/TRUE_MODEL.PVO' /  

INCLUDE
 '../TRUE_MODEL/include/TRUE_MODEL.RCP' /


--------------------------------------------------------------------------------------
SOLUTION
--------------------------------------------------------------------------------------
EQUIL
2000.000 200.00 2280.00  .000  2000.000  .000     1      0       0 /

PBVD
 1    10 
 1000 10 /


--------------------------------------------------------------------------------------
SUMMARY
--------------------------------------------------------------------------------------

FOPR
FWPR
FLPR
FLPT
FOPT
FGPT
FWPT
FPR
FWIT

WOPR
 'P1'
 'P2'
 'P3'
 'P4'
/

WWPR
 'P1'
 'P2'
 'P3'
 'P4'
/

WGPR
 'P1'
 'P2'
 'P3'
 'P4'
/

WWIR
 'INJ1'
/

WBHP
 'P1'
 'P2'
 'P3'
 'P4'
/


FVIR
FVPR

RPTONLY

RUNSUM

SEPARATE

RPTSMRY
 1  /

DATE


TCPU

SCHEDULE
----------------------------------------------------------------------------------------
RPTRST
   BASIC=2/

WELSPECS
 'INJ1' G1 25 25 2000 WATER /
 'P1'   G1  1  1 1* 'OIL'   /
 'P2'   G1  1 50 1* 'OIL'   /
 'P3'   G1 50  1 1* 'OIL'   /
 'P4'   G1 50 50 1* 'OIL'   /
/

COMPDAT
 'INJ1' 25 25 1 1 OPEN 2* 0.25 /
 'P1' 1 1  1 1 OPEN 2* 0.25 /
 'P2' 1 50 1 1 OPEN 2* 0.25 /
 'P3' 50 1 1 1 OPEN 2* 0.25 /
 'P4' 50 50 1 1 OPEN 2* 0.25 /
/

INCLUDE
  ../TRUE_MODEL/include/TRUE_MODEL.SCHEDULE /

END