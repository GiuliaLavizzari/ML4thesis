#!/bin/bash

stringOP="cHbox cHDD cHl1 cHl3 cHq1 cHq3 cHWB cHW cll1 cll cqq11 cqq1 cqq31 cqq3"

python lossperbatch.py 1 7
echo "done with SM"

for oper in $stringOP; do
    python lossperbatchBSM.py 1 7 $oper
    python finalBSM1.py 1 7 $oper > output${oper}_m1_d7.txt
    echo "done with $oper"
done  
    


