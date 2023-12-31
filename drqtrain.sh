# -u for unbuffered stdout
# add PWD and PWD/EfficientZero so imports works
# redicret stdout and stderr to file (rewrite mode)
OUTFILE="drqoutput$HOSTNAME.txt"
PYTHONPATH=$PYTHONPATH:${PWD}:"${PWD}/drqv2" nohup python -u rlcam_drqv2_mql.py &> $OUTFILE
