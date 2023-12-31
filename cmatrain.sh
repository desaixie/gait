# -u for unbuffered stdout
# add PWD and PWD/EfficientZero so imports works
# redicret stdout and stderr to file (rewrite mode)
OUTFILE="cmaoutput$HOSTNAME.txt"
nohup python -u CMA_ES.py &> $OUTFILE
