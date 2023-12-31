# -u for unbuffered stdout
# add PWD and PWD/EfficientZero so imports works
# redicret stdout and stderr to file (rewrite mode)
OUTFILE="curloutput$HOSTNAME.txt"
PYTHONPATH=$PYTHONPATH:${PWD}:"${PWD}/curl" nohup python -u curl_train.py &> $OUTFILE
