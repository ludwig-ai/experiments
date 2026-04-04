cd amazon_review_polarity
python3 run_auto_train_standard_nodeless.title.py >run.nodeless 2>&1 &
cd ..
#
cd reuters_r8
python3 run_auto_train_standard_nodeless.py >run.nodeless 2>&1 &
cd ..
#
cd ohsumed_7400
python3 run_auto_train_standard_nodeless.py >run.nodeless 2>&1 &
cd ..
#
wait
