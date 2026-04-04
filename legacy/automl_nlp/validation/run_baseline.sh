cd amazon_review_polarity
python3 run_auto_train_standard_baseline.title.py >run.baseline 2>&1
cd ..
#
cd reuters_r8
python3 run_auto_train_standard_baseline.py >run.baseline 2>&1
cd ..
#
cd ohsumed_7400
python3 run_auto_train_standard_baseline.py >run.baseline 2>&1
cd ..
#
