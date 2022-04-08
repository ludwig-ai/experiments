cd amazon_review_polarity
python3 run_auto_train_standard.py >run.standard 2>&1
cd ..
#
cd amazon_reviews
python3 run_auto_train_standard.py >run.standard 2>&1
cd ..
#
cd bbcnews
python3 run_auto_train_standard.py >run.standard 2>&1
cd ..
#
cd imdb
python3 run_auto_train_standard.py >run.standard 2>&1
cd ..
#
cd ohsumed_7400
python3 run_auto_train_standard.py >run.standard 2>&1
cd ..
#
cd reuters_r8
python3 run_auto_train_standard.py >run.standard 2>&1
cd ..
#
