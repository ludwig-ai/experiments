cd mushroom_edibility
python3 run_auto_train_1hr_nodeless.py >run.1hr 2>&1 &
cd ..
#
cd forest_cover
python3 run_auto_train_1hr_nodeless.py >run.1hr 2>&1 &
cd ..
#
cd higgs
python3 run_auto_train_1hr_nodeless.py >run.1hr 2>&1 &
cd ..
#
wait
#
cd mushroom_edibility
python3 run_auto_train_2hr_nodeless.py >run.2hr 2>&1 &
cd ..
#
cd forest_cover
python3 run_auto_train_2hr_nodeless.py >run.2hr 2>&1 &
cd ..
#
cd higgs
python3 run_auto_train_2hr_nodeless.py >run.2hr 2>&1 &
cd ..
#
wait
#
cd mushroom_edibility
python3 run_auto_train_4hr_nodeless.py >run.4hr 2>&1 &
cd ..
#
cd forest_cover
python3 run_auto_train_4hr_nodeless.py >run.4hr 2>&1 &
cd ..
#
cd higgs
python3 run_auto_train_4hr_nodeless.py >run.4hr 2>&1 &
cd ..
#
wait
#
