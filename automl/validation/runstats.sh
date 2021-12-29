cd ames_housing
echo "1hr ames_housing**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.1hr root_mean_squared_error
echo "2hr ames_housing**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.2hr root_mean_squared_error
echo "4hr ames_housing**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.4hr root_mean_squared_error
cd ..
#
cd forest_cover
echo "1hr forest_cover**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.1hr accuracy
echo "2hr forest_cover**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.2hr accuracy
echo "4hr forest_cover**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.4hr accuracy
cd ..
#
cd higgs
echo "1hr higgs**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.1hr accuracy
echo "2hr higgs**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.2hr accuracy
echo "4hr higgs**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.4hr accuracy
cd ..
#
cd mushroom_edibility
echo "1hr mushroom_edibility**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.1hr accuracy
echo "2hr mushroom_edibility**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.2hr accuracy
echo "4hr mushroom_edibility**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.4hr accuracy
cd ..
#
cd synthetic_fraud
echo "1hr synthetic_fraud**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.1hr accuracy
echo "2hr synthetic_fraud**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.2hr accuracy
echo "4hr synthetic_fraud**************"
python ../../../utils/best_hyperopt_statistics.py hyperopt_statistics.json.4hr accuracy
cd ..
#
