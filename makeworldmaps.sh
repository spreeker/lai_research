set -x

#cd lai_v006_world/2002/
#bash download.sh
#cd ..
#cd ..

# python worldmap.py --nasa

python worldmap.py --multi_regression --greenrange 0 9
python worldmap.py --multi_regression --greenrange 0 2
python worldmap.py --multi_regression --greenrange 1 3
python worldmap.py --multi_regression --greenrange 2 4
python worldmap.py --multi_regression --greenrange 3 5
python worldmap.py --multi_regression --greenrange 4 6
python worldmap.py --multi_regression --greenrange 5 7
python worldmap.py --multi_regression --greenrange 6 8
python worldmap.py --multi_regression --greenrange 7 9
