
if [ "$1" = "" ]; then
# if [ "$1" -eq "" ]; then
	problem=1104_1_1_4608
else
	problem=$1
fi

if [ "$2" = "" ]; then
# if [ "$1" -eq "" ]; then
	first=1
else
	first=$2
fi

if [ "$3" = "" ]; then
# if [ "$1" -eq "" ]; then
	justPrechoose=0
else
	justPrechoose=$3
fi

if [ "$justPrechoose" -eq "1" ]; then
# if [ "$1" -eq "" ]; then
	python3 genyaml_Round1.py --prob=$problem --first=$first --justPrechoose=$justPrechoose
else
	python3 genyaml_Round1.py --prob=$problem --first=$first --justPrechoose=$justPrechoose
	./run_start_Round1.sh $problem $first
	python3 Round1Merge.py --prob=$problem --first=$first
fi
# python3 genyaml_Round1.py --prob=$problem --first=$first --justPrechoose=$justPrechoose
# ./run_start_Round1.sh $problem $first
# python3 Round1Merge.py --prob=$problem --first=$first