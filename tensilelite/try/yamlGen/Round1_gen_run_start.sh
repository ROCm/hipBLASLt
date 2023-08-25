
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

if [ "$4" = "" ]; then
# if [ "$1" -eq "" ]; then
	homepath=0
else
	homepath=$4
fi

if [ "$5" = "" ]; then
# if [ "$1" -eq "" ]; then
	MI16=0
else
	MI16=$5
fi

if [ "$6" = "" ]; then
# if [ "$1" -eq "" ]; then
	MI32=0
else
	MI32=$6
fi

echo MI32
echo $MI32
echo MI16
echo $MI16

echo Victorhomepath
echo $homepath

if [ "$justPrechoose" -eq "1" ]; then
	echo "justPrechoose is 1"
	echo $justPrechoose
# if [ "$1" -eq "" ]; then
	python3 genyaml_Round1.py --prob=$problem --first=$first --justPrechoose=$justPrechoose --homepath=$homepath --MI16=$MI16 --MI32=$MI32
else
	echo "justPrechoose is 0"
	echo $justPrechoose
	python3 genyaml_Round1.py --prob=$problem --first=$first --justPrechoose=$justPrechoose --homepath=$homepath --MI16=$MI16 --MI32=$MI32
	./run_start_Round1.sh $problem $first $homepath
	python3 Round1Merge.py --prob=$problem --first=$first --homepath=$homepath
fi
# python3 genyaml_Round1.py --prob=$problem --first=$first --justPrechoose=$justPrechoose
# ./run_start_Round1.sh $problem $first
# python3 Round1Merge.py --prob=$problem --first=$first