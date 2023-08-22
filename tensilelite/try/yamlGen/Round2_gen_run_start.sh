
if [ "$1" = "" ]; then
# if [ "$1" -eq "" ]; then
	problem=1104_1_1_4608
else
	problem=$1
fi

python3 genyaml_Round2.py --prob=$problem
./run_start_Round2.sh $problem
python3 Round2Merge.py --prob=$problem