
if [ "$1" = "" ]; then
# if [ "$1" -eq "" ]; then
    problem=1104_1_1_4608
else
    problem=$1
fi

python3 genyaml_Round3.py --prob=$problem
./run_start_Round3.sh $problem
python3 Round3Merge.py --prob=$problem