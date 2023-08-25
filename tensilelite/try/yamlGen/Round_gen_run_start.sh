
echo "The script you are running has:"
echo "basename: [$(basename "$0")]"
echo "dirname : [$(dirname "$0")]"
echo "pwd     : [$(pwd)]"

echo $(pwd)/../../../

name1=$(pwd)
echo $name1
homepath=${name1%"tensilelite/try/yamlGen"}
echo $homepath

problemMI250=(
    "1104_1_1_4608"
    "1104_16_1_4608"
    "1104_1335_1_4608"
    "1104_1408_1_4608"
    "4608_1_1_4608"
    "4608_16_1_4608"
    "4608_1335_1_4608"
    "4608_1408_1_4608"
    "16_1_1_4608"
    "16_16_1_4608"
    "16_1335_1_4608"
    "16_1408_1_4608"
    "768_1_1_4608"
    "768_16_1_4608"
    "768_1335_1_4608"
    "768_1408_1_4608"
    "4608_1_1_320"
    "4608_16_1_320"
    "4608_1335_1_320"
    "4608_1408_1_320"
)

problem=(
    "16_16_1_1024"
    "16_16_1_8192"
    "16_16_1_65536"
    "16_2048_1_1024"
    "16_2048_1_8192"
    "16_2048_1_65536"
    "16_8192_1_1024"
    "16_8192_1_8192"
    "16_8192_1_65536"

    "2048_16_1_1024"
    "2048_16_1_8192"
    "2048_16_1_65536"
    "2048_2048_1_1024"
    "2048_2048_1_8192"
    "2048_2048_1_65536"
    "2048_8192_1_1024"
    "2048_8192_1_8192"
    "2048_8192_1_65536"

    "8192_16_1_1024"
    "8192_16_1_8192"
    "8192_16_1_65536"
    "8192_2048_1_1024"
    "8192_2048_1_8192"
    "8192_2048_1_65536"
    "8192_8192_1_1024"
    "8192_8192_1_8192"
    "8192_8192_1_65536"
)

first=50 # must more than two
justPrechoose=0

MI16=1
MI32=0

for prob in ${problem[@]}
do
	echo $prob
	./Round1_gen_run_start.sh $prob $first $justPrechoose $homepath $MI16 $MI32
	if [ "$justPrechoose" = "0" ]; then
		./Round2_gen_run_start.sh $prob $homepath
		./Round3_gen_run_start.sh $prob $homepath
	fi
done

if [ "$justPrechoose" = "0" ]; then

    for N in "${problem[@]}"
    do
        FILE="$homepath"tensilelite/try/aldebaran_Cijk_Ailk_Bljk_HHS_BH_Bias_AH_SDV_SAV.yaml
    if test -f "$FILE"; then
        echo "$FILE exists."
        python3 "$homepath"tensilelite/Tensile/Utilities/archive/merge_rocblas_yaml_files.py "$homepath"tensilelite/try/ ./$N/tunning_BFVF_Round3/tunning_BFVF_Round3_0/3_LibraryLogic/ "$homepath"tensilelite/try/
    else
        echo "$FILE not exists."
        cp ./$N/tunning_BFVF_Round3/tunning_BFVF_Round3_0/3_LibraryLogic/aldebaran_Cijk_Ailk_Bljk_HHS_BH_Bias_AH_SDV_SAV.yaml "$homepath"tensilelite/try/
    fi
    done

fi
