
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

first=100
justPrechoose=0

for prob in ${problem[@]}
do
	echo $prob
	./Round1_gen_run_start.sh $prob $first $justPrechoose
	./Round2_gen_run_start.sh $prob $first
	./Round3_gen_run_start.sh $prob $first
done

for N in "${problem[@]}"
do
FILE=/hipBLASLt/tensilelite/try/aldebaran_Cijk_Ailk_Bljk_HHS_BH.yaml
if test -f "$FILE"; then
    echo "$FILE exists."
    python3 /hipBLASLt/tensilelite/Tensile/Utilities/archive/merge_rocblas_yaml_files.py /hipBLASLt/tensilelite/try/ ./$N/tunning_BFVF_Round3/tunning_BFVF_Round3_0/3_LibraryLogic/ /hipBLASLt/tensilelite/try/
else
	echo "$FILE not exists."
	cp ./$N/tunning_BFVF_Round3/tunning_BFVF_Round3_0/3_LibraryLogic/aldebaran_Cijk_Ailk_Bljk_HHS_BH.yaml /hipBLASLt/tensilelite/try/
fi
done