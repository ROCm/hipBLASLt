cd /hipBLASLt/

echo $1

mkdir tensilelite/try/yamlGen/$1/tunning_BFVF_Round1/
# filename='tensilelite/try/yamlGen/1104_1_1_4608/tunning_BFVF_Round1/tunning_BFVF_Round1*'
# rm -r $filename
# echo "$filename is removed"

# for i in {1..2}
# for i in {3..5}
# for i in {6..7}
# for i in {8..9}
# for i in {10..19}
echo FIRST
echo $2

# for i in {1..$($2/2)}
for ((i=1;i<=$2;i=i+1))
do
    read -r ID <<< "${i}"

    filename='tensilelite/try/yamlGen/'$1'/tunning_BFVF_Round1/tunning_BFVF_Round1_'${ID}
    rm -r $filename
    echo "$filename is removed"

    # Check the file is exists or not
    filename='tensilelite/try/yamlGen/tunning_BFVF_Round/1_BenchmarkProblems/*'
    # rm -r $filename
    # echo "$filename is removed"

    filename='tensilelite/try/yamlGen/tunning_BFVF_Round/2_BenchmarkData/*'
    rm -r $filename
    echo "$filename is removed"

    filename='tensilelite/try/yamlGen/tunning_BFVF_Round/3_LibraryLogic/*'
    rm -r $filename
    echo "$filename is removed"

# stdout=$(
        tensilelite/Tensile/bin/Tensile \
            tensilelite/try/yamlGen/$1/FP16_NN_MI250X_${ID}.yaml \
            tensilelite/try/yamlGen/tunning_BFVF_Round \
        # | tail -1
    # )
    # echo ${stdout}

    mkdir tensilelite/try/yamlGen/$1/tunning_BFVF_Round1/tunning_BFVF_Round1_${ID}
    # mv tensilelite/try/yamlGen/1104_1_1_4608/tunning_BFVF_Round1/1_BenchmarkProblems tensilelite/try/yamlGen/1104_1_1_4608/tunning_BFVF_Round1/tunning_BFVF_Round1_${ID}/

    # filename='tensilelite/try/yamlGen/1104_1_1_4608/tunning_BFVF_Round1/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.yaml'
    # rm $filename
    # echo "$filename is removed"

    mv tensilelite/try/yamlGen/tunning_BFVF_Round/2_BenchmarkData tensilelite/try/yamlGen/$1/tunning_BFVF_Round1/tunning_BFVF_Round1_${ID}/
    mv tensilelite/try/yamlGen/tunning_BFVF_Round/3_LibraryLogic tensilelite/try/yamlGen/$1/tunning_BFVF_Round1/tunning_BFVF_Round1_${ID}/

    filename='tensilelite/try/yamlGen/tunning_BFVF_Round/1_BenchmarkProblems/*'
    rm -r $filename
    echo "$filename is removed"
done
