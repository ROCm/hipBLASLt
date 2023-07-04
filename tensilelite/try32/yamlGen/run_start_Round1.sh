cd /operator/hipBLASLt

# filename='tensilelite/try32/yamlGen/tunning_BFVF_Round1/tunning_BFVF_Round1*'
# rm -r $filename
# echo "$filename is removed"

# for i in {1..2}
# for i in {3..5}
# for i in {6..7}
# for i in {8..9}
# for i in {10..19}
for i in {1..23}
do
    read -r ID <<< "${i}"

    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round1/tunning_BFVF_Round1_'${ID}
    rm -r $filename
    echo "$filename is removed"

    # Check the file is exists or not
    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round1/1_BenchmarkProblems/*'
    rm -r $filename
    echo "$filename is removed"

    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round1/2_BenchmarkData/*'
    rm -r $filename
    echo "$filename is removed"

    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round1/3_LibraryLogic/*'
    rm -r $filename
    echo "$filename is removed"

# stdout=$(
        tensilelite/Tensile/bin/Tensile \
            tensilelite/try32/yamlGen/FP16_NN_MI250X_testBFVF_${ID}.yaml \
            tensilelite/try32/yamlGen/tunning_BFVF_Round1 \
        # | tail -1
    # )
    # echo ${stdout}

    mkdir tensilelite/try32/yamlGen/tunning_BFVF_Round1/tunning_BFVF_Round1_${ID}
    # mv tensilelite/try32/yamlGen/tunning_BFVF_Round1/1_BenchmarkProblems tensilelite/try32/yamlGen/tunning_BFVF_Round1/tunning_BFVF_Round1_${ID}/

    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round1/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.yaml'
    rm $filename
    echo "$filename is removed"

    mv tensilelite/try32/yamlGen/tunning_BFVF_Round1/2_BenchmarkData tensilelite/try32/yamlGen/tunning_BFVF_Round1/tunning_BFVF_Round1_${ID}/
    mv tensilelite/try32/yamlGen/tunning_BFVF_Round1/3_LibraryLogic tensilelite/try32/yamlGen/tunning_BFVF_Round1/tunning_BFVF_Round1_${ID}/

    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round1/1_BenchmarkProblems/*'
    rm -r $filename
    echo "$filename is removed"
done
