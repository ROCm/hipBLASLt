cd /operator/hipBLASLt

for i in {0..22}
do
    read -r ID <<< "${i}"

    filename='tensilelite/try/yamlGen/tunning_BFVF_Round3/tunning_BFVF_Round3_'${ID}
    rm -r $filename
    echo "$filename is removed"

    # Check the file is exists or not
    filename='tensilelite/try/yamlGen/tunning_BFVF_Round3/1_BenchmarkProblems/*'
    rm -r $filename
    echo "$filename is removed"

    filename='tensilelite/try/yamlGen/tunning_BFVF_Round3/2_BenchmarkData/*'
    rm -r $filename
    echo "$filename is removed"

    filename='tensilelite/try/yamlGen/tunning_BFVF_Round3/3_LibraryLogic/*'
    rm -r $filename
    echo "$filename is removed"

    #mkdir tensilelite/try32/16x16/tunning_BFVF_${ID}; cp -r tensilelite/try32/tunning_BFVF_${ID}/3_LibraryLogic/ tensilelite/try32/16x16/tunning_BFVF_${ID}/; cp -r tensilelite/try32/tunning_BFVF_${ID}/2_BenchmarkData/ tensilelite/try32/16x16/tunning_BFVF_${ID}/
    # rm -rf tensilelite/try32/tunning_BFVF_${ID}/1_BenchmarkProblems/
# stdout=$(
    tensilelite/Tensile/bin/Tensile \
        tensilelite/try/yamlGen/FP16_NN_MI250X_testBFVF_Round3_${ID}.yaml \
        tensilelite/try/yamlGen/tunning_BFVF_Round3 \
        # | tail -1
    # )
    # echo ${stdout}
    mkdir tensilelite/try/yamlGen/tunning_BFVF_Round3/tunning_BFVF_Round3_${ID}
    # mv tensilelite/try/yamlGen/tunning_BFVF_Round3/1_BenchmarkProblems tensilelite/try/yamlGen/tunning_BFVF_Round3/tunning_BFVF_Round3_${ID}/

    filename='tensilelite/try/yamlGen/tunning_BFVF_Round3/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.yaml'
    rm $filename
    echo "$filename is removed"

    mv tensilelite/try/yamlGen/tunning_BFVF_Round3/2_BenchmarkData tensilelite/try/yamlGen/tunning_BFVF_Round3/tunning_BFVF_Round3_${ID}/
    mv tensilelite/try/yamlGen/tunning_BFVF_Round3/3_LibraryLogic tensilelite/try/yamlGen/tunning_BFVF_Round3/tunning_BFVF_Round3_${ID}/

    filename='tensilelite/try/yamlGen/tunning_BFVF_Round3/1_BenchmarkProblems/*'
    rm -r $filename
    echo "$filename is removed"
done
