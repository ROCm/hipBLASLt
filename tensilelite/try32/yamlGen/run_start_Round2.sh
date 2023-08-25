cd $2

echo $1

mkdir tensilelite/try32/yamlGen/$1/tunning_BFVF_Round2/

for i in {0..0}
do
    read -r ID <<< "${i}"

    filename='tensilelite/try32/yamlGen/'$1'/tunning_BFVF_Round2/tunning_BFVF_Round2_'${ID}
    rm -r $filename
    echo "$filename is removed"

    # Check the file is exists or not
    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round/1_BenchmarkProblems/*'
    # rm -r $filename
    # echo "$filename is removed"

    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round/2_BenchmarkData/*'
    rm -r $filename
    echo "$filename is removed"

    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round/3_LibraryLogic/*'
    rm -r $filename
    echo "$filename is removed"

    #mkdir tensilelite/try32/16x16/tunning_BFVF_${ID}; cp -r tensilelite/try32/tunning_BFVF_${ID}/3_LibraryLogic/ tensilelite/try32/16x16/tunning_BFVF_${ID}/; cp -r tensilelite/try32/tunning_BFVF_${ID}/2_BenchmarkData/ tensilelite/try32/16x16/tunning_BFVF_${ID}/
    # rm -rf tensilelite/try32/tunning_BFVF_${ID}/1_BenchmarkProblems/
# stdout=$(
    tensilelite/Tensile/bin/Tensile \
        tensilelite/try32/yamlGen/$1/FP16_NN_MI250X_Round2_${ID}.yaml \
        tensilelite/try32/yamlGen/tunning_BFVF_Round \
        # | tail -1
    # )
    # echo ${stdout}
    mkdir tensilelite/try32/yamlGen/$1/tunning_BFVF_Round2/tunning_BFVF_Round2_${ID}
    # mv tensilelite/try32/yamlGen/1104_1_1_4608/tunning_BFVF_Round2/1_BenchmarkProblems tensilelite/try32/yamlGen/1104_1_1_4608/tunning_BFVF_Round2/tunning_BFVF_Round2_${ID}/

    # filename='tensilelite/try32/yamlGen/1104_1_1_4608/tunning_BFVF_Round2/2_BenchmarkData/Cijk_Ailk_Bljk_HHS_BH_00_CSVWinner.yaml'
    # rm $filename
    # echo "$filename is removed"

    mv tensilelite/try32/yamlGen/tunning_BFVF_Round/2_BenchmarkData tensilelite/try32/yamlGen/$1/tunning_BFVF_Round2/tunning_BFVF_Round2_${ID}/
    mv tensilelite/try32/yamlGen/tunning_BFVF_Round/3_LibraryLogic tensilelite/try32/yamlGen/$1/tunning_BFVF_Round2/tunning_BFVF_Round2_${ID}/

    filename='tensilelite/try32/yamlGen/tunning_BFVF_Round/1_BenchmarkProblems/*'
    rm -r $filename
    echo "$filename is removed"
done
