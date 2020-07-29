#!/bin/bash

wav="/home/Data/blizzard2013_for_test/new_download/BC2013_segmented_v0_wav2/a_room_with_a_view/CB-RV-10-114.wav"
function get_sr() {
    sr_cmd="sox --i $1 | grep 'Sample Rate' | awk -F ':' '{print \$2}' | tr -d '[:space:]'"
    sr=$(eval $sr_cmd)
    echo $sr
}
sr=$(get_sr $wav)
sr_set=22050

# Test1
if [[ $sr == $sr_set ]]; then
    echo "it is " $sr_set
fi

# Test2 Checking all wav
#db="/home/Data/blizzard2013/new_download"
db="/home/Data/blizzard2013_for_test/new_download"
fict_prev=""
fiction=""
counter=0
if false; then
    find ${db} -name "*.wav" | sort |  while read -r wav_f; do
        sr_o=$(get_sr $wav_f)
        if [[ $sr_o != $sr_set ]]; then
            fiction=$([[ $wav_f =~ .*/(.*)/.*wav ]] && echo ${BASH_REMATCH[1]})
            id=$(basename $wav_f | sed -e "s/\.[^\.]*$//g")
            counter=$((counter+1))
            #echo $fiction $id
        fi
        if [[ $fict_prev != $fiction ]]; then
            if [[ $fict_prev != "" ]]; then
                echo $fict_prev $sr_o $counter
            fi
            fict_prev=$fiction
            counter=0
        fi
    done
fi
#echo "converting sr is finished"

#TEST 3 - Only checking one sample in each fiction
if false; then
    last_fict=""
    find ${db} -name "*.wav" | sort |  while read -r wav_f; do
        temp_fict=$([[ $wav_f =~ .*/(.*)/.*wav ]] && echo ${BASH_REMATCH[1]})
        if [[ $last_fict != $temp_fict ]]; then
            sr_o=$(get_sr $wav_f)
            fiction=$([[ $wav_f =~ .*/(.*)/.*wav ]] && echo ${BASH_REMATCH[1]})
            echo "The sr of " $fiction " is " $sr_o
            # Convert sr to sr_set
            if [[ $sr_o != $sr_set ]]; then
                ../down_sample.py $wav_f $wav_f_new $sr_set
            fi
        fi
        last_fict=$temp_fict
    done
fi


#TEST 3 - Converting
sou_dir="/home/Data/blizzard2013/new_download/BC2013_segmented_v0_wav1/jane_eyre"
des_dir="/home/Data/blizzard2013/new_download_bk/BC2013_segmented_v0_wav1/jane_eyre"
[ ! -e ${des_dir} ] && mkdir -p ${des_dir}

if false; then
    find ${sou_dir} -name "*.wav" | sort |  while read -r wav_f; do
        des_f=$des_dir/$(basename $wav_f)
        echo "res_wav:" $wav_f
        echo "des_wav:" $des_f
        sox $wav_f -r $sr_set $des_f
    done
fi
#echo "converting sr is finished"