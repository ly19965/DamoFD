#!/bin/bash
#cd "$(dirname "$0")"
#set -e

#cd ../

save_dir=./save_dir/DamoFD_10g
mkdir -p ${save_dir}


resolution=224
budget_flops=1414e6 # 8659/ ( 640 * 480 / (224 * 224))
max_layers=20
population_size=512
evolution_max_iter=96000

echo "SuperConvK3BNRELU(3,24,2,1)SuperResK3K3(24,48,2,64,5)SuperResK3K3(48,64,2,64,5)SuperResK3K3(64,96,2,128,3)SuperResK3K3(96,128,2,16,1)" > ${save_dir}/init_plainnet.txt

# --search_space SearchSpace/search_space_XXBL.py \
#  --search_space SearchSpace/search_space_IDW_fixfc.py \
python evolution_search.py --gpu 0 \
  --zero_shot_score DDSAR \
  --budget_flops ${budget_flops} \
  --max_layers ${max_layers} \
  --search_space SearchSpace/search_space_XXBL.py \
  --batch_size 64 \
  --input_image_size ${resolution} \
  --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
  --num_classes 1000 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir}
