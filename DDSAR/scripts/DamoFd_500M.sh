#!/bin/bash
#cd "$(dirname "$0")"
#set -e

#cd ../

save_dir=./save_dir/DamoFD_500m
mkdir -p ${save_dir}


resolution=224
budget_flops=65e6 # 403 / ( 640 * 480 / (224 * 224))
max_layers=14
population_size=512
evolution_max_iter=96000

echo "SuperConvK3BNRELU(3,32,2,1)SuperResIDWE1K3(8,32,2,8,1)SuperResIDWE4K3(32,48,2,32,1)SuperResIDWE1K3(48,96,2,48,1)SuperResIDWE1K3(96,128,2,96,1)" > ${save_dir}/init_plainnet.txt

python evolution_search.py --gpu 0 \
  --zero_shot_score DDSAR \
  --search_space SearchSpace/search_space_IDW_fixfc.py \
  --budget_flops ${budget_flops} \
  --max_layers ${max_layers} \
  --batch_size 64 \
  --input_image_size ${resolution} \
  --plainnet_struct_txt ${save_dir}/init_plainnet.txt \
  --num_classes 1000 \
  --evolution_max_iter ${evolution_max_iter} \
  --population_size ${population_size} \
  --save_dir ${save_dir}
