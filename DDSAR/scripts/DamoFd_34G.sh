#!/bin/bash
#cd "$(dirname "$0")"
#set -e

#cd ../

save_dir=./save_dir/DamoFD_34g
mkdir -p ${save_dir}


resolution=224
budget_flops=4375e6 # 26787 / ( 640 * 480 / (224 * 224))
max_layers=20
population_size=512
evolution_max_iter=96000

echo "SuperConvK3BNRELU(3,56,2,2)SuperResK3K3(56,112,2,96,5)SuperResK3K3(112,224,2,96,5)SuperResK3K3(224,448,2,128,3)SuperResK3K3(448,1024,2,16,1)" > ${save_dir}/init_plainnet.txt

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
