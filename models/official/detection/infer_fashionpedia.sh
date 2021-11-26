#!/bin/sh

export PYTHONPATH="$PYTHONPATH:/home/gitlocal/tpu/models/official/efficientnet/"
export PYTHONPATH="$PYTHONPATH:/home/gitlocal/tpu/models/"
export CUDA_VISIBLE_DEVICES=0

python inference_fashionpedia.py \
    --image_size=1280 \
    --checkpoint_path=projects/fashionpedia/spinenet/fashionpedia-spinenet-143/model.ckpt \
    --label_map_file=projects/fashionpedia/dataset/fashionpedia_label_map.csv \
    --image_file_pattern=/home/data/*.jpg \
    --output_html=projects/fashionpedia/output/test.html \
    --config_file=projects/fashionpedia/spinenet/spinenet143_amrcnn.yaml \
    --output_file=projects/fashionpedia/output/test.npy \
    --max_boxes_to_draw 20 \
    --min_score_threshold 0.6
