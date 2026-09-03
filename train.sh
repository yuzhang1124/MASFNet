CUDA_VISIBLE_DEVICES="0" \
python train.py \
--hiera_path "/home/" \
--train_image_path "/home/" \
--train_mask_path "/home/" \
--save_path "/home/" \
--epoch 50 \
--lr 0.0001 \
--batch_size 6