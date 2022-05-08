cd Real-ESRGAN
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus_artwork_working --input ../results/fastcut --output ../results/fastcut_finetuned  --tile 512
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus_artwork --input ../results/cut --output ../results/cut_finetuned  --tile 512
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus_artwork --input ../results/cut_blend --output ../results/cut_blend_finetuned  --tile 512
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus_artwork --input ../results/fastcut_blend --output ../results/fastcut_blend_finetuned  --tile 512

python3 inference_realesrgan.py --model_name RealESRGAN_x4plus --input ../results/cut --output ../results/cut_pretrained  --tile 512
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus --input ../results/fastcut --output ../results/fastcut_pretrained  --tile 512
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus --input ../results/cut_diff --output ../results/cut_diff_pretrained  --tile 512
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus --input ../results/fastcut_diff --output ../results/fastcut_diff_pretrained  --tile 512

python3 inference_realesrgan.py --model_name RealESRGAN_x4plus --input ../results/cut_2048 --output ../results/cut_2048_pretrained  --tile 512
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus --input ../results/fastcut_2048 --output ../results/fastcut_2048_pretrained  --tile 512
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus_artwork --input ../results/cut_2048 --output ../results/cut_2048_finetuned  --tile 512
python3 inference_realesrgan.py --model_name RealESRGAN_x4plus_artwork --input ../results/fastcut_2048 --output ../results/fastcut_2048_finetuned  --tile 512


