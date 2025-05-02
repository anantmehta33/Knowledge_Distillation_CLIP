## Uncertainty-Aware Knowledge Distillation

In this project, we show how to train a CLIP model by using Global Contrastive Loss (GCL) on a 1M subset of the image-text dataset [DFN-2B](https://huggingface.co/datasets/apf1/datafilteringnetworks_2b), using confidence aware knowledge distillation.

We propose **three types of Uncertainty-Aware Distillation Losses**, modulated by a dynamic confidence score.

Let:

- `αᵢ`: Confidence score for each sample *i*, derived from the teacher model (range: 0 to 1)
- `λ`: Tunable scaling factor for each loss component
- `L_dist⁽ⁱ⁾`: Distillation loss for sample *i* — either an individual loss or a combination of multiple components

where, `L_<sub>dist</sub>⁽ⁱ⁾` = (αᵢ + 1) · λ₁· L_FD⁽ⁱ⁾
           + (αᵢ + 1) · λ₂ · L_KL⁽ⁱ⁾
           + (αᵢ + 1) · λ₃ · L_ICL⁽ⁱ⁾

![image](https://github.com/user-attachments/assets/7b066619-acb6-4faf-8a0c-cc7f350d6d73)

### Environment

Setting up a new virtual environment with Conda:
````bash
env_name='clipkd'
conda create -n "$env_name" python=3.11
conda activate "$env_name"
pip install -r requirements-training.txt
pip install -r requirements-eval.txt
````

### Training

**Data**:
```bash
mkdir datasets
# dfn_data, ~40GB
gdown --folder --output ./datasets/ 1SEhMped23ACVRzNIdgo4aI81rONqnbzi
cd ./datasets/dfn_data
for i in {0..6}; do tar xf part0${i}.tar; rm part0${i}.tar; done
cd -
```

The following command trains a ViT-B/16 CLIP model using FastCLIP on DFN on 2 GPUs, with (per-GPU) batch size 320 for 30 epochs:
```bash
source ~/.bashrc
conda activate clipkd

export PYTHONPATH="$PYTHONPATH:$PWD/src"
export HUGGINGFACE_HUB_CACHE='./checkpoints/huggingface'

loss_type=KD_MSE_KL_ICL
dist_coeff=var_shifted

torchrun \
    --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --rdzv-id=4204 --rdzv-backend=c10d --rdzv-endpoint='127.0.0.1' \
    src/training/main.py \
    --save-frequency 1 \
    --train-data './datasets/dfn_data/00000{000..139}.tar' \
    --train-num-samples 1000000 --data_size 1400000 \
    --warmup 500 \
    --batch-size 320 \
    --epochs 30 \
    --workers 6 \
    --model ViT-B-16 \
    --name ${loss_type}_${dist_coeff}_fastclip \
    --seed 2025 \
    --wd 0.2 \
    --local-loss \
    --fastclip --multiply_tau --temperature_scheme global_learnable \
    --lr 3.125e-4 --lr_tau 7.8125e-5 --lr_tau_scheduler step_thresh --rho 11.0 \
    --gamma 0.9 --gamma_schedule cosine --gamma_decay_epochs 10 \
    --distill-model ViT-B-32 \
    --distill-pretrained '/scratch/group/optmai/anant/Ref_modelling/fast_clip/src/clip_vit_b32_openai.pth' \
    --loss_type ${loss_type} \
    --dist_coeff ${dist_coeff} \
    --get_confidences yes \
    --gather-with-grad

```

In src/training/main.py, we create the model, optimizer, loss, dataloader, etc. And in src/training/train.py, we do the training step by step.

To leverage the reference model ViT-B/32 CLIP from OpenAI, you need to create it with:
```python
import fast_clip

ref_model, _, _ = fast_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
```

### Evaluation

**Data**:
```bash
git clone -b project git@github.com:xywei00/datacomp.git
python ./datacomp/download_evalsets.py ./datasets/datacomp
```

To evaluate a trained CLIP model, run the following command:
```bash
# train_output_dir should be the one containing 'checkpoints', 'out.log', etc.
train_output_dir='./logs/fastclipv3'
data_dir='./datasets/datacomp'
arch='ViT-B-16'
epoch=10

python ./datacomp/evaluate.py --train_output_dir "${train_output_dir}" --data_dir "${data_dir}" --epoch "${epoch}" --arch "${arch}"
```
