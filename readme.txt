environment setup:
cd feature-point-refinement
conda env create -f environment.yml
pip install -e .

training prepration:
prepare Megadepth data in the refine_pkg/data folder as LoFTR did
https://github.com/zju3dv/LoFTR

clone baseline method from LightGlue and SuperGlue:
https://github.com/magicleap/SuperGluePretrainedNetwork
https://github.com/cvg/LightGlue

training command:
modify config files first in refine_pkg/config

then run:
python train.py --model sp_lg
