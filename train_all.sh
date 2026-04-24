python trainvae.py --logdir exp_dir


python trainmdrnn.py --logdir exp_dir

xvfb-run -a -s "-screen 0 1400x900x24" python traincontroller_rnn.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display



python trainmdattn.py --logdir exp_dir

xvfb-run -a -s "-screen 0 1400x900x24" python traincontroller_attn.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display
xvfb-run -a -s "-screen 0 1400x900x24" python traincontroller_attn.py --logdir exp_dir --n-samples 4 --pop-size 4 --target-return 950 --display --reload