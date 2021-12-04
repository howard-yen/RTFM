default-config:
	$(eval CONFIGS=--demb 30 --drnn_small 10 --drnn 100 --drep 400 --num_actors 20 --batch_size 24 --learning_rate 0.0007 --total_frames=100000000 --height 6 --width 6)

run-txt2pi: default-config
	python run_exp.py \
		--env rtfm:groups_simple_stationary-v0 \
		--model paper_txt2pi \
		$(CONFIGS)

train:
	python run_exp.py \
		--env rtfm:groups_simple_stationary-v0 \
		--model paper_txt2pi \
		--mode train \
		--demb 30 \
		--drnn_small 10 \
		--drnn 100 \
		--drep 400 \
		--num_actors 8 \
		--batch_size 16 \
		--learning_rate 0.0007 \
		--total_frames 10000 \
		--height 6 \
		--width 6 \
		--num_threads 8

train-text:
	python run_exp.py \
		--env rtfm:groups_simple_stationary-v0 \
		--model text_only \
		--mode train \
		--demb 30 \
		--drnn_small 10 \
		--drnn 100 \
		--drep 400 \
		--num_actors 20 \
		--batch_size 14 \
		--learning_rate 0.0007 \
		--total_frames 1000000 \
		--height 6 \
		--width 6 \
		--num_threads 8 \
		--unroll_length 80


play:
	python play_gym.py -c --env groups_nl --shuffle_wiki

pickle-image:
	python pickle_img_tensors.py