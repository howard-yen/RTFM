default-config:
	$(eval CONFIGS=--demb 30 --drnn_small 10 --drnn 100 --drep 400 --num_actors 20 --batch_size 24 --learning_rate 0.0007 --total_frames=100000000 --height 6 --width 6)

experiment-config:
	$(eval CONFIGS=--drep 400 --num_actors 20 --batch_size 24 --learning_rate 0.0007 --total_frames 1000000 --height 6 --width 6 --num_threads 8 --unroll_length 80 --env rtfm:groups_simple_stationary-v0)

debug-config:
	$(eval CONFIGS=--drep 400 --num_actors 1 --batch_size 1 --learning_rate 0.0007 --total_frames 100 --height 6 --width 6 --num_threads 1 --unroll_length 30 --env rtfm:groups_simple_stationary-v0)


text-only:
	$(eval CHECKPOINT_DIR=checkpoints/groups_simple_stationary:text_only:yeswiki:default)
	$(eval MODEL_TYPE=text_only)

text-only-tiny:
	$(eval CHECKPOINT_DIR=checkpoints/groups_simple_stationary:text_only_tiny:yeswiki:default)
	$(eval MODEL_TYPE=text_only_tiny)

visual:
	$(eval CHECKPOINT_DIR=checkpoints/groups_simple_stationary:vbert:yeswiki:default)
	$(eval MODEL_TYPE=vbert)

run-txt2pi: default-config
	python run_exp.py \
		--env rtfm:groups_simple_stationary-v0 \
		--model paper_txt2pi \
		$(CONFIGS)

train:
	python run_exp.py \
		--env rtfm:groups_simple_stationary_nl-v0 \
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

train-text: text-only experiment-config
	python run_exp.py \
		--model $(MODEL_TYPE) \
		--mode train \
		$(CONFIGS)

train-text-tiny: text-only-tiny experiment-config
	python run_exp.py \
		--model $(MODEL_TYPE) \
		--mode train \
		$(CONFIGS)

train-visual: visual debug-config
	python run_exp.py \
		--model $(MODEL_TYPE) \
		--mode train \
		$(CONFIGS)

resume-train-text: text-only
	python run_exp.py \
		--env rtfm:groups_simple_stationary-v0 \
		--model text_only \
		--mode train \
		--resume $(CHECKPOINT_DIR)/model.tar \
		$(CONFIGS)


plot-log: text-only-tiny
	python visualize_log.py \
		--log_file $(CHECKPOINT_DIR)/logs.csv

play:
	python play_gym.py -c --env groups_nl --shuffle_wiki

pickle-image:
	python pickle_img_tensors.py
