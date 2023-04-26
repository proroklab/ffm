CONFIGS_RNN_P=(configs/pomdp/ant_blt/p/rnn.yml
	configs/pomdp/cheetah_blt/p/rnn.yml
	configs/pomdp/hopper_blt/p/rnn.yml
	configs/pomdp/walker_blt/p/rnn.yml)

CONFIGS_RNN_V=(configs/pomdp/hopper_blt/v/rnn.yml
	configs/pomdp/walker_blt/v/rnn.yml
	configs/pomdp/ant_blt/v/rnn.yml
	configs/pomdp/cheetah_blt/v/rnn.yml)

CONFIGS_FFM_P=(configs/pomdp/ant_blt/p/ffm.yml
	configs/pomdp/cheetah_blt/p/ffm.yml
	configs/pomdp/hopper_blt/p/ffm.yml
	configs/pomdp/walker_blt/p/ffm.yml)

CONFIGS_FFM_V=(configs/pomdp/hopper_blt/v/ffm.yml
	configs/pomdp/walker_blt/v/ffm.yml
	configs/pomdp/ant_blt/v/ffm.yml
	configs/pomdp/cheetah_blt/v/ffm.yml)

SEED=$1
CUDA=$2
declare -n CONFIGS=$3

echo $CONFIGS

for cfg in ${CONFIGS[@]}; do
	python3 policies/main.py --cfg $cfg --cuda $2 --seed $1 &
done
wait
