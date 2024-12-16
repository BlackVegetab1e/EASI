# DR Train
python train_policy/train_DR_Uniform.py --seed 0 --env_id Ant --num_steps $((1*10**7)) --eval_interval $((10**5)) --log_mark SAC_DR;
python train_policy/train_DR_Uniform.py --seed 0 --env_id Cartpole --num_steps $((5*10**5)) --eval_interval $((5*10**3)) --log_mark SAC_DR;
python train_policy/train_DR_Uniform.py --seed 0 --env_id Ballbalance --num_steps $((6*10**6)) --eval_interval $((3*10**4)) --log_mark SAC_DR;

# Demonstration Collect
python train_policy/collect_demo.py --env_id Ant --seed 0 --trajectory_length 200 --collect_steps 40000 --expert_weight example/example_policy/Ant_DR/actor.pth;
python train_policy/collect_demo.py --env_id Cartpole --seed 0 --trajectory_length 200 --collect_steps 40000 --expert_weight example/example_policy/Cartpole_DR/actor.pth;
python train_policy/collect_demo.py --env_id Ballbalance --seed 0 --trajectory_length 500 --collect_steps 100000 --expert_weight example/example_policy/Ballbalance_DR/actor.pth;

# Parameter Search

python Search_gail_Gaussian.py --env_id Ant --tag WD --seed 0 --expert_weight example/example_policy/Ant_DR/actor.pth --expert_data example/example_expert_state_trans/Ant_DR/size40000_traj_length200_real_domain_cpu_seed_0.pth;
python Search_gail_Gaussian.py --env_id Cartpole --tag WD --seed 0 --epoch_disc 10 --expert_weight example/example_policy/Cartpole_DR/actor.pth --expert_data example/example_expert_state_trans/Cartpole_DR/size40000_traj_length200_real_domain_cpu_seed_0.pth;
python Search_gail_Gaussian.py --env_id Ballbalance --tag WD --seed 0 --trajectory_length 500 --expert_weight example/example_policy/Ballbalance_DR/actor.pth --expert_data example/example_expert_state_trans/Ballbalance_DR/size100000_traj_length500_real_domain_cpu_seed_0.pth;

# Search Train
python train_policy/train_DR_Search.py --seed 0 --env_id Ant --num_steps $((1*10**7)) --eval_interval $((10**5)) --log_mark SAC_Search --search_params_dir logs/Ant/search_gaussian/WDWD/seed_0;
python train_policy/train_DR_Search.py --seed 0 --env_id Cartpole --num_steps $((5*10**5)) --eval_interval $((5*10**3)) --log_mark SAC_Search --search_params_dir logs/Cartpole/search_gaussian/WDWD/seed_0; 
python train_policy/train_DR_Search.py --seed 0 --env_id Ballbalance --num_steps $((6*10**6)) --eval_interval $((3*10**4)) --log_mark SAC_Search --search_params_dir logs/Ballbalance/search_gaussian/WDWD/seed_0;


# Baseline ORACLE train 
python train_policy/train_expert.py --seed 0 --env_id Ant --num_steps $((1*10**7)) --eval_interval $((10**5));
python train_policy/train_expert.py --seed 0 --env_id Cartpole --num_steps $((5*10**5)) --eval_interval $((5*10**3));
python train_policy/train_expert.py --seed 0 --env_id Ballbalance --num_steps $((6*10**6)) --eval_interval $((3*10**4));


# Evaluate DR policy
python train_policy/evaluate_target_domain.py --env_id Ant --tag DR --seed 0 --actor_weight_dir logs/Ant/SAC_DR/seed0-20241216-1130/model;
python train_policy/evaluate_target_domain.py --env_id Cartpole --tag DR --seed 0 --actor_weight_dir logs/Cartpole/SAC_DR/seed0-20241216-1134/model;
python train_policy/evaluate_target_domain.py --env_id Ballbalance --tag DR --seed 0 --evaluate_steps 50000 --actor_weight_dir logs/Ballbalance/SAC_DR/seed0-20241216-1135/model;


# Evaluate Search policy
python train_policy/evaluate_target_domain.py --env_id Ant --tag search --seed 0 --actor_weight_dir logs/Ant/SAC_Search/seed0-20241216-1355/model;
python train_policy/evaluate_target_domain.py --env_id Cartpole --tag search --seed 0 --actor_weight_dir logs/Cartpole/SAC_Search/seed0-20241216-1355/model;
python train_policy/evaluate_target_domain.py --env_id Ballbalance --tag search --seed 0 --evaluate_steps 50000 --actor_weight_dir logs/Ballbalance/SAC_Search/seed0-20241216-1355/model;