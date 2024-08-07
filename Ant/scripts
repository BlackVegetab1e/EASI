python train_DR_Search.py --seed 0 --search_params_dir logs/Ant/search_gaussian/seed_0;
python train_DR_Search.py --seed 1 --search_params_dir logs/Ant/search_gaussian/seed_1;
python train_DR_Search.py --seed 2 --search_params_dir logs/Ant/search_gaussian/seed_2;


python train_DR_Uniform.py --seed 0;
python train_DR_Uniform.py --seed 1;
python train_DR_Uniform.py --seed 2;


python train_expert.py --seed 0;
python train_expert.py --seed 1;
python train_expert.py --seed 2;



python evaluate_target_domain.py --seed 0 --actor_weight_dir logs/Ant/SAC_DR_search/seed0-20240419-2051/model --logmark SAC_DR_search;
python evaluate_target_domain.py --seed 1 --actor_weight_dir logs/Ant/SAC_DR_search/seed1-20240419-2119/model --logmark SAC_DR_search;
python evaluate_target_domain.py --seed 2 --actor_weight_dir logs/Ant/SAC_DR_search/seed2-20240419-2147/model --logmark SAC_DR_search;


python Search_gail_Gaussian_Ant.py --env_id Ant --OOD --tag OOD_1.0 --seed 0 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant/expert/size40000_traj_length200_real_domain_cpu_seed_0.pth;
