# data budget

python train_result_collect_new.py --env_id Ant_budget --seed 0 --number_of_env 100 --collect_steps 20000 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth;
python train_result_collect_new.py --env_id Ant_budget --seed 0 --number_of_env 50 --collect_steps 10000 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth;
python train_result_collect_new.py --env_id Ant_budget --seed 0 --number_of_env 25 --collect_steps 5000 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth;
python train_result_collect_new.py --env_id Ant_budget --seed 0 --number_of_env 1 --collect_steps 200 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth;

python Search_gail_Gaussian_Ant.py --env_id Ant_budget --tag traj100 --seed 0 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant_budget/expert/size20000_traj_length200_real_domain_cpu_seed_0.pth;
python Search_gail_Gaussian_Ant.py --env_id Ant_budget --tag traj50 --seed 0 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant_budget/expert/size10000_traj_length200_real_domain_cpu_seed_0.pth;
python Search_gail_Gaussian_Ant.py --env_id Ant_budget --tag traj1 --seed 0 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant_budget/expert/size200_traj_length200_real_domain_cpu_seed_0.pth;

python Search_gail_Gaussian_Ant.py --env_id Ant_budget --tag traj100 --seed 1 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant_budget/expert/size20000_traj_length200_real_domain_cpu_seed_0.pth;
python Search_gail_Gaussian_Ant.py --env_id Ant_budget --tag traj50 --seed 1 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant_budget/expert/size10000_traj_length200_real_domain_cpu_seed_0.pth;
python Search_gail_Gaussian_Ant.py --env_id Ant_budget --tag traj1 --seed 1 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant_budget/expert/size200_traj_length200_real_domain_cpu_seed_0.pth;

python Search_gail_Gaussian_Ant.py --env_id Ant_budget --tag traj100 --seed 2 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant_budget/expert/size20000_traj_length200_real_domain_cpu_seed_0.pth;
python Search_gail_Gaussian_Ant.py --env_id Ant_budget --tag traj50 --seed 2 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant_budget/expert/size10000_traj_length200_real_domain_cpu_seed_0.pth;
python Search_gail_Gaussian_Ant.py --env_id Ant_budget --tag traj1 --seed 2 --expert_weight logs/Ant/SAC_DR_test/seed0-20240419-1558/final_model/actor.pth --expert_data logs/Ant_budget/expert/size200_traj_length200_real_domain_cpu_seed_0.pth;







python train_DR_Search.py --env_id Ant_budget --log_mark SAC_DR_search --tag traj100 --seed 0 --search_params_dir logs/Ant_budget/search_gaussian/seed_0traj100;
python train_DR_Search.py --env_id Ant_budget --log_mark SAC_DR_search --tag traj100 --seed 1 --search_params_dir logs/Ant_budget/search_gaussian/seed_1traj100;
python train_DR_Search.py --env_id Ant_budget --log_mark SAC_DR_search --tag traj100 --seed 2 --search_params_dir logs/Ant_budget/search_gaussian/seed_2traj100;

python train_DR_Search.py --env_id Ant_budget --log_mark SAC_DR_search --tag traj50 --seed 0 --search_params_dir logs/Ant_budget/search_gaussian/seed_0traj50;
python train_DR_Search.py --env_id Ant_budget --log_mark SAC_DR_search --tag traj50 --seed 1 --search_params_dir logs/Ant_budget/search_gaussian/seed_1traj50;
python train_DR_Search.py --env_id Ant_budget --log_mark SAC_DR_search --tag traj50 --seed 2 --search_params_dir logs/Ant_budget/search_gaussian/seed_2traj50;


python train_DR_Search.py --env_id Ant_budget --log_mark SAC_DR_search --tag traj1 --seed 0 --search_params_dir logs/Ant_budget/search_gaussian/seed_0traj1;
python train_DR_Search.py --env_id Ant_budget --log_mark SAC_DR_search --tag traj1 --seed 1 --search_params_dir logs/Ant_budget/search_gaussian/seed_1traj1;
python train_DR_Search.py --env_id Ant_budget --log_mark SAC_DR_search --tag traj1 --seed 2 --search_params_dir logs/Ant_budget/search_gaussian/seed_2traj1;






python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 0 --tag DR  --actor_weight_dir logs/Ant/SAC_DR_test/seed0-20240419-1558/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 1 --tag DR  --actor_weight_dir logs/Ant/SAC_DR_test/seed1-20240419-1626/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 2 --tag DR  --actor_weight_dir logs/Ant/SAC_DR_test/seed2-20240419-1654/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 3 --tag DR  --actor_weight_dir logs/Ant/SAC_DR_test/seed3-20240518-1711/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 4 --tag DR  --actor_weight_dir logs/Ant/SAC_DR_test/seed4-20240518-1711/model;





python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 0 --tag traj1  --actor_weight_dir logs/Ant_budget/SAC_DR_search/traj1/seed0-20240516-1458/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 1 --tag traj1  --actor_weight_dir logs/Ant_budget/SAC_DR_search/traj1/seed1-20240516-1640/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 2 --tag traj1  --actor_weight_dir logs/Ant_budget/SAC_DR_search/traj1/seed2-20240516-1821/model;


python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 0 --tag traj50  --actor_weight_dir logs/Ant_budget/SAC_DR_search/traj50/seed0-20240516-1458/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 1 --tag traj50  --actor_weight_dir logs/Ant_budget/SAC_DR_search/traj50/seed1-20240516-1640/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 2 --tag traj50  --actor_weight_dir logs/Ant_budget/SAC_DR_search/traj50/seed2-20240516-1821/model;


python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 0 --tag traj100  --actor_weight_dir logs/Ant_budget/SAC_DR_search/traj100/seed0-20240516-1458/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 1 --tag traj100  --actor_weight_dir logs/Ant_budget/SAC_DR_search/traj100/seed1-20240516-1640/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 2 --tag traj100  --actor_weight_dir logs/Ant_budget/SAC_DR_search/traj100/seed2-20240516-1821/model;



python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 0 --tag traj200  --actor_weight_dir logs/Ant/SAC_DR_search/seed0-20240419-2051/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 1 --tag traj200  --actor_weight_dir logs/Ant/SAC_DR_search/seed1-20240419-2119/model;
python evaluate_target_domain_OOD_quick.py --env_id Ant_budget --seed 2 --tag traj200  --actor_weight_dir logs/Ant/SAC_DR_search/seed2-20240419-2147/model;
