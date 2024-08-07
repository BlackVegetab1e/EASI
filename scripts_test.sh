seed=4


python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 0.25 --actor_weight_dir logs/BallBalance_Long/SAC_DR/seed0-20240514-0012/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 0.5 --actor_weight_dir logs/BallBalance_Long/SAC_DR/seed0-20240514-0012/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 1.0 --actor_weight_dir logs/BallBalance_Long/SAC_DR/seed0-20240514-0012/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 2.0 --actor_weight_dir logs/BallBalance_Long/SAC_DR/seed0-20240514-0012/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 4.0 --actor_weight_dir logs/BallBalance_Long/SAC_DR/seed0-20240514-0012/model;

python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 0.25 --actor_weight_dir logs/BallBalance_Long_OOD/SAC_DR_search/seed0-20240514-0948-ood0.25/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 0.5 --actor_weight_dir logs/BallBalance_Long_OOD/SAC_DR_search/seed0-20240514-1145-ood0.5/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 1.0 --actor_weight_dir logs/BallBalance_Long/SAC_DR_search/seed0-20240514-0948/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 2.0 --actor_weight_dir logs/BallBalance_Long_OOD/SAC_DR_search/seed0-20240514-0948-ood2.0/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'DR' --OOD_rate 4.0 --actor_weight_dir logs/BallBalance_Long_OOD/SAC_DR_search/seed0-20240514-1316-ood4.0/model;

python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'Baseline' --OOD_rate 0.25 --actor_weight_dir logs/BallBalance_Long_OOD/SAC_baseline/seed0-20240514-0012-OOD0.25/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'Baseline' --OOD_rate 0.5 --actor_weight_dir logs/BallBalance_Long_OOD/SAC_baseline/seed0-20240514-0142-OOD0.5/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'Baseline' --OOD_rate 1.0 --actor_weight_dir logs/BallBalance_Long_OOD/SAC_baseline/seed0-20240514-0313-OOD1.0/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'Baseline' --OOD_rate 2.0 --actor_weight_dir logs/BallBalance_Long_OOD/SAC_baseline/seed0-20240514-0424-OOD2.0/model;
python evaluate_target_domain_OOD_quick.py --env_id BallBalance_Long --seed $seed --tag 'Baseline' --OOD_rate 4.0 --actor_weight_dir logs/BallBalance_Long_OOD/SAC_baseline/seed0-20240514-0523-OOD4.0/model;







