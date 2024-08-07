python trainGARAT_Policy.py --seed 0 --OOD_RATE 0.25 --GAT_policy logs/Ant_GAT_pure/GAT_OOD/OOD025/seed1/model/isaac_step42000000/actor.pth;

python trainGARAT_Policy.py --seed 0 --OOD_RATE 0.5 --GAT_policy logs/Ant_GAT_pure/GAT_OOD/OOD05/seed1/model/isaac_step42000000/actor.pth;





python trainGARAT.py --tag OOD05 --seed 0 --expert_data logs/Ant_OOD/expert/OOD_RATE_0.5_40000_traj_length200_real_domain_cpu_seed_0.pth --headless True

python trainGARAT.py --tag OOD025 --seed 0 --expert_data logs/Ant_OOD/expert/OOD_RATE_0.25_40000_traj_length200_real_domain_cpu_seed_0.pth --headless True


python train_finetune.py --OOD_rate 0.5 --seed 0
python train_finetune.py --OOD_rate 0.25 --seed 0



python evaluate_target_domain_OOD_quick.py --tag SAC_DR_TF --seed 0 --actor_weight_dir logs/Ant/SAC_DR_FT_new/0.25/seed2-20240806-1404/model --OOD --OOD_rate 0.25
python evaluate_target_domain_OOD_quick.py --tag SAC_DR_TF --seed 0 --actor_weight_dir logs/Ant/SAC_DR_FT_new/0.5/seed2-20240806-1404/model --OOD --OOD_rate 0.5