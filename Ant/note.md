

收集demo

python train_result_collect.py --cuda --env_id Ant-isaac  --weight weights/Ant_actor.pth  --buffer_size 1000000 --std 0.01 --p_rand 0.0 --seed 0 --traj_length 200

python train_imitation.py --algo gail --cuda --env_id InvertedPendulum-v2 --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0


python train_env_policy_by_GAIL.py --expert_weight weights/Ant_actor.pth


set_actor_rigid_shape_properties(self: Gym, arg0: Env, arg1: int, arg2: List[RigidShapeProperties])→ bool

<!-- TODO 发挥IsaacGym的并行优势-->
使用这个方法可以改变摩擦力
现在的写法很不好，只能有一个模型训练，而且修改的参数还是重力。不符合要求
下一步应该把这些东西给弄成，单个env里面可变的参数，然后一堆环境同时跑，再去训练会比较好。


在weights里面，有一个Ant-actor_from_bad.....那个是从sim的场景中训练出来的参数，下一步需要将这个策略放到一个real的环境中。