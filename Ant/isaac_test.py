import time

from isaacgym import gymapi, gymtorch

gym = gymapi.acquire_gym()
class RigidContactEnv:

    def __init__(self, env_cfg: dict, cuda_id: int = 0, use_viewer: bool = True):
        self.env_cfg = env_cfg

        # Setup simulation instance and viewer (if requested).
        self.gym = gymapi.acquire_gym()
        self.sim = self._create_sim(cuda_id)
        self.viewer = self._create_viewer() if use_viewer else None
        if "plane" in self.env_cfg["base"].keys():
            self._create_plane()

        # Load assets.
        self.asset_options = self._create_asset_options()
        self.assets_dict = self._load_assets()

        # Build scene.
        self.scene_dict = self._create_scene()

    def _create_sim(self, gpu_id: int):
        """
        Setup simulation parameters.
        """
        sim_type = gymapi.SIM_PHYSX
        sim_params = gymapi.SimParams()
        sim_params.dt = 1 / 60.
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)

        sim = self.gym.create_sim(gpu_id, gpu_id, sim_type, sim_params)
        return sim

    def _load_assets(self) -> dict:
        assets_dict = {}

        # Load box.
        options = self.asset_options
        options.fix_base_link = False
        options.disable_gravity = False
        box_asset = self.gym.create_box(self.sim, 0.1, 0.1, 0.1, options)
        assets_dict["box"] = box_asset

        # Attach force sensor to box.
        sensor_props = gymapi.ForceSensorProperties()
        sensor_props.use_world_frame = True
        box_sensor = self.gym.create_asset_force_sensor(box_asset, 0, gymapi.Transform(), sensor_props)
        assets_dict["box_sensor"] = box_sensor

        # Load table.
        options = self.asset_options
        options.fix_base_link = True
        options.disable_gravity = True
        table_asset = self.gym.create_box(self.sim, 20.0, 20.0, 0.5, options)
        assets_dict["table"] = table_asset

        return assets_dict

    def _create_asset_options(self):
        """
        Set asset options common to all assets.
        TODO: Parameterize from config?
        """
        options = gymapi.AssetOptions()
        options.flip_visual_attachments = False
        options.armature = 0.0
        options.thickness = 0.0
        options.linear_damping = 0.0
        options.angular_damping = 0.0
        options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        options.min_particle_mass = 1e-20

        return options

    def _create_viewer(self):
        """
        Create viewer and axes objects.
        """
        cam_pos = self.env_cfg["base"]["cam_pos"]
        cam_target = self.env_cfg["base"]["cam_target"]

        camera_props = gymapi.CameraProperties()
        camera_props.horizontal_fov = 5.0
        camera_props.width = 1920
        camera_props.height = 1080
        viewer = self.gym.create_viewer(self.sim, camera_props)
        camera_pos = gymapi.Vec3(cam_pos[0], cam_pos[1], cam_pos[2])
        camera_target = gymapi.Vec3(cam_target[0], cam_target[1], cam_target[2])
        self.gym.viewer_camera_look_at(viewer, None, camera_pos, camera_target)
        return viewer

    def _create_plane(self):
        # Add plane.
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(self.env_cfg["base"]["plane"]["normal"][0],
                                          self.env_cfg["base"]["plane"]["normal"][1],
                                          self.env_cfg["base"]["plane"]["normal"][2])
        plane_params.segmentation_id = self.env_cfg["base"]["plane"]["segmentation_id"]
        plane_params.static_friction = self.env_cfg["base"]["plane"]["static_friction"]
        plane_params.dynamic_friction = self.env_cfg["base"]["plane"]["dynamic_friction"]
        plane_params.distance = self.env_cfg["base"]["plane"]["distance"]
        self.gym.add_ground(self.sim, plane_params)

    def _create_scene(self) -> dict:
        scene_dict = {}

        # Create environment.
        env_dim = 0.5
        env_handle = self.gym.create_env(self.sim,
                                         gymapi.Vec3(-env_dim, -env_dim, -env_dim),
                                         gymapi.Vec3(env_dim, env_dim, env_dim), 1)
        scene_dict["env"] = env_handle

        # Add box.
        box_pose = gymapi.Transform()
        box_pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        box_pose.r = gymapi.Quat(0.99638031, 0.04735953, 0.05234912, 0.04735953)
        box_handle = self.gym.create_actor(env_handle, self.assets_dict["box"], box_pose, "box", 0, 1)
        scene_dict["box"] = box_handle

        box_props = self.gym.get_actor_rigid_shape_properties(env_handle, box_handle)
        box_props[0].friction = 1.0
        box_props[0].rolling_friction = 1.0
        box_props[0].torsion_friction = 1.0

        self.gym.set_actor_rigid_shape_properties(env_handle, box_handle, box_props)

        # "Connect" to box sensor.
        assert self.gym.get_actor_force_sensor_count(env_handle, box_handle) == 1
        box_sensor_handle = self.gym.get_actor_force_sensor(env_handle, box_handle, 0)
        scene_dict["box_sensor"] = box_sensor_handle

        # Add table.
        table_pose = gymapi.Transform()
        table_handle = self.gym.create_actor(env_handle, self.assets_dict["table"], table_pose, "table", 0, 2)
        scene_dict["table"] = table_handle

        table_props = self.gym.get_actor_rigid_shape_properties(env_handle, table_handle)
        table_props[0].friction = 1.0
        table_props[0].rolling_friction = 1.0
        table_props[0].torsion_friction = 1.0
        self.gym.set_actor_rigid_shape_properties(env_handle, table_handle, table_props)


  

        return scene_dict

    def step(self):
        """
        Step simulation.
        """
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)

        if self.viewer is not None:
            self.render()

    def render(self):
        self.gym.step_graphics(self.sim)
        self.gym.draw_env_rigid_contacts(self.viewer, self.scene_dict["env"], gymapi.Vec3(1.0, 0.5, 0.0), 1.0, False)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.clear_lines(self.viewer)

    def reset(self):
        rb_state = self.gym.get_actor_rigid_body_states(self.scene_dict["env"], self.scene_dict["box"],
                                                        gymapi.STATE_ALL)
        rb_state[0]["vel"]["linear"]["x"] = 5.0
        self.gym.set_actor_rigid_body_states(self.scene_dict["env"], self.scene_dict["box"], rb_state, gymapi.STATE_VEL)

    def run_sim(self):
        while True:
            self.step()

            rigid_contacts = self.gym.get_env_rigid_contacts(self.scene_dict["env"])
            sensor_data = self.scene_dict["box_sensor"].get_forces()
            if len(rigid_contacts) > 0:
                print("contact")
            gym.refresh_net_contact_force_tensor(self.sim)
            _net_cf = gym.acquire_net_contact_force_tensor(self.sim)
            net_cf = gymtorch.wrap_tensor(_net_cf)
            
            print(net_cf)

            time.sleep(0.1)


if __name__ == '__main__':
    env_cfg = {
        "base": {
            "cam_pos": [20.0, 20.0, 3.0],
            "cam_target": [3.0, 0.0, 0.25],
            "plane": {
                "normal": [0.0, 0.0, 1.0],
                "segmentation_id": 1,
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "distance": 0.0
            }
        },
    }

    env = RigidContactEnv(env_cfg)
    env.reset()
    env.run_sim()