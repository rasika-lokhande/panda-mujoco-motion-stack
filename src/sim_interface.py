import mujoco
import mujoco.viewer
import numpy as np

from utils.config import config


class SimInterface:

    def __init__(self, model_path: str, add_target_mocap: bool = False):
        if add_target_mocap:
            self.model = self._load_with_target_mocap(model_path)
        else:
            self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self._load_joint_info()
        self._load_body_info()
        self._load_site_info()
        self.theta_home = [0, 0, 0, -1.57079, 0, 1.57079, -0.7853]
        self.set_joint_angles(self.theta_home) # set to ready pose

    @staticmethod
    def _load_with_target_mocap(model_path):
        # Adds a draggable, non-colliding mocap body named "target" (no joints, so qpos is unchanged)
        spec = mujoco.MjSpec.from_file(model_path)
        body = spec.worldbody.add_body(name="target", mocap=True)
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0],
                      rgba=[0.2, 0.9, 0.3, 0.3], contype=0, conaffinity=0)
        return spec.compile()

    def _load_joint_info(self):
        self.joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        self.jid_list = list(range(self.model.njnt))
        self.jnt_pos_list = [self.model.jnt_pos[jid] for jid in self.jid_list]
        self.jnt_axis_list = [self.model.jnt_axis[jid] for jid in self.jid_list]

    def _load_body_info(self):
        self.bid_list = [self.model.jnt_bodyid[jid] for jid in self.jid_list]
        self.body_pos_list = [self.model.body_pos[bid] for bid in self.bid_list]
        self.body_quat_list = [self.model.body_quat[bid] for bid in self.bid_list]
        self.body_parentid_list = [self.model.body_parentid[bid] for bid in self.bid_list]

    def _load_site_info(self):
        self.site_names = [self.model.site(i).name for i in range(self.model.nsite)]
        self.sid_list = list(range(self.model.nsite))
        self.site_bodyid_list = [self.model.site_bodyid[sid] for sid in self.sid_list]

    def step(self):
        mujoco.mj_step(self.model, self.data)

    def run_viewer(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                viewer.sync()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    # ============ Accessor functions ============

    def get_joint_names(self):
        return self.joint_names

    def get_joint_id_list(self):
        return self.jid_list

    def get_joint_positions(self):
        return self.jnt_pos_list

    def get_joint_axes(self):
        return self.jnt_axis_list

    def get_body_id_list(self):
        return self.bid_list

    def get_body_positions(self):
        return self.body_pos_list

    def get_body_orientations(self):
        return self.body_quat_list

    def get_body_parent_ids(self):
        return self.body_parentid_list

    def get_site_names(self):
        return self.site_names

    def get_site_id_list(self):
        return self.sid_list

    def get_site_id(self, site_name: str):
        return self.model.site(site_name).id

    def get_site_body_id(self, site_id: int):
        return self.model.site_bodyid[site_id]


    def get_current_joint_angles_list(self):
        return self.data.qpos.copy()

    def get_current_joint_angle(self, jid):
        qadr = self.model.jnt_qposadr[jid]
        return self.data.qpos[qadr]
    

    def set_joint_angles(self, angles):
        self.data.qpos[:] = angles
        mujoco.mj_forward(self.model, self.data)

    def get_ee_pose(self, ee_site_id):
        ## In built FK
        pos = self.data.site_xpos[ee_site_id].copy()
        rot = self.data.site_xmat[ee_site_id].copy().reshape(3, 3)
        return pos, rot
    



    # ============ Debug ============

    def print_info(self):
        print("=== Joints ===")
        for jid, name in zip(self.jid_list, self.joint_names):
            print(f"  {jid}: {name}")
            print(f"    body_id:  {self.bid_list[jid]}")
            print(f"    jnt_pos:  {self.jnt_pos_list[jid]}")
            print(f"    jnt_axis: {self.jnt_axis_list[jid]}")

        print("\n=== Bodies (parents of joints) ===")
        for jid, bid in enumerate(self.bid_list):
            print(f"  body {bid} (joint '{self.joint_names[jid]}'):")
            print(f"    parent_id: {self.body_parentid_list[jid]}")
            print(f"    body_pos:  {self.body_pos_list[jid]}")
            print(f"    body_quat: {self.body_quat_list[jid]}")

        print("\n=== Sites ===")
        for sid, name in zip(self.sid_list, self.site_names):
            print(f"  {sid}: {name}")
            print(f"    body_id: {self.site_bodyid_list[sid]}")

        print(f"\nnq (position dims): {self.model.nq}")
        print(f"nv (velocity dims): {self.model.nv}")


        print("\n=== Current Joint Angles ===")
        print({str(self.get_current_joint_angles_list())})

        

if __name__ == "__main__":
    sim = SimInterface(model_path=config["MODEL_PATH"])

    sim.print_info()
    #sim.run_viewer()