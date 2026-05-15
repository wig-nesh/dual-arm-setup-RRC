import pybullet as p
import pybullet_data
import time
import numpy as np
import os


class XArmSim:
    def __init__(self, urdf_path=None):
        if urdf_path is None:
            # Fallback to a common path or let user provide it
            self.urdf_path = "/home/wignesh/Documents/rrc/test/gym-xarm/gym_xarm/envs/urdf/xarm7.urdf"
        else:
            self.urdf_path = urdf_path

        self.physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        # Load environment
        p.loadURDF("plane.urdf")
        self.tableId = p.loadURDF("table/table.urdf", [0.4, 0, 0], useFixedBase=True)

        # Load xArm
        startPos = [0, 0, 0.625]
        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.xarmId = p.loadURDF(
            self.urdf_path, startPos, startOrientation, useFixedBase=True
        )

        self.arm_joint_indices = [1, 2, 3, 4, 5, 6, 7]
        self.arm_eef_index = 8

    def set_joint_positions(self, joint_angles):
        """Sets the simulation arm to the given joint angles (degrees)."""
        # xArmAPI returns degrees, PyBullet expects radians
        for idx, j_idx in enumerate(self.arm_joint_indices):
            rad = np.deg2rad(joint_angles[idx])
            # resetJointState is an immediate "teleport"
            p.resetJointState(self.xarmId, j_idx, rad)
            # setJointMotorControl2 ensures the joints stay there during simulation steps
            p.setJointMotorControl2(
                self.xarmId, j_idx, p.POSITION_CONTROL, targetPosition=rad, force=1000
            )
        # Settle the simulation to ensure forward kinematics are stable
        for _ in range(20):
            p.stepSimulation()

    def get_eef_pose(self):
        """Returns current EEF [x, y, z] and [qx, qy, qz, qw] in world space."""
        state = p.getLinkState(
            self.xarmId, self.arm_eef_index, computeForwardKinematics=1
        )
        return state[0], state[1]

    def simulate_trajectory(
        self, start_angles, target_pos_offset=[0, 0, 0.1], steps=100
    ):
        """
        Animates a trajectory from start_angles to target (start_eef + offset).
        Returns the final joint angles in degrees.
        """
        # Ensure the simulation starts exactly at the robot's current pose
        self.set_joint_positions(start_angles)

        # Force immediate update of link states
        p.stepSimulation()
        initial_pos, initial_orn = self.get_eef_pose()

        # DEBUG: print start position in sim
        print(f"Simulation Start Pose: {initial_pos}")

        target_pos = [
            initial_pos[0] + target_pos_offset[0],
            initial_pos[1] + target_pos_offset[1],
            initial_pos[2] + target_pos_offset[2],
        ]

        final_angles_deg = []

        print(f"Simulating movement to offset {target_pos_offset}...")
        for i in range(steps):
            fraction = (i + 1) / steps
            curr_target = [
                initial_pos[0] + (target_pos_offset[0] * fraction),
                initial_pos[1] + (target_pos_offset[1] * fraction),
                initial_pos[2] + (target_pos_offset[2] * fraction),
            ]

            jointPoses = p.calculateInverseKinematics(
                self.xarmId,
                self.arm_eef_index,
                targetPosition=curr_target,
                targetOrientation=initial_orn,
                solver=p.IK_DLS,
            )

            for idx, j_idx in enumerate(self.arm_joint_indices):
                p.setJointMotorControl2(
                    self.xarmId,
                    j_idx,
                    p.POSITION_CONTROL,
                    targetPosition=jointPoses[idx],
                    force=1000,
                )

            p.stepSimulation()
            time.sleep(1.0 / 240.0)

            if i == steps - 1:
                # Convert back to degrees for the real arm
                final_angles_deg = [np.rad2deg(ang) for ang in jointPoses[:7]]

        return final_angles_deg

    def close(self):
        p.disconnect()
