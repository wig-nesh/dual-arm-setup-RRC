import os
import sys
import numpy as np
import trimesh
import open3d as o3d
import argparse
import random

# Add project root and current directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import ModelClient


def get_partial_pcd(mesh, n_points=2048):
    """
    Simulates a partial view pointcloud from a random viewpoint.
    """
    num_oversampled = int(n_points * 3)
    pcd_full, _ = trimesh.sample.sample_surface(mesh, num_oversampled)

    # Angle ranges from user instructions
    distance = random.uniform(1.5, 2.0)
    azimuth = np.radians(random.choice([-70, -80, -90, -100, -110]))
    elevation = np.radians(random.uniform(20, 40))

    cam_pos = np.array(
        [
            distance * np.cos(elevation) * np.cos(azimuth),
            distance * np.cos(elevation) * np.sin(azimuth),
            distance * np.sin(elevation),
        ]
    )

    ray_origins = np.repeat(cam_pos[None, :], num_oversampled, axis=0)
    ray_dirs = pcd_full - ray_origins
    ray_dist = np.linalg.norm(ray_dirs, axis=1)
    ray_dirs = ray_dirs / ray_dist[:, None]

    # Ray tracing to find visible points
    locations, index_ray, index_tri = mesh.ray.intersects_location(
        ray_origins=ray_origins, ray_directions=ray_dirs, multiple_hits=False
    )

    if len(locations) == 0:
        raise RuntimeError("No visible points found from given view")

    hit_dist = np.linalg.norm(locations - ray_origins[index_ray], axis=1)

    # Check which sampled points are actually visible (not occluded)
    eps = 1e-4
    visible_mask = np.abs(hit_dist - ray_dist[index_ray]) < eps
    visible_indices = index_ray[visible_mask]

    if len(visible_indices) == 0:
        raise RuntimeError("No visible points found after occlusion check")

    # Resample to exact n_points
    if len(visible_indices) >= n_points:
        choice = np.random.choice(visible_indices, n_points, replace=False)
    else:
        choice = np.random.choice(visible_indices, n_points, replace=True)

    return pcd_full[choice].astype(np.float32)


def visualize_results(pcd_np, grasp_pairs_np, pair_scores_np, gripper_mesh_path):
    """
    Visualizes the pointcloud and predicted grasp pairs in Open3D with interactive cycling.
    """
    if grasp_pairs_np.shape[0] == 0:
        print("No grasp pairs to visualize.")
        return

    # Sort pairs by score (highest first)
    if pair_scores_np is not None:
        sort_idx = np.argsort(pair_scores_np)[::-1]
        grasp_pairs_np = grasp_pairs_np[sort_idx]
        pair_scores_np = pair_scores_np[sort_idx]

    # Load base meshes
    base_gripper = o3d.io.read_triangle_mesh(gripper_mesh_path)
    base_gripper.compute_vertex_normals()
    base_gripper.scale(8.0, center=(0, 0, 0))

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

    # Dynamic meshes (to be updated)
    g1 = o3d.geometry.TriangleMesh(base_gripper)
    g1.paint_uniform_color([0.8, 0.2, 0.2])
    g2 = o3d.geometry.TriangleMesh(base_gripper)
    g2.paint_uniform_color([0.2, 0.2, 0.8])

    f1 = o3d.geometry.TriangleMesh(base_frame)
    f2 = o3d.geometry.TriangleMesh(base_frame)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_np)
    pcd.paint_uniform_color([0.5, 0.5, 0.5])

    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8)

    state = {"idx": 0}

    def update_meshes(vis):
        idx = state["idx"]
        pair = grasp_pairs_np[idx]
        score_str = (
            f" | Score: {pair_scores_np[idx]:.4f}" if pair_scores_np is not None else ""
        )
        print(f"Showing Grasp Pair {idx + 1}/{len(grasp_pairs_np)}{score_str}")

        # Reset vertices to base and transform
        import copy

        g1.vertices = base_gripper.vertices
        g1.transform(pair[0])
        g1.compute_vertex_normals()

        g2.vertices = base_gripper.vertices
        g2.transform(pair[1])
        g2.compute_vertex_normals()

        f1.vertices = base_frame.vertices
        f1.transform(pair[0])
        f1.compute_vertex_normals()

        f2.vertices = base_frame.vertices
        f2.transform(pair[1])
        f2.compute_vertex_normals()

        vis.update_geometry(g1)
        vis.update_geometry(g2)
        vis.update_geometry(f1)
        vis.update_geometry(f2)

    def next_grasp(vis):
        state["idx"] = (state["idx"] + 1) % len(grasp_pairs_np)
        update_meshes(vis)
        return False

    def prev_grasp(vis):
        state["idx"] = (state["idx"] - 1) % len(grasp_pairs_np)
        update_meshes(vis)
        return False

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Dual Grasp Viewer (Right/Left Arrow to cycle)")

    vis.add_geometry(pcd)
    vis.add_geometry(world_frame)
    vis.add_geometry(g1)
    vis.add_geometry(g2)
    vis.add_geometry(f1)
    vis.add_geometry(f2)

    # GLFW Key codes: 262 is Right Arrow, 263 is Left Arrow
    vis.register_key_callback(262, next_grasp)
    vis.register_key_callback(263, prev_grasp)

    print("\nOpening Open3D Visualization...")
    print("Red Gripper: Arm 1 | Blue Gripper: Arm 2")
    print("Use RIGHT and LEFT arrow keys to cycle through grasp pairs.")

    # Initialize first grasp
    update_meshes(vis)

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Model Client Trajectory Test")
    parser.add_argument(
        "--url", type=str, default="http://localhost:8000", help="Model Server URL"
    )
    parser.add_argument(
        "--obj", type=str, default="client/test_object.obj", help="Path to test object"
    )
    parser.add_argument(
        "--gripper", type=str, default="client/gripper.obj", help="Path to gripper mesh"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Only visualize the point cloud and world frame, skip server request.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.obj):
        print(f"Error: Object not found at {args.obj}")
        return

    # 1. Load Mesh
    print(f"Loading mesh from {args.obj}...")
    mesh = trimesh.load(args.obj)

    # 2. Generate Partial Pointcloud
    print("Generating partial view pointcloud...")
    pcd_np = get_partial_pcd(mesh, n_points=2048)

    # Scale pointcloud by 8x as requested
    pcd_np = pcd_np * 8.0
    print(f"PCD generated and scaled by 8x: {pcd_np.shape}")

    if args.test:
        print("\nTest mode enabled. Visualizing point cloud only...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_np)
        pcd.paint_uniform_color([0.5, 0.5, 0.5])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, frame])
        return

    # 3. Call Server
    client = ModelClient(args.url, timeout=120.0)
    print(f"Sending request to {args.url}/predict...")

    try:
        results = client.predict(pcd_np)

        if "refined_grasp_pairs" in results:
            grasp_pairs = results["refined_grasp_pairs"]
            pair_scores = results.get("pair_scores", None)

            print(f"Received {grasp_pairs.shape[0]} refined grasp pairs.")

            # 4. Visualize
            if os.path.exists(args.gripper):
                visualize_results(pcd_np, grasp_pairs, pair_scores, args.gripper)
            else:
                print(
                    f"Warning: Gripper mesh not found at {args.gripper}, skipping visualization."
                )
        else:
            print("Server returned success but no 'refined_grasp_pairs' in response.")
            print(f"Keys found: {list(results.keys())}")

    except Exception as e:
        print(f"Inference failed: {e}")


if __name__ == "__main__":
    main()
