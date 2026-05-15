# DEPENDENCIES NOT HANDLED!!! figure it out yourself on server
# this is on 117

import io
import json
import random
import traceback

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, request, Response, jsonify

from se3dif.utils import load_experiment_specifications
from se3dif.models import loader
from se3dif.losses.grasp_regression_loss import params_to_grasp_width
from se3dif.utils.geometry_utils import (
    grasp_pair_combinations,
    farthest_point_sampling_seeded,
)


# =====================================================================================
# Seeding
# =====================================================================================

def seed_all(seed=28):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =====================================================================================
# SE3 / geometry helpers  (from Script 2, unchanged)
# =====================================================================================

def get_local_point_cube_fast(pcd, grasps, cube_size=0.14, max_pts=40, offset=0.06):
    B, N, _ = pcd.shape
    _, M, _, _ = grasps.shape
    device = pcd.device

    R = grasps[..., :3, :3]
    t = grasps[..., :3, 3]

    approach = R[..., 1]
    center = t + offset * approach

    diff = pcd.unsqueeze(1) - center.unsqueeze(2)
    pts_local = torch.matmul(diff, R)

    half = cube_size / 2
    mask = (pts_local.abs() <= half).all(-1)

    mask_f = mask.reshape(-1, N)
    BM = mask_f.shape[0]

    base = (torch.arange(B, device=device) * N).repeat_interleave(M)

    rand_scores = torch.rand(BM, N, device=device)
    rand_scores = rand_scores.masked_fill(~mask_f, -1e6)

    topk_idx = torch.topk(rand_scores, max_pts, dim=1).indices

    valid_counts = mask_f.sum(dim=1)
    empty = valid_counts == 0

    if empty.any():
        rand_idx = torch.randint(0, N, (empty.sum(), max_pts), device=device)
        topk_idx[empty] = rand_idx

    few = (valid_counts > 0) & (valid_counts < max_pts)

    if few.any():
        few_idx = torch.where(few)[0]
        valid_lists = mask_f[few]
        valid_positions = valid_lists.nonzero(as_tuple=False)
        col_ids = valid_positions[:, 1]
        counts = valid_counts[few]
        offsets = torch.cumsum(
            torch.cat([torch.zeros(1, device=device, dtype=torch.long), counts[:-1]]),
            dim=0
        )
        sampled = torch.zeros((len(few_idx), max_pts), device=device, dtype=torch.long)
        for i in range(len(few_idx)):
            start = offsets[i]
            end = start + counts[i]
            valid_cols = col_ids[start:end]
            rand_sel = valid_cols[torch.randint(0, len(valid_cols), (max_pts,), device=device)]
            sampled[i] = rand_sel
        topk_idx[few] = sampled

    pcd_f = pcd.reshape(B * N, 3)
    idx = topk_idx + base[:, None]
    out = pcd_f[idx]

    return out.reshape(B, M, max_pts, 3)


def se3_exp(xi):
    t = xi[..., :3]
    w = xi[..., 3:]

    theta = torch.norm(w, dim=-1, keepdim=True) + 1e-8
    k = w / theta

    K = torch.zeros((*w.shape[:-1], 3, 3), device=w.device)
    K[..., 0, 1] = -k[..., 2]
    K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]
    K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]
    K[..., 2, 1] =  k[..., 0]

    I = torch.eye(3, device=w.device).expand_as(K)
    R = I + torch.sin(theta)[..., None] * K + (1 - torch.cos(theta))[..., None] * (K @ K)

    T = torch.zeros((*xi.shape[:-1], 4, 4), device=xi.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0

    return T


def matrix_inv(T):
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    R_inv = R.transpose(-1, -2)
    t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)
    T_inv = torch.zeros_like(T)
    T_inv[..., :3, :3] = R_inv
    T_inv[..., :3, 3] = t_inv
    T_inv[..., 3, 3] = 1.0
    return T_inv


def refine_grasps_mppi(
    grasps,
    visual_context,
    model,
    control_points,
    inner_points,
    device,
    N_samples=64,
    N_iters=10,
    lambda_=0.7,
    chunk_size=512,
    w_free=1.0,
    w_contact=1.0,
    w_reg=0.05,
):
    if grasps.shape[0] == 0:
        return grasps

    grasp = grasps.unsqueeze(0)
    B, M, _, _ = grasp.shape
    S = N_samples
    MS = M * S

    mu = torch.zeros((B, M, 6), device=device)

    R_grasp = grasp[..., :3, :3]
    sigma_trans_local = torch.tensor([0.05, 0.05, 0.05], device=device)
    Sigma_trans_local = torch.diag(sigma_trans_local ** 2)
    Sigma_trans_world = R_grasp @ Sigma_trans_local @ R_grasp.transpose(-1, -2)

    sigma_rot_local = torch.tensor([0.1, 0.1, 0.1], device=device)
    Sigma_rot = torch.diag(sigma_rot_local ** 2).expand(B, M, 3, 3)

    Sigma = torch.zeros((B, M, 6, 6), device=device)
    Sigma[..., :3, :3] = Sigma_trans_world
    Sigma[..., 3:, 3:] = Sigma_rot

    eye = torch.eye(6, device=device).view(1, 1, 6, 6)

    for _ in range(N_iters):
        eps = torch.randn(B, M, S, 6, device=device)
        L = torch.linalg.cholesky(Sigma)
        xi_samples = mu.unsqueeze(2) + torch.matmul(eps, L.transpose(-1, -2))
        xi_flat = xi_samples.reshape(B, MS, 6)

        T_delta = se3_exp(xi_flat)
        grasp_rep = grasp.repeat_interleave(S, dim=1)
        T_new = torch.matmul(grasp_rep, T_delta)

        cost = torch.empty((B, MS), device=device)

        for start in range(0, MS, chunk_size):
            end = min(start + chunk_size, MS)
            T_chunk = T_new[:, start:end]
            Bc, Mc, _, _ = T_chunk.shape
            T_chunk_f = T_chunk.reshape(-1, 4, 4)
            inverse_grasps = matrix_inv(T_chunk_f)

            local_partial = get_local_point_cube_fast(
                visual_context.unsqueeze(0),
                T_chunk,
                cube_size=0.1 * 8,
                offset=0.05 * 8,
                max_pts=visual_context.shape[0] // 20
            )

            R_ = inverse_grasps[..., :3, :3]
            t_ = inverse_grasps[..., :3, 3]
            offset_axis = torch.zeros_like(R_[..., :, 1])
            offset_axis[..., 1] = 1
            t_ = t_ - 0.05 * 8.0 * offset_axis

            local_partial = torch.matmul(local_partial, R_.transpose(-1, -2))
            local_partial = local_partial + t_.unsqueeze(1)
            local_partial *= 10.0

            P = local_partial.shape[-2]
            local_partial = local_partial.reshape(Bc * Mc, P, 3)

            with torch.no_grad():
                model.set_local_latent(local_partial, Mc)
                cp = control_points.unsqueeze(0).expand(Bc * Mc, -1, -1)
                ip = inner_points.unsqueeze(0).expand(Bc * Mc, -1, -1)
                query_pts = torch.cat([cp, ip], dim=1)
                occ = torch.sigmoid(model.compute_local_occ(query_pts))
                occ = occ.reshape(Bc * Mc, query_pts.shape[1])
                occ_outer = occ[:, :cp.shape[1]]
                occ_inner = occ[:, cp.shape[1]:]

            occ_outer = occ_outer.reshape(Bc, Mc, -1)
            occ_inner = occ_inner.reshape(Bc, Mc, -1)

            L_free = occ_outer.mean(dim=-1)
            topk_inside = torch.topk(occ_inner, 10, dim=-1).values
            kth_best = topk_inside[..., -1]
            L_contact = 1.0 - kth_best

            xi_chunk = xi_flat[:, start:end]
            L_reg = xi_chunk.pow(2).sum(-1)

            cost[:, start:end] = w_free * L_free + w_contact * L_contact + w_reg * L_reg
            model.local_z = None

        cost_reshaped = cost.view(B, M, S)
        cost_min = cost_reshaped.min(dim=2, keepdim=True).values
        weights = torch.exp(-(cost_reshaped - cost_min) / lambda_)
        weights = weights / (weights.sum(dim=2, keepdim=True) + 1e-8)

        mu = (weights.unsqueeze(-1) * xi_samples).sum(dim=2)
        diff = xi_samples - mu.unsqueeze(2)
        Sigma = torch.einsum('bmsi,bmsj->bmij', weights.unsqueeze(-1) * diff, diff) + 1e-5 * eye
        Sigma = 0.9 * Sigma

    refined_grasps = (grasp @ se3_exp(mu)).squeeze(0)
    return refined_grasps


def final_grasp_check(
    grasps,
    visual_context,
    model,
    control_points,
    inner_points,
    device,
    chunk_size=512,
):
    if grasps.shape[0] == 0:
        return torch.zeros(0, dtype=torch.bool, device=device)

    M = grasps.shape[0]
    good_mask = torch.zeros(M, dtype=torch.bool, device=device)

    for start in range(0, M, chunk_size):
        end = min(start + chunk_size, M)
        T_chunk = grasps[start:end]
        Mc = T_chunk.shape[0]
        inverse_grasps = matrix_inv(T_chunk)

        local_partial = get_local_point_cube_fast(
            visual_context.unsqueeze(0),
            T_chunk.unsqueeze(0),
            cube_size=0.1 * 8,
            offset=0.05 * 8,
            max_pts=visual_context.shape[0] // 20
        )
        local_partial = local_partial.squeeze(0)

        R_ = inverse_grasps[..., :3, :3]
        t_ = inverse_grasps[..., :3, 3]
        offset_axis = torch.zeros_like(R_[..., :, 1])
        offset_axis[..., 1] = 1
        t_ = t_ - 0.05 * 8.0 * offset_axis

        local_partial = torch.matmul(local_partial, R_.transpose(-1, -2))
        local_partial = local_partial + t_.unsqueeze(1)
        local_partial *= 10.0

        with torch.no_grad():
            model.set_local_latent(local_partial, Mc)
            cp = control_points.unsqueeze(0).expand(Mc, -1, -1)
            ip = inner_points.unsqueeze(0).expand(Mc, -1, -1)
            query_pts = torch.cat([cp, ip], dim=1)
            occ = torch.sigmoid(model.compute_local_occ(query_pts))
            occ = occ.reshape(Mc, query_pts.shape[1])
            occ_outer = occ[:, :cp.shape[1]]
            occ_inner = occ[:, cp.shape[1]:]

        outer_max = occ_outer.max(dim=-1).values
        inner_max = occ_inner.max(dim=-1).values
        valid = (outer_max < 0.2) & (inner_max > 0.7)

        good_mask[start:end] = valid
        model.local_z = None

    return good_mask


# =====================================================================================
# Startup: load model and fixed assets once, then block until requests arrive
# =====================================================================================

seed_all(2882)

spec_file = './configs/'
args = load_experiment_specifications(spec_file, load_yaml='dual_arm_params_combined')

device = 'cuda:0'

with open('./scales.json') as f:
    scales = json.load(f)

args['device'] = device

print("[INFO] Loading grasp model...")
model = loader.load_model(args)
model.eval()
model = model.to(device)
print("[INFO] Model loaded and ready")

control_points = (
    torch.tensor(np.load('./se3dif/models/points/new_gripper_points.npy'))
    .to(device)
    .float()
    * 5.0 / 0.56
)
offset_axis_cp = torch.zeros_like(control_points)
offset_axis_cp[:, 1] = 1
control_points = control_points - (0.05 * 8.0 * 5.0 / 0.56) * offset_axis_cp

inner_points = (
    torch.tensor(np.load('./se3dif/models/points/new_gripper_points_inner.npy'))
    .to(device)
    .float()
    * 5.0 / 0.56
)
offset_axis_ip = torch.zeros_like(inner_points)
offset_axis_ip[:, 1] = 1
inner_points = inner_points - (0.05 * 8.0 * 5.0 / 0.56) * offset_axis_ip

# MPPI hyper-parameters (matching Script 2)
N_SAMPLES = 12
N_ITERS   = 4
W_FREE    = 1.0
W_CONTACT = 1.0
W_REG     = 0.05

app = Flask(__name__)


# =====================================================================================
# Routes
# =====================================================================================

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # ------------------------------------------------------------------
        # 1. Validate and deserialise incoming pointcloud
        # ------------------------------------------------------------------
        if "data" not in request.files:
            return jsonify({"error": "Missing 'data' file in multipart request"}), 400

        raw = request.files["data"].read()

        try:
            buf = io.BytesIO(raw)
            npz = np.load(buf)
            points_np = npz["points"]           # (N, 3) float32
        except Exception as e:
            return jsonify({"error": f"Bad NPZ payload: {e}"}), 400

        if points_np.ndim != 2 or points_np.shape[1] != 3:
            return jsonify({"error": f"Expected (N,3) array, got {points_np.shape}"}), 400

        if points_np.dtype != np.float32:
            points_np = points_np.astype(np.float32)

        visual_context = torch.tensor(points_np, dtype=torch.float32, device=device)  # (N,3)

        # ------------------------------------------------------------------
        # 2. Single-grasp inference  (Script 1 logic)
        # ------------------------------------------------------------------
        with torch.no_grad():
            model.set_latent(visual_context.unsqueeze(0))   # expects (1, N, 3)

        num_grasps_required = 2048

        loc = torch.stack(
            [torch.eye(4, device=device) for _ in range(num_grasps_required)]
        )
        loc[..., :3, 3] = visual_context
        loc[..., 3, 3] = 1.0

        with torch.no_grad():
            approach_dir, base_dir, width = model.grasp_forward_partial_scalar(
                loc.unsqueeze(0)
            )

        base_dir = F.normalize(base_dir, p=2, dim=-1)
        dot_product = torch.sum(base_dir * approach_dir, dim=-1, keepdim=True)
        projection = dot_product * base_dir
        approach_dir = F.normalize(approach_dir - projection, p=2, dim=-1)

        pred_grasps = params_to_grasp_width(
            (approach_dir, base_dir, width, visual_context), depth=0.52
        )  # (1, N, 4, 4)

        graspable_scores = model.graspable_region_scores.clone().reshape(-1)

        idx_to_keep      = graspable_scores > 0.8
        pred_grasps      = pred_grasps[:, idx_to_keep]
        graspable_scores = graspable_scores[idx_to_keep]
        contact_points   = visual_context[idx_to_keep]      # (K, 3)

        sel_grasps  = pred_grasps.reshape(-1, 4, 4)         # (K, 4, 4)
        sel_scores  = graspable_scores                       # (K,)
        sel_context = contact_points                         # (K, 3)

        if sel_grasps.shape[0] < 2:
            return jsonify(
                {"error": "Not enough single grasps after graspability filtering"}
            ), 422

        # ------------------------------------------------------------------
        # 3. Grasp pairing + refinement  (Script 2 logic)
        # ------------------------------------------------------------------
        grasps_to_take     = sel_grasps
        graspable_scores_2 = sel_scores
        cp_to_take         = sel_context

        if grasps_to_take.shape[0] > 200:
            idx_to_take = farthest_point_sampling_seeded(
                cp_to_take,
                k=200,
                pcd_seeds=cp_to_take[torch.argmax(graspable_scores_2)].unsqueeze(0)
            )
            grasps_to_take     = grasps_to_take[idx_to_take]
            cp_to_take         = cp_to_take[idx_to_take]
            graspable_scores_2 = graspable_scores_2[idx_to_take]

        pair_indices = grasp_pair_combinations(grasps=grasps_to_take, min_dist=0.5)

        grasp_pairs = torch.stack([
            grasps_to_take[pair_indices[:, 0]],
            grasps_to_take[pair_indices[:, 1]]
        ], dim=1)

        cp_pairs = torch.stack([
            cp_to_take[pair_indices[:, 0]],
            cp_to_take[pair_indices[:, 1]]
        ], dim=1)

        shuffle_idx  = torch.randperm(grasp_pairs.shape[0])
        grasp_pairs  = grasp_pairs[shuffle_idx]
        cp_pairs     = cp_pairs[shuffle_idx]
        pair_indices = pair_indices[shuffle_idx]

        pair_single_min = torch.minimum(
            graspable_scores_2[pair_indices[:, 0]],
            graspable_scores_2[pair_indices[:, 1]]
        )

        if grasp_pairs.shape[0] < 2:
            return jsonify({"error": "Not enough grasp pairs formed"}), 422

        # Dual classification
        with torch.no_grad():
            model.set_latent(visual_context.unsqueeze(0))

            H = grasp_pairs.clone()
            H[..., :3, 3] = cp_pairs
            pair_scores = model.grasp_forward_dual_classification(
                pcd=None,
                grasps=H.unsqueeze(0).to(device)
            )

        pair_scores = pair_scores.reshape(-1)
        sorted_idx  = torch.argsort(pair_scores, descending=True)

        pair_scores     = pair_scores[sorted_idx]
        pair_single_min = pair_single_min[sorted_idx]
        grasp_pairs     = grasp_pairs[sorted_idx]
        pair_indices    = pair_indices[sorted_idx]
        cp_pairs        = cp_pairs[sorted_idx]

        # Bucket selection (top-100, epsilon-greedy by dual score then single score)
        epsilon = 0.03
        topk    = 100

        selected_pairs        = []
        selected_scores       = []
        selected_pair_indices = []
        current_bucket_idx    = []
        current_bucket_single = []
        current_max           = None

        for i in range(pair_scores.shape[0]):
            ps = pair_scores[i]

            if current_max is None:
                current_max           = ps
                current_bucket_idx    = [i]
                current_bucket_single = [pair_single_min[i]]

            elif current_max - ps <= epsilon:
                current_bucket_idx.append(i)
                current_bucket_single.append(pair_single_min[i])

            else:
                bucket_single = torch.tensor(current_bucket_single, device=pair_scores.device)
                order = torch.argsort(bucket_single, descending=True)
                for o in order:
                    idx = current_bucket_idx[o]
                    selected_pairs.append(grasp_pairs[idx])
                    selected_scores.append(pair_scores[idx])
                    selected_pair_indices.append(pair_indices[idx])
                    if len(selected_pairs) >= topk:
                        break
                if len(selected_pairs) >= topk:
                    break
                current_max           = ps
                current_bucket_idx    = [i]
                current_bucket_single = [pair_single_min[i]]

        if len(selected_pairs) < topk and current_bucket_idx:
            bucket_single = torch.tensor(current_bucket_single, device=pair_scores.device)
            order = torch.argsort(bucket_single, descending=True)
            for o in order:
                idx = current_bucket_idx[o]
                selected_pairs.append(grasp_pairs[idx])
                selected_scores.append(pair_scores[idx])
                selected_pair_indices.append(pair_indices[idx])
                if len(selected_pairs) >= topk:
                    break

        grasp_pairs           = torch.stack(selected_pairs)
        selected_pair_indices = torch.stack(selected_pair_indices)

        # Unique-grasp MPPI refinement
        unique_grasps, inverse = torch.unique(
            grasp_pairs.reshape(-1, 4, 4),
            dim=0,
            return_inverse=True
        )
        pair_to_grasp_idx = inverse.reshape(-1, 2)

        refined_grasps = refine_grasps_mppi(
            unique_grasps,
            visual_context,
            model,
            control_points,
            inner_points,
            device,
            N_samples=N_SAMPLES,
            N_iters=N_ITERS,
            lambda_=0.7,
            w_free=W_FREE,
            w_contact=W_CONTACT,
            w_reg=W_REG,
        )

        good_mask = final_grasp_check(
            refined_grasps,
            visual_context,
            model,
            control_points,
            inner_points,
            device,
        )

        refined_pruned_grasps = refined_grasps[good_mask]

        unique_to_pruned = -torch.ones(
            unique_grasps.shape[0], dtype=torch.long, device=device
        )
        unique_to_pruned[good_mask] = torch.arange(
            refined_pruned_grasps.shape[0], device=device
        )

        pruned_pair_indices = unique_to_pruned[pair_to_grasp_idx]
        valid_pair_mask     = (pruned_pair_indices != -1).all(dim=1)
        pruned_pair_indices = pruned_pair_indices[valid_pair_mask]
        refined_grasp_pairs = refined_pruned_grasps[pruned_pair_indices]

        torch.cuda.empty_cache()

        # ------------------------------------------------------------------
        # 4. Serialise and return as compressed NPZ
        # ------------------------------------------------------------------
        pairs_to_send = refined_grasp_pairs.detach().cpu().numpy().astype(np.float32)
        score_to_send = torch.tensor(selected_scores, device=device)[valid_pair_mask].cpu().numpy().astype(np.float32)
        out_npz = {
            "refined_grasp_pairs": pairs_to_send,  # (P, 2, 4, 4)
            "pair_scores": score_to_send,           # (P,)
        }   

        out_buf = io.BytesIO()
        np.savez_compressed(out_buf, **out_npz)
        out_buf.seek(0)

        return Response(
            out_buf.read(),
            mimetype="application/octet-stream",
        )

    except Exception as e:
        traceback.print_exc()
        return jsonify(
            {
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
        ), 500


# =====================================================================================
# Main
# =====================================================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        threaded=False,   # single-threaded: one GPU inference at a time
    )
