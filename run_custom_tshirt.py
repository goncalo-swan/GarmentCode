"""
Custom T-Shirt Generation & Simulation Pipeline
Uses the BUILT-IN GarmentCode parametric system (MetaGarment + Shirt + Sleeve)
which handles armhole/sleeve cap edge matching automatically.

Production data: CALITEE_International Production (sizes 48, 50, 52)
Body: GLOBAL MEN size 50
Simulation body mesh: global_men_size50_apose (custom SMPL, 181cm)

Measurements from techpack:
  - Bust (chest circumference): from "Chest width (meas. 2cm below armhole)"
  - Arm Length: from "SHORT sleeve - Shoulder and sleeve length"
  - Bicep: from "Sleeve width (meas. 2cm below armhole)"
  - Back Length (Nape to Waist): from "Back length"
"""
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml

import pygarment as pyg
from pygarment.data_config import Properties
from production_to_design import ProductionToDesign


# ============================================================
# Production measurements per garment size (full circumference, cm)
# Source: 002 : CALITEE_International Production techpack
# ============================================================
GARMENT_SIZES = [48, 50, 52]

PRODUCTION_DATA = {
    48: {
        'Bicep': 43.6,
        'Arm_Length': 36.6,
        'Bust': 101,
        'Nape_to_Waist': 71,
        'Sleeve_Opening': 34.1,  # M_cuff: cuff opening circumference
        'Collar_Width': 16.7,    # N: neck opening width
        'Front_Neck_Drop': 10.7, # O1: front neckline depth from HPS
        'Back_Neck_Drop': 1.9,   # O2: back neckline depth from HPS
    },
    50: {
        'Bicep': 44.8,
        'Arm_Length': 37.3,
        'Bust': 105,
        'Nape_to_Waist': 72,
        'Sleeve_Opening': 35.2,
        'Collar_Width': 17.0,
        'Front_Neck_Drop': 11.0,
        'Back_Neck_Drop': 2.1,
    },
    52: {
        'Bicep': 46.0,
        'Arm_Length': 37.9,
        'Bust': 109,
        'Nape_to_Waist': 73,
        'Sleeve_Opening': 36.3,
        'Collar_Width': 17.2,
        'Front_Neck_Drop': 11.4,
        'Back_Neck_Drop': 2.2,
    },
}

# Body parameters for GLOBAL MEN size 50
# Simulation runs on A-pose body (works correctly with sleeves).
# The result is then reposed to the custom pose using SMPL vertex correspondence.
APOSE_BODY_YAML = './assets/bodies/global_men_size50_apose.yaml'
APOSE_BODY_NAME = 'global_men_size50_apose'
CUSTOM_BODY_NAME = 'global_men_size50_custom_pose'


def map_production_to_design(prod, body_yaml_path):
    """Map production measurements to GarmentCode design parameters."""
    mapper = ProductionToDesign(body_yaml_path)
    body = mapper.body

    # Convert production measurements to what the mapper expects
    bust_circ = prod['Bust']
    back_length = prod['Nape_to_Waist']  # HPS to hem

    garment_measurements = {
        'bust_circumference': bust_circ,
        'length': back_length,
        'hps_to_cuff': prod['Arm_Length'],  # Full HPS-to-cuff (I)
    }

    # Collar: use actual production values when available, fall back to ratios
    if 'Front_Neck_Drop' in prod:
        garment_measurements['collar_depth_front'] = prod['Front_Neck_Drop']
    else:
        garment_measurements['collar_depth_front'] = back_length * 0.153

    if 'Back_Neck_Drop' in prod:
        garment_measurements['collar_depth_back'] = prod['Back_Neck_Drop']
    else:
        garment_measurements['collar_depth_back'] = back_length * 0.029

    if 'Collar_Width' in prod:
        garment_measurements['collar_width'] = prod['Collar_Width']

    # Sleeve cuff: pass absolute circumference, mapper will use actual arm_width
    if 'Sleeve_Opening' in prod:
        garment_measurements['sleeve_opening_circ'] = prod['Sleeve_Opening']

    design = mapper.map_shirt(garment_measurements)
    return design


def generate_pattern(size, design, body_yaml_path, output_base):
    """Generate pattern using built-in MetaGarment system."""
    from assets.garment_programs.meta_garment import MetaGarment
    from assets.bodies.body_params import BodyParameters

    garment_name = f'calitee_tshirt_size{size}'

    body = BodyParameters(body_yaml_path)
    garment = MetaGarment(garment_name, body, design)
    pattern = garment.assembly()

    if garment.is_self_intersecting():
        print(f'  WARNING: {garment_name} has self-intersecting panels')

    folder = pattern.serialize(
        output_base,
        tag='_' + datetime.now().strftime("%y%m%d-%H-%M-%S"),
        to_subfolder=True,
        with_3d=False,
        with_text=False,
        view_ids=False,
        with_printable=True
    )

    print(f'  Pattern generated: {garment_name} -> {folder}')
    return Path(folder), garment_name


def simulate_pattern(pattern_folder, garment_name, output_base):
    """Run physics simulation on A-pose body."""
    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.simulation import run_sim
    from pygarment.meshgen.sim_config import PathCofig

    sim_config_path = './assets/Sim_props/tshirt_sim_props.yaml'
    props = Properties(sim_config_path)
    props.set_section_stats(
        'sim', fails={}, sim_time={}, spf={},
        fin_frame={}, body_collisions={}, self_collisions={}
    )
    props.set_section_stats('render', render_time={})

    spec_files = list(pattern_folder.glob('*_specification.json'))
    if not spec_files:
        print(f'  ERROR: No specification file found in {pattern_folder}')
        return None
    spec_file = spec_files[0]
    in_name = spec_file.stem.replace('_specification', '')

    paths = PathCofig(
        in_element_path=pattern_folder,
        out_path=output_base,
        in_name=in_name,
        body_name=APOSE_BODY_NAME,
        smpl_body=True,
        add_timestamp=True
    )

    print(f'  Generating box mesh for {in_name}...')
    resolution_scale = props['sim']['config']['resolution_scale']
    garment_box_mesh = BoxMesh(paths.in_g_spec, resolution_scale)
    garment_box_mesh.load()
    garment_box_mesh.serialize(
        paths, store_panels=False,
        uv_config=props['render']['config']['uv_texture']
    )

    props.serialize(paths.element_sim_props)

    print(f'  Running simulation for {in_name}...')
    run_sim(
        garment_box_mesh.name,
        props,
        paths,
        save_v_norms=False,
        store_usd=False,
        optimize_storage=False,
        verbose=False
    )

    props.serialize(paths.element_sim_props)
    print(f'  Simulation complete: {in_name} -> {paths.out_el}')
    return paths.out_el


def _load_body_cm(body_obj_path):
    """Load a body OBJ, convert to cm, shift feet to y=0."""
    import igl
    verts, faces = igl.read_triangle_mesh(str(body_obj_path))
    # Convert meters to cm if needed
    if verts.max() < 3.0:
        verts = verts * 100.0
    # Shift so feet are at y=0
    min_y = verts[:, 1].min()
    if min_y < 0:
        verts[:, 1] += abs(min_y)
    return verts, faces


def repose_garment(sim_folder,
                   apose_obj='./assets/bodies/global_men_size50_apose.obj',
                   custom_obj='./assets/bodies/global_men_size50_custom_pose.obj'):
    """Repose A-pose simulated garment to custom pose via barycentric projection.

    Both body OBJs are the same SMPL mesh (6890 verts, same topology) in different
    poses.  For each garment vertex we find the closest point on the A-pose body
    surface, compute barycentric coordinates within that triangle, then evaluate
    the same barycentric coords on the corresponding custom-pose triangle to get
    a smooth displacement field.
    """
    import igl

    sim_folder = Path(sim_folder)
    sim_files = list(sim_folder.glob('*_sim.obj'))
    if not sim_files:
        print(f'  No *_sim.obj found in {sim_folder}')
        return None
    sim_path = sim_files[0]

    # Load garment vertices (already in cm, y-shifted by the simulation)
    g_verts, g_faces = igl.read_triangle_mesh(str(sim_path))

    # Load both body meshes (same SMPL topology, different poses)
    a_verts, a_faces = _load_body_cm(apose_obj)
    c_verts, c_faces = _load_body_cm(custom_obj)

    assert len(a_verts) == len(c_verts), \
        f'Body vertex count mismatch: {len(a_verts)} vs {len(c_verts)}'
    assert np.array_equal(a_faces, c_faces), \
        'Body face topology mismatch between A-pose and custom pose'

    # For each garment vertex, find closest point on A-pose body surface.
    # Returns: squared distances, face indices, closest points
    sq_dists, face_ids, closest_pts = igl.point_mesh_squared_distance(
        g_verts.astype(np.float64),
        a_verts.astype(np.float64),
        a_faces.astype(np.int32)
    )

    # Compute barycentric coordinates of each closest point within its triangle
    tri_verts = a_verts[a_faces[face_ids]]  # (N_garment, 3_corners, 3_xyz)
    v0 = tri_verts[:, 0]
    v1 = tri_verts[:, 1]
    v2 = tri_verts[:, 2]

    # Barycentric coords via the standard area method
    e0 = v1 - v0   # (N, 3)
    e1 = v2 - v0   # (N, 3)
    ep = closest_pts - v0  # (N, 3)

    d00 = np.sum(e0 * e0, axis=1)
    d01 = np.sum(e0 * e1, axis=1)
    d11 = np.sum(e1 * e1, axis=1)
    dp0 = np.sum(ep * e0, axis=1)
    dp1 = np.sum(ep * e1, axis=1)

    denom = d00 * d11 - d01 * d01
    denom = np.maximum(denom, 1e-12)  # avoid division by zero for degenerate triangles

    bary_u = (d11 * dp0 - d01 * dp1) / denom  # weight for v1
    bary_v = (d00 * dp1 - d01 * dp0) / denom  # weight for v2
    bary_w = 1.0 - bary_u - bary_v             # weight for v0

    # Evaluate the same barycentric coords on the custom-pose triangles
    c_tri_verts = c_verts[a_faces[face_ids]]  # same face topology
    custom_pts = (bary_w[:, None] * c_tri_verts[:, 0] +
                  bary_u[:, None] * c_tri_verts[:, 1] +
                  bary_v[:, None] * c_tri_verts[:, 2])

    # Displacement = how the closest surface point moved from A-pose to custom pose
    garment_disp = custom_pts - closest_pts
    g_verts_reposed = g_verts + garment_disp

    # Write reposed OBJ by replacing vertex positions (preserves UVs, materials, etc.)
    reposed_path = sim_folder / sim_path.name.replace('_sim.obj', '_reposed.obj')
    with open(sim_path, 'r') as f:
        lines = f.readlines()

    v_idx = 0
    with open(reposed_path, 'w') as f:
        for line in lines:
            if line.startswith('v ') and not line.startswith('vt') and not line.startswith('vn'):
                v = g_verts_reposed[v_idx]
                f.write(f'v {v[0]} {v[1]} {v[2]}\n')
                v_idx += 1
            else:
                f.write(line)

    # Also overwrite the _sim.obj so the built-in renderer picks it up
    import shutil
    shutil.copy(reposed_path, sim_path)

    print(f'  Reposed garment saved to {reposed_path}')
    print(f'  Max displacement: {np.abs(garment_disp).max():.2f} cm')
    print(f'  Mean displacement: {np.linalg.norm(garment_disp, axis=1).mean():.2f} cm')
    return reposed_path


def render_reposed(sim_folder,
                   custom_obj='./assets/bodies/global_men_size50_custom_pose.obj'):
    """Re-render the simulation output using the custom-pose body and reposed garment."""
    from pygarment.meshgen.render.pythonrender import render_images
    from pygarment.meshgen.sim_config import PathCofig

    sim_folder = Path(sim_folder)
    sim_config_path = './assets/Sim_props/tshirt_sim_props.yaml'
    props = Properties(sim_config_path)
    render_props = props['render']['config']

    # Load custom-pose body (same transforms as simulation: scale to cm, y-shift)
    body_v, body_f = _load_body_cm(custom_obj)

    # Build a minimal PathCofig-like object with the paths the renderer needs
    # The renderer uses paths.g_sim and paths.render_path(side)
    spec_files = list(sim_folder.glob('*_specification.json'))
    sim_files = list(sim_folder.glob('*_sim.obj'))
    if not sim_files:
        print(f'  No *_sim.obj in {sim_folder}')
        return

    # Create a simple namespace to satisfy paths.g_sim and paths.render_path()
    class RenderPaths:
        def __init__(self, sim_path, out_dir, sim_tag):
            self.g_sim = sim_path
            self.out_el = out_dir
            self.sim_tag = sim_tag
        def render_path(self, camera_name=''):
            fname = f'{self.sim_tag}_render_{camera_name}.png' if camera_name else f'{self.sim_tag}_render.png'
            return self.out_el / fname

    sim_path = sim_files[0]
    sim_tag = sim_path.stem.replace('_sim', '')
    paths = RenderPaths(sim_path, sim_folder, sim_tag)

    render_images(paths, body_v, body_f, render_props)
    print(f'  Renders saved to {sim_folder}')


def save_combined_mesh(sim_folder,
                       body_obj_path=f'./assets/bodies/{CUSTOM_BODY_NAME}.obj',
                       garment_suffix='_reposed.obj'):
    """Create combined body + garment mesh."""
    import trimesh

    sim_folder = Path(sim_folder)

    # Use reposed garment if available, fall back to sim
    garment_files = list(sim_folder.glob(f'*{garment_suffix}'))
    if not garment_files:
        garment_files = list(sim_folder.glob('*_sim.obj'))
    if not garment_files:
        print(f'  No garment mesh found in {sim_folder}')
        return

    garment_path = garment_files[0]
    garment = trimesh.load(str(garment_path), process=False)
    body = trimesh.load(str(body_obj_path), process=False)

    # Convert body to cm if in meters
    if body.vertices.max() < 3.0:
        body.vertices = body.vertices * 100.0

    # Apply y-shift to match simulation
    min_y = body.vertices[:, 1].min()
    if min_y < 0:
        body.vertices[:, 1] += abs(min_y)

    # Save combined OBJ
    body_v = np.array(body.vertices)
    body_f = np.array(body.faces)
    garm_v = np.array(garment.vertices)
    garm_f = np.array(garment.faces)
    garm_f_offset = garm_f + len(body_v)

    output_path = sim_folder / 'combined.obj'
    with open(output_path, 'w') as f:
        f.write("# Combined body + garment mesh\n\n")
        for v in body_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in garm_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\ng body\n")
        for face in body_f:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        f.write("\ng garment\n")
        for face in garm_f_offset:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f'  Combined mesh saved to {output_path}')

    # Also save GLB
    body.visual = trimesh.visual.ColorVisuals(
        mesh=body, face_colors=np.tile([80, 80, 80, 255], (len(body.faces), 1)))
    garment.visual = trimesh.visual.ColorVisuals(
        mesh=garment, face_colors=np.tile([220, 220, 220, 255], (len(garment.faces), 1)))
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name='body')
    scene.add_geometry(garment, node_name='garment')
    glb_path = sim_folder / 'combined.glb'
    scene.export(str(glb_path))
    print(f'  Combined GLB saved to {glb_path}')


# ============================================================
# Main pipeline
# ============================================================

if __name__ == '__main__':
    sys_props = Properties('./system.json')
    output_base = sys_props['output']

    print("=" * 60)
    print("  Custom T-Shirt Pipeline (Built-in Parametric System)")
    print("  Production: CALITEE_International Production")
    print("  Body: GLOBAL MEN size 50")
    print("  Garment sizes: 48, 50, 52")
    print(f"  Simulation body: {APOSE_BODY_NAME} (A-pose)")
    print(f"  Target pose body: {CUSTOM_BODY_NAME}")
    print("=" * 60)

    # Step 1: Generate patterns for all sizes using built-in system
    print("\n--- Step 1: Generating patterns ---")
    generated = []
    for size in GARMENT_SIZES:
        prod = PRODUCTION_DATA[size]
        print(f'\nSize {size}: Bust={prod["Bust"]}, Arm_Length={prod["Arm_Length"]}, '
              f'Nape_to_Waist={prod["Nape_to_Waist"]}')
        design = map_production_to_design(prod, APOSE_BODY_YAML)
        print(f'  Design: width={design["shirt"]["width"]["v"]:.3f}, '
              f'length={design["shirt"]["length"]["v"]:.3f}, '
              f'sleeve_length={design["sleeve"]["length"]["v"]:.3f}')
        folder, name = generate_pattern(size, design, APOSE_BODY_YAML, output_base)
        generated.append((folder, name, size))

    # Step 2: Simulate each pattern on A-pose body
    print("\n--- Step 2: Running simulations (A-pose) ---")
    sim_results = []
    for folder, name, size in generated:
        print(f'\nSimulating size {size}...')
        try:
            sim_folder = simulate_pattern(folder, name, output_base)
            if sim_folder:
                sim_results.append((sim_folder, size))
        except Exception as e:
            print(f'  Simulation failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    # Step 3: Repose garments from A-pose to custom pose and re-render
    print("\n--- Step 3: Reposing garments to custom pose ---")
    for sim_folder, size in sim_results:
        print(f'\nReposing size {size}...')
        try:
            repose_garment(sim_folder)
            render_reposed(sim_folder)
        except Exception as e:
            print(f'  Reposing/render failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    # Step 4: Create combined meshes
    print("\n--- Step 4: Creating combined body+garment meshes ---")
    for sim_folder, size in sim_results:
        print(f'\nCombining size {size}...')
        try:
            save_combined_mesh(sim_folder)
        except Exception as e:
            print(f'  Combined mesh failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print("\nGenerated patterns:")
    for folder, name, size in generated:
        print(f"  Size {size}: {folder}")
    print("\nSimulation results:")
    for sim_folder, size in sim_results:
        print(f"  Size {size}: {sim_folder}")
    print(f"\nAll output in: {output_base}")
