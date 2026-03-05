"""
Custom Pants Generation & Simulation Pipeline
Uses the BUILT-IN GarmentCode parametric system (MetaGarment + Pants + StraightWB).

Production data: CALITEE Heritage Pants (sizes 48, 50, 52)
Body: GLOBAL MEN size 50
Simulation body mesh: global_men_size50_apose (custom SMPL, 181cm)

Measurements from techpack:
  - Waist: "Waistband width relaxed"
  - Low Hip: "Hip width"
  - Inseam: "Inseam"
  - Thigh: "Thigh width"
  - Knee: "Knee width"
  - Calf: "Calf width (meas. 46 cm from crotch along inseam)"
  - Ankle: "OPEN hem - Leg opening meas. straight"
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from pathlib import Path
from datetime import datetime
import json

import numpy as np
import yaml
import svgpathtools as svgpath

import pygarment as pyg
from pygarment.data_config import Properties
from production_to_design import ProductionToDesign


# ============================================================
# Production measurements per garment size (full circumference, cm)
# Source: 45674 R-GMH4206F23 - REGULAR-MEN'S HERITAGE PANTS
# ============================================================
GARMENT_SIZES = [48, 50, 52]

PRODUCTION_DATA = {
    48: {
        'Waist': 76.5,
        'Low_Hip': 107,
        'Inseam': 77.4,
        'Thigh': 71.1,
        'Knee': 52.8,
        'Calf': 50.3,
        'Ankle': 43.2,
    },
    50: {
        'Waist': 80,
        'Low_Hip': 111,
        'Inseam': 77.5,
        'Thigh': 73,
        'Knee': 54,
        'Calf': 51.5,
        'Ankle': 44,
    },
    52: {
        'Waist': 83.5,
        'Low_Hip': 115,
        'Inseam': 77.6,
        'Thigh': 75.3,
        'Knee': 55.2,
        'Calf': 52.7,
        'Ankle': 44.8,
    },
}

# Body parameters for GLOBAL MEN size 50
BODY_YAML = './assets/bodies/global_men_size50_apose.yaml'
BODY_NAME = 'global_men_size50_apose'


def map_production_to_design(prod, body_yaml_path):
    """Map production measurements to GarmentCode design parameters."""
    mapper = ProductionToDesign(body_yaml_path)
    body = mapper.body

    # Panel length ≈ inseam + crotch_hip_diff
    # (the panel extends from hem to waist; the inseam is crotch-to-hem,
    #  and crotch_hip_diff is the vertical drop from hipline to crotch)
    panel_length = prod['Inseam'] + body['crotch_hip_diff']

    garment_measurements = {
        'waist_circumference': prod['Waist'],
        'hip_circumference': prod['Low_Hip'],
        'length': panel_length,
        'leg_opening': prod['Ankle'],
    }

    design = mapper.map_pants(garment_measurements)
    return design


def generate_pattern(size, design, body_yaml_path, output_base):
    """Generate pattern using built-in MetaGarment system."""
    from assets.garment_programs.meta_garment import MetaGarment
    from assets.bodies.body_params import BodyParameters

    garment_name = f'calitee_pants_size{size}'

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

    sim_config_path = './assets/Sim_props/pants_sim_props.yaml'
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
        body_name=BODY_NAME,
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


# ============================================================
# Measurement verification from spec JSON
# ============================================================

def _rel_to_abs(start, end, rel_pt):
    """Convert relative Bezier control point to absolute 2D coordinates."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    edge = end - start
    edge_perp = np.array([-edge[1], edge[0]])
    return start + rel_pt[0] * edge + rel_pt[1] * edge_perp


def edge_length(edge_dict, verts):
    """Return the arc-length of a single edge from a panel JSON dict."""
    ep = edge_dict['endpoints']
    v1 = verts[ep[0]]
    v2 = verts[ep[1]]

    if 'curvature' not in edge_dict:
        return float(np.linalg.norm(np.array(v2) - np.array(v1)))

    curv = edge_dict['curvature']
    ctype = curv['type']

    if ctype == 'circle':
        radius, large_arc, sweep = curv['params']
        arc = svgpath.Arc(
            complex(v1[0], v1[1]),
            complex(radius, radius),
            0, int(large_arc), int(sweep),
            complex(v2[0], v2[1]),
        )
        return abs(arc.length())
    elif ctype == 'cubic':
        cp1 = _rel_to_abs(v1, v2, curv['params'][0])
        cp2 = _rel_to_abs(v1, v2, curv['params'][1])
        curve = svgpath.CubicBezier(
            complex(v1[0], v1[1]),
            complex(cp1[0], cp1[1]),
            complex(cp2[0], cp2[1]),
            complex(v2[0], v2[1]),
        )
        return abs(curve.length())
    elif ctype == 'quadratic':
        cp1 = _rel_to_abs(v1, v2, curv['params'][0])
        curve = svgpath.QuadraticBezier(
            complex(v1[0], v1[1]),
            complex(cp1[0], cp1[1]),
            complex(v2[0], v2[1]),
        )
        return abs(curve.length())
    else:
        return float(np.linalg.norm(np.array(v2) - np.array(v1)))


def _panel_edge_lengths(panel):
    """Return list of arc-lengths for every edge of a panel."""
    verts = panel['vertices']
    return [edge_length(e, verts) for e in panel['edges']]


def _free_edges(panels, stitches, panel_name):
    """Return indices and edge dicts of free (unstitched) edges."""
    stitched = set()
    for stitch in stitches:
        for side in stitch:
            stitched.add((side['panel'], side['edge']))
    return [
        (i, panels[panel_name]['edges'][i])
        for i in range(len(panels[panel_name]['edges']))
        if (panel_name, i) not in stitched
    ]


def verify_measurements(spec_path, size, prod):
    """Extract pants measurements from spec JSON and compare to production targets.

    Panel structure (from stitching analysis):
      pant_f_r: e0=bottom(hem), e1=inseam, e2=crotch_bottom, e3=crotch_top,
                e4=top(waist), e5=outside_top, e6=outside_bottom
      pant_b_r: e0=outside_bottom, e1=outside_top, e2..e9=top(with darts),
                e10=crotch_top, e11=inseam, e12=bottom(hem)
      wb_front: lower_interface on e3 (top edge)
      wb_back:  lower_interface on e1 (top edge)
    """
    with open(spec_path, 'r') as f:
        spec = json.load(f)

    panels = spec['pattern']['panels']
    stitches = spec['pattern']['stitches']

    # Build stitching map for edge identification
    stitched = set()
    stitch_map = {}  # (panel, edge) -> (other_panel, other_edge)
    for stitch in stitches:
        for side in stitch:
            stitched.add((side['panel'], side['edge']))
        if len(stitch) == 2:
            a, b = stitch
            stitch_map[(a['panel'], a['edge'])] = (b['panel'], b['edge'])
            stitch_map[(b['panel'], b['edge'])] = (a['panel'], a['edge'])

    results = {}

    # --- Waist circumference ---
    # Find edges with 'lower_interface' label on waistband panels
    waist_circ = 0.0
    for pname in ['wb_front', 'wb_back']:
        panel = panels.get(pname)
        if not panel:
            continue
        for e in panel['edges']:
            if e.get('label') == 'lower_interface':
                waist_circ += edge_length(e, panel['vertices'])
    results['Waist'] = waist_circ if waist_circ > 0 else None

    # --- Leg opening (ankle circumference) ---
    # Free (unstitched) edges on pant panels at y ≈ 0 (hem level)
    # For right leg: pant_f_r and pant_b_r free edges at bottom
    leg_opening = 0.0
    for pname in ['pant_f_r', 'pant_b_r']:
        panel = panels.get(pname)
        if not panel:
            continue
        verts = panel['vertices']
        for i, e in enumerate(panel['edges']):
            if (pname, i) in stitched:
                continue
            ep = e['endpoints']
            v1, v2 = verts[ep[0]], verts[ep[1]]
            # Bottom edge: both endpoints near y = min_y of panel
            min_y = min(v[1] for v in verts)
            if abs(v1[1] - min_y) < 1.0 and abs(v2[1] - min_y) < 1.0:
                leg_opening += edge_length(e, verts)
    # front_bottom + back_bottom = full circumference of one leg opening (no ×2)
    results['Ankle'] = leg_opening if leg_opening > 0 else None

    # --- Inseam ---
    # The inseam is the stitched edge between front and back panels on the inside.
    # Identified from stitching: pant_f_r e1 ↔ pant_b_r e11
    # The inseam length = length of either stitched edge (they match)
    inseam = 0.0
    pant_fr = panels.get('pant_f_r')
    pant_br = panels.get('pant_b_r')
    if pant_fr and pant_br:
        # Find the stitch that connects pant_f_r to pant_b_r on the inseam
        # The inseam edge on pant_f_r goes from the bottom (y≈0) upward (the inside leg)
        fr_verts = pant_fr['vertices']
        for i, e in enumerate(pant_fr['edges']):
            partner = stitch_map.get(('pant_f_r', i))
            if partner and partner[0] == 'pant_b_r':
                ep = e['endpoints']
                v1, v2 = fr_verts[ep[0]], fr_verts[ep[1]]
                # Inseam: one endpoint near bottom (y≈0), other higher up (crotch area)
                min_y = min(v[1] for v in fr_verts)
                if (abs(v1[1] - min_y) < 1.0 or abs(v2[1] - min_y) < 1.0):
                    inseam = edge_length(e, fr_verts)
                    break
    results['Inseam'] = inseam if inseam > 0 else None

    # --- Hip circumference ---
    # Use the pant panel top edges (stitched to waistband) to get waist-level width.
    # The actual hip circumference at the hip line is wider due to crotch extensions.
    # For a practical approximation: find the max horizontal extent of each pant panel
    # at the top (waist/hip level), considering both the top edge and crotch edges.
    hip_circ = 0.0
    for pname in ['pant_f_r', 'pant_b_r']:
        panel = panels.get(pname)
        if not panel:
            continue
        verts = panel['vertices']
        # Find vertices at the top y-level (waist/hip area, y > 80% of max_y)
        max_y = max(v[1] for v in verts)
        top_verts = [v for v in verts if v[1] > max_y * 0.8]
        if top_verts:
            # Width at top = max_x - min_x
            xs = [v[0] for v in top_verts]
            hip_circ += max(xs) - min(xs)
    results['Hip_approx'] = 2.0 * hip_circ if hip_circ > 0 else None

    # --- Compute design-based hip circumference ---
    # The hip is controlled by width_v × body.hips. Read design params from mapper.
    with open(BODY_YAML) as f:
        body_data = yaml.safe_load(f)['body']
    body_hips = body_data['hips']
    # width_v = hip_circumference / body.hips
    design_hip = prod['Low_Hip']  # This is what was passed to map_pants
    # The actual designed hip might be clipped: width_v = clip(hip/body_hips, 1.0, 1.5)
    width_v = np.clip(design_hip / body_hips, 1.0, 1.5)
    results['Hip_designed'] = width_v * body_hips

    # --- Print results ---
    print(f'\n  {"="*60}')
    print(f'  Measurement Verification – Size {size}')
    print(f'  {"="*60}')
    print(f'  {"Measurement":<25} {"Target":>10} {"Measured":>10} {"Delta":>10}  Notes')
    print(f'  {"-"*75}')

    checks = [
        ('Waist',        'Waist',    'WB outer edge = body waist (waistband.waist.v=1.0)'),
        ('Hip_designed',  'Low_Hip',  'width_v × body.hips (design param, clipped [1.0,1.5])'),
        ('Hip_approx',   'Low_Hip',  'Geometric approx from panel vertex extent'),
        ('Inseam',       'Inseam',   'Front inseam edge length (crotch-to-hem)'),
        ('Ankle',        'Ankle',    'Free bottom edges (front+back = one leg circumference)'),
    ]

    for meas_key, prod_key, note in checks:
        measured = results.get(meas_key)
        target = prod[prod_key]
        if measured is not None:
            delta = measured - target
            status = 'OK' if abs(delta) <= 1.0 else ''
            print(f'  {meas_key:<25} {target:>10.1f} {measured:>10.1f} {delta:>+10.1f}  {status} {note}')
        else:
            print(f'  {meas_key:<25} {target:>10.1f} {"N/A":>10} {"":>10}  {note}')

    print(f'\n  Notes:')
    print(f'    - Waist: WB always matches body waist ({body_hips:.0f}×waist_ratio). '
          f'Production waist ({prod["Waist"]}) < body waist ({body_data["waist"]}) = elastic/stretch fit.')
    print(f'    - Ankle: flare_v derived from panel geometry (accounts for crotch extension).')
    print(f'    - Thigh/Knee/Calf: {prod["Thigh"]}/{prod["Knee"]}/{prod["Calf"]} cm '
          f'(not individually controllable, system uses width+flare)')

    return results


def save_combined_mesh(sim_folder,
                       body_obj_path=f'./assets/bodies/{BODY_NAME}.obj'):
    """Create combined body + garment mesh."""
    import trimesh

    sim_folder = Path(sim_folder)
    garment_files = list(sim_folder.glob('*_sim.obj'))
    if not garment_files:
        print(f'  No garment mesh found in {sim_folder}')
        return

    garment_path = garment_files[0]
    garment = trimesh.load(str(garment_path), process=False)
    body = trimesh.load(str(body_obj_path), process=False)

    if body.vertices.max() < 3.0:
        body.vertices = body.vertices * 100.0
    min_y = body.vertices[:, 1].min()
    if min_y < 0:
        body.vertices[:, 1] += abs(min_y)

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
    print("  Custom Pants Pipeline (Built-in Parametric System)")
    print("  Production: CALITEE Heritage Pants")
    print("  Body: GLOBAL MEN size 50")
    print("  Garment sizes: 48, 50, 52")
    print(f"  Simulation body: {BODY_NAME} (A-pose)")
    print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 60)

    # Step 1: Generate patterns for all sizes
    print("\n--- Step 1: Generating patterns ---")
    generated = []
    for size in GARMENT_SIZES:
        prod = PRODUCTION_DATA[size]
        print(f'\nSize {size}: Waist={prod["Waist"]}, Hip={prod["Low_Hip"]}, '
              f'Inseam={prod["Inseam"]}, Ankle={prod["Ankle"]}')
        design = map_production_to_design(prod, BODY_YAML)
        print(f'  Design: width={design["pants"]["width"]["v"]:.3f}, '
              f'length={design["pants"]["length"]["v"]:.3f}, '
              f'flare={design["pants"]["flare"]["v"]:.3f}')
        folder, name = generate_pattern(size, design, BODY_YAML, output_base)
        generated.append((folder, name, size))

    # Step 2: Verify measurements from spec JSONs
    print("\n--- Step 2: Verifying pattern measurements ---")
    for folder, name, size in generated:
        spec_files = list(folder.glob('*_specification.json'))
        if spec_files:
            verify_measurements(spec_files[0], size, PRODUCTION_DATA[size])
        else:
            print(f'  No spec file found for size {size}')

    # Step 3: Simulate each pattern on A-pose body
    print("\n--- Step 3: Running simulations (A-pose) ---")
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
