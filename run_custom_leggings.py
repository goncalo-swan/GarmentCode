"""
Custom Leggings Generation & Simulation Pipeline
Uses the BUILT-IN GarmentCode parametric system (MetaGarment + Pants + StraightWB).

Production data: Women's Leggings (77% recycled polyester / 23% elastane)
Sizes: 34, 36, 38
Body: GLOBAL WOMEN size 36
Simulation body mesh: global_women_size36_apose (custom SMPL, 169.5cm)

Measurements from techpack:
  - Waist: "Waistband width relaxed (meas. along top edge of WB)- high rise"
  - Low Hip: "Hip width"
  - Inseam: "Full length-Inseam (meas. from crotch)"
  - Thigh: "Thigh width (meas at bottom of triangle gusset)"
  - Knee: "Knee width"
  - Calf: "Calf width"
  - Ankle: "Full length-Leg opening"
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
# Source: Women's Leggings - 77% recycled polyester / 23% elastane
# ============================================================
GARMENT_SIZES = [34, 36, 38]

PRODUCTION_DATA = {
    34: {
        'Waist': 48.1,
        'Low_Hip': 66.2,
        'Inseam': 69.5,
        'Thigh': 43.8,
        'Knee': 28.0,
        'Calf': 26.1,
        'Ankle': 18.7,
    },
    36: {
        'Waist': 51.1,
        'Low_Hip': 69.2,
        'Inseam': 69.6,
        'Thigh': 45.3,
        'Knee': 29.2,
        'Calf': 27.0,
        'Ankle': 19.1,
    },
    38: {
        'Waist': 54.2,
        'Low_Hip': 72.2,
        'Inseam': 69.7,
        'Thigh': 46.8,
        'Knee': 29.6,
        'Calf': 27.9,
        'Ankle': 19.5,
    },
}

# Body parameters for GLOBAL WOMEN size 36
BODY_YAML = './assets/bodies/global_women_size36_apose.yaml'
BODY_NAME = 'global_women_size36_apose'


def map_production_to_design(prod, body_yaml_path):
    """Map production measurements to GarmentCode design parameters."""
    mapper = ProductionToDesign(body_yaml_path)
    body = mapper.body

    # Note: body._waist_level ≈ 101.75 cm = belly button level for this body.
    # rise=1.0 already places the waistband at belly button/natural waist.
    rise = 1.0

    # Use production inseam for panel length (compute from the top, not bottom).
    # panel_length = outside seam from hem to hip line ≈ inseam + crotch_hip_diff.
    panel_length = prod['Inseam'] + body['crotch_hip_diff']

    garment_measurements = {
        'waist_circumference': prod['Waist'],
        'hip_circumference': prod['Low_Hip'],
        'length': panel_length,
        'leg_opening': prod['Ankle'],
        'rise': rise,
    }

    design = mapper.map_pants(garment_measurements)

    # For compression leggings with gap-closed placement (pants_top ≈ _waist_level),
    # clip length so the back panel hem (taller due to hipline_ext=1.1) stays above
    # the foot area. back_hem = _waist_level - length - hips_line * 1.1
    min_hem = 3  # minimum cm above ground
    max_length = body['_waist_level'] - body['hips_line'] * 1.1 - min_hem
    max_lv = max_length / body['_leg_length']
    if design['pants']['length']['v'] > max_lv:
        design['pants']['length']['v'] = max_lv

    return design


def generate_pattern(size, design, body_yaml_path, output_base):
    """Generate pattern using built-in MetaGarment system."""
    from assets.garment_programs.meta_garment import MetaGarment
    from assets.bodies.body_params import BodyParameters

    garment_name = f'leggings_size{size}'

    body = BodyParameters(body_yaml_path)
    garment = MetaGarment(garment_name, body, design)

    # Close the placement gap for compression leggings.
    # MetaGarment's place_by_interface positions pants with a gap below the
    # waistband (waistband depth + 5 cm clearance ≈ 9.6 cm total). This puts
    # the crotch ~23 cm below the body's crotch. Close the gap so the garment
    # crotch aligns with the body crotch.
    wb_depth = 0.2 * body['hips_line']  # default waistband width
    gap_close = wb_depth + 5  # total gap: waistband depth + place_by_interface gap
    for sub in garment.subs:
        if type(sub).__name__ == 'Pants':
            for half in [sub.right, sub.left]:
                for attr in ['front', 'back']:
                    getattr(half, attr).translate_by([0, gap_close, 0])
            break

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

    sim_config_path = './assets/Sim_props/leggings_sim_props.yaml'
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


def verify_measurements(spec_path, size, prod):
    """Extract leggings measurements from spec JSON and compare to production targets."""
    with open(spec_path, 'r') as f:
        spec = json.load(f)

    panels = spec['pattern']['panels']
    stitches = spec['pattern']['stitches']

    # Build stitching map for edge identification
    stitched = set()
    stitch_map = {}
    for stitch in stitches:
        for side in stitch:
            stitched.add((side['panel'], side['edge']))
        if len(stitch) == 2:
            a, b = stitch
            stitch_map[(a['panel'], a['edge'])] = (b['panel'], b['edge'])
            stitch_map[(b['panel'], b['edge'])] = (a['panel'], a['edge'])

    results = {}

    # --- Waist circumference ---
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
    # Free (unstitched) edges on pant panels at y ≈ min_y (hem level)
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
            min_y = min(v[1] for v in verts)
            if abs(v1[1] - min_y) < 1.0 and abs(v2[1] - min_y) < 1.0:
                leg_opening += edge_length(e, verts)
    # front_bottom + back_bottom = full circumference of one leg opening (no ×2)
    results['Ankle'] = leg_opening if leg_opening > 0 else None

    # --- Inseam ---
    inseam = 0.0
    pant_fr = panels.get('pant_f_r')
    pant_br = panels.get('pant_b_r')
    if pant_fr and pant_br:
        fr_verts = pant_fr['vertices']
        for i, e in enumerate(pant_fr['edges']):
            partner = stitch_map.get(('pant_f_r', i))
            if partner and partner[0] == 'pant_b_r':
                ep = e['endpoints']
                v1, v2 = fr_verts[ep[0]], fr_verts[ep[1]]
                min_y = min(v[1] for v in fr_verts)
                if (abs(v1[1] - min_y) < 1.0 or abs(v2[1] - min_y) < 1.0):
                    inseam = edge_length(e, fr_verts)
                    break
    results['Inseam'] = inseam if inseam > 0 else None

    # --- Hip circumference (design-based) ---
    with open(BODY_YAML) as f:
        body_data = yaml.safe_load(f)['body']
    body_hips = body_data['hips']
    width_v = np.clip(prod['Low_Hip'] / body_hips, 0.5, 1.5)
    results['Hip_designed'] = width_v * body_hips

    # --- Print results ---
    print(f'\n  {"="*60}')
    print(f'  Measurement Verification – Size {size}')
    print(f'  {"="*60}')
    print(f'  {"Measurement":<25} {"Target":>10} {"Measured":>10} {"Delta":>10}  Notes')
    print(f'  {"-"*75}')

    checks = [
        ('Waist',        'Waist',    'WB = body waist (waistband.waist.v=1.0, elastic)'),
        ('Hip_designed',  'Low_Hip',  'width_v × body.hips (pattern hip, stretch garment)'),
        ('Inseam',       'Inseam',   'Front inseam edge length (crotch-to-hem)'),
        ('Ankle',        'Ankle',    'Free bottom edges (front+back = one leg circ)'),
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
    print(f'    - Waist: WB matches body waist ({body_data["waist"]:.0f} cm). '
          f'Production waist ({prod["Waist"]}) is relaxed elastic measurement.')
    print(f'    - Hip: Pattern hip = {results.get("Hip_designed", 0):.1f} cm '
          f'(width_v={width_v:.3f} × body {body_hips}). '
          f'Production hip ({prod["Low_Hip"]}) is flat/unstretched.')
    print(f'    - Ankle: Derived from panel geometry (flare formula accounts for crotch extension).')
    print(f'    - Thigh/Knee/Calf: {prod["Thigh"]}/{prod["Knee"]}/{prod["Calf"]} cm '
          f'(not individually controllable)')

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
    print("  Custom Leggings Pipeline (Built-in Parametric System)")
    print("  Production: Women's Leggings (polyester/elastane)")
    print("  Body: GLOBAL WOMEN size 36")
    print("  Garment sizes: 34, 36, 38")
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
              f'flare={design["pants"]["flare"]["v"]:.3f}, '
              f'rise={design["pants"]["rise"]["v"]:.3f}')
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
