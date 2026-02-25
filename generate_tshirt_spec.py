"""
    Generate t-shirt sewing patterns from exact spec-sheet measurements
    and simulate them on a body mesh.

    Sizes: 48, 50, 52 (European)
    Body: male mannequin (height 180, chest 100)
"""
import math
from pathlib import Path
from datetime import datetime

import numpy as np
from scipy.spatial.transform import Rotation as R

import pygarment as pyg
from pygarment.data_config import Properties


# ============================================================
# Spec measurements per size (full garment circumference, cm)
# ============================================================
SPECS = {
    48: dict(A=101, D=101, I=36.6, J=43.6, M=37.1, N=16.7, O1=10.7, O2=1.9, P=2.0, Z=71,
             M_cuff=34.1, M1=2.0),
    50: dict(A=105, D=105, I=37.3, J=44.8, M=38.2, N=17.0, O1=11.0, O2=2.1, P=2.0, Z=72,
             M_cuff=35.2, M1=2.0),
    52: dict(A=109, D=109, I=37.9, J=46.0, M=39.3, N=17.2, O1=11.4, O2=2.2, P=2.0, Z=73,
             M_cuff=36.3, M1=2.0),
}

# Body measurements (male mannequin) -- used for 2D panel geometry
BODY = dict(
    height=180,
    head_l=24.75,
    chest=100,
    waist=88,
    shoulder_length=15,
    shoulder_angle_deg=23.5,
    shoulder_drop=6,
    armhole_depth=13.75,
    arm_length=64,
    bicep=32,
    wrist=17.5,
    across_shoulder=44,
    across_back=40.25,
    across_front=37.5,
    neck_w=16.7,
    arm_pose_angle=45,
)

# Simulation body (mean_male) -- used for 3D panel placement only
SIM_BODY = dict(
    height=178.873,
    head_l=27.2653,
    shoulder_w=38.5276,
    arm_pose_angle=45.4775,
)


# ============================================================
# Panel classes
# ============================================================

class TorsoHalfPanel(pyg.Panel):
    """Half of a front or back torso panel, built from spec measurements.

    Edges (counterclockwise):
        0: hem        (P0 -> P1)
        1: side seam  (P1 -> P2)
        2: armhole    (P2 -> P3)  -- cubic bezier curve
        3: shoulder   (P3 -> P4)
        4: neckline   (P4 -> P5)  -- circle arc
        5: center     (P5 -> P0)  -- fold line
    """

    def __init__(self, name, spec, body, sim_body, is_front=False):
        super().__init__(name)

        A = spec['A']
        D = spec['D']
        N = spec['N']
        Z = spec['Z']
        neck_drop = spec['O1'] if is_front else spec['O2']

        sh_angle = math.radians(body['shoulder_angle_deg'])
        sh_length = body['shoulder_length']
        sh_drop = body['shoulder_drop']
        ah_depth = body['armhole_depth']

        # Key points (half-panel, x negative = towards side)
        p0 = [0, 0]                              # hem center
        p1 = [-D / 4, 0]                         # hem side
        underarm_y = Z - sh_drop - ah_depth
        p2 = [-A / 4, underarm_y]                # underarm
        shoulder_x = N / 2 + sh_length * math.cos(sh_angle)
        shoulder_y = Z - sh_drop
        p3 = [-shoulder_x, shoulder_y]            # shoulder tip
        p4 = [-N / 2, Z]                          # HPS
        p5 = [0, Z - neck_drop]                   # center neck

        # Build edges
        e_hem = pyg.Edge(p0, p1)
        e_side = pyg.Edge(p1, p2)

        # Armhole: cubic bezier
        e_armhole = pyg.CurveEdge(p2, p3, control_points=[[0.4, 0.15], [0.75, 0.3]])

        e_shoulder = pyg.Edge(p3, p4)

        # Neckline: circle arc curving inward
        e_neckline = pyg.CircleEdgeFactory.from_three_points(
            p4, p5,
            [(p4[0] + p5[0]) / 2, (p4[1] + p5[1]) / 2 - neck_drop * 0.3]
        )

        e_center = pyg.Edge(p5, p0)

        self.edges = pyg.EdgeSequence(
            e_hem, e_side, e_armhole, e_shoulder, e_neckline, e_center
        )

        # Interfaces
        self.interfaces = {
            'hem': pyg.Interface(self, e_hem),
            'side': pyg.Interface(self, e_side),
            'armhole': pyg.Interface(self, e_armhole),
            'shoulder': pyg.Interface(self, e_shoulder),
            'neckline': pyg.Interface(self, e_neckline),
            'center': pyg.Interface(self, e_center),
        }

        # Set label for body segmentation
        self.set_panel_label('torso')

        # 3D placement (using simulation body measurements)
        top_y = sim_body['height'] - sim_body['head_l']
        z_offset = 20 if is_front else -15
        self.top_center_pivot()
        self.translate_to([0, top_y, z_offset])


class SleeveHalfPanel(pyg.Panel):
    """Half-sleeve panel (front or back half).

    Edges (counterclockwise):
        0: opening edge   (P0 -> P1)
        1: underseam      (P1 -> P2)
        2: armhole cap    (P2 -> P3) -- cubic bezier curve
        3: top edge       (P3 -> P0)
    """

    def __init__(self, name, spec, body, sim_body, is_front=False):
        super().__init__(name)

        J = spec['J']
        M = spec['M']
        I = spec['I']
        sh_length = body['shoulder_length']

        sleeve_length = I - sh_length
        half_arm_width = J / 2
        half_opening = M / 2

        # Points (x=length direction, y=width)
        p0 = [0, 0]
        p1 = [0, -half_opening]
        p2 = [sleeve_length, -half_arm_width]
        p3 = [sleeve_length, 0]

        e_opening = pyg.Edge(p0, p1)
        e_underseam = pyg.Edge(p1, p2)
        e_cap = pyg.CurveEdge(p2, p3, control_points=[[0.5, 0.2], [0.8, 0.35]])
        e_top = pyg.Edge(p3, p0)

        self.edges = pyg.EdgeSequence(e_opening, e_underseam, e_cap, e_top)

        self.interfaces = {
            'opening': pyg.Interface(self, e_opening),
            'underseam': pyg.Interface(self, e_underseam),
            'cap': pyg.Interface(self, e_cap),
            'top': pyg.Interface(self, e_top),
        }

        self.set_panel_label('arm')

        # 3D placement (using simulation body measurements)
        z_offset = 15 if is_front else -15
        self.set_pivot(p3)
        self.translate_to([
            -sim_body['shoulder_w'] / 2,
            sim_body['height'] - sim_body['head_l'],
            z_offset
        ])


# ============================================================
# Component classes
# ============================================================

class SleeveGroup(pyg.Component):
    """Sleeve as a component: front + back half-panels, rotated to arm pose."""

    def __init__(self, name, spec, body, sim_body):
        super().__init__(name)

        self.sleeve_f = SleeveHalfPanel(f'{name}_f', spec, body, sim_body, is_front=True)
        self.sleeve_b = SleeveHalfPanel(f'{name}_b', spec, body, sim_body, is_front=False)

        self.stitching_rules = pyg.Stitches(
            (self.sleeve_f.interfaces['top'], self.sleeve_b.interfaces['top']),
            (self.sleeve_f.interfaces['underseam'], self.sleeve_b.interfaces['underseam']),
        )

        self.interfaces = {
            'cap_f': self.sleeve_f.interfaces['cap'],
            'cap_b': self.sleeve_b.interfaces['cap'],
        }

        # Rotate to match simulation body arm pose angle
        self.rotate_by(R.from_euler(
            'XYZ', [0, 0, sim_body['arm_pose_angle']], degrees=True))


class TShirtHalf(pyg.Component):
    """One half of the t-shirt (right side): front + back + sleeve."""

    def __init__(self, name, spec, body, sim_body):
        super().__init__(name)

        self.back = TorsoHalfPanel(f'{name}_back', spec, body, sim_body, is_front=False)
        self.front = TorsoHalfPanel(f'{name}_front', spec, body, sim_body, is_front=True)
        self.sleeve = SleeveGroup(f'{name}_sleeve', spec, body, sim_body)

        self.stitching_rules = pyg.Stitches(
            # Side seams
            (self.front.interfaces['side'], self.back.interfaces['side']),
            # Shoulder seams
            (self.front.interfaces['shoulder'], self.back.interfaces['shoulder']),
            # Armhole: front sleeve cap <-> front body armhole
            (self.sleeve.interfaces['cap_f'], self.front.interfaces['armhole']),
            # Armhole: back sleeve cap <-> back body armhole
            (self.sleeve.interfaces['cap_b'], self.back.interfaces['armhole']),
        )

        self.interfaces = {
            'center_front': self.front.interfaces['center'],
            'center_back': self.back.interfaces['center'],
            'neckline_front': self.front.interfaces['neckline'],
            'neckline_back': self.back.interfaces['neckline'],
        }


class TShirt(pyg.Component):
    """Full t-shirt: right half + mirrored left half."""

    def __init__(self, name, spec, body, sim_body):
        super().__init__(name)

        self.right = TShirtHalf(f'{name}_r', spec, body, sim_body)
        self.left = TShirtHalf(f'{name}_l', spec, body, sim_body)
        self.left.mirror()

        # Stitch the two body halves at center
        self.stitching_rules = pyg.Stitches(
            (self.right.interfaces['center_front'],
             self.left.interfaces['center_front']),
            (self.right.interfaces['center_back'],
             self.left.interfaces['center_back']),
        )


# ============================================================
# Main: generate patterns + run simulation for each size
# ============================================================

def generate_pattern(size, spec, body, sim_body, output_base):
    """Generate pattern for one size and return the output folder path."""
    garment_name = f'tshirt_size{size}'
    tshirt = TShirt(garment_name, spec, body, sim_body)
    pattern = tshirt.assembly()

    if tshirt.is_self_intersecting():
        print(f'WARNING: {garment_name} has self-intersecting panels')

    folder = pattern.serialize(
        output_base,
        tag='_' + datetime.now().strftime("%y%m%d-%H-%M-%S"),
        to_subfolder=True,
        with_3d=False,
        with_text=False,
        view_ids=False,
        with_printable=True
    )

    print(f'Pattern generated: {garment_name} -> {folder}')
    return Path(folder), garment_name


def simulate_pattern(pattern_folder, garment_name, output_base):
    """Run physics simulation for a generated pattern on the mean_male body."""
    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.simulation import run_sim
    from pygarment.meshgen.sim_config import PathCofig

    sim_config_path = './assets/Sim_props/default_sim_props.yaml'
    props = Properties(sim_config_path)
    props.set_section_stats(
        'sim', fails={}, sim_time={}, spf={},
        fin_frame={}, body_collisions={}, self_collisions={}
    )
    props.set_section_stats('render', render_time={})

    spec_files = list(pattern_folder.glob('*_specification.json'))
    if not spec_files:
        print(f'ERROR: No specification file found in {pattern_folder}')
        return
    spec_file = spec_files[0]
    in_name = spec_file.stem.replace('_specification', '')

    paths = PathCofig(
        in_element_path=pattern_folder,
        out_path=output_base,
        in_name=in_name,
        body_name='mean_male',
        smpl_body=False,
        add_timestamp=True
    )

    print(f'Generating box mesh for {in_name}...')
    resolution_scale = props['sim']['config']['resolution_scale']
    garment_box_mesh = BoxMesh(paths.in_g_spec, resolution_scale)
    garment_box_mesh.load()
    garment_box_mesh.serialize(
        paths, store_panels=False,
        uv_config=props['render']['config']['uv_texture']
    )

    props.serialize(paths.element_sim_props)

    print(f'Running simulation for {in_name}...')
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
    print(f'Simulation complete: {in_name} -> {paths.out_el}')


if __name__ == '__main__':
    sys_props = Properties('./system.json')
    output_base = sys_props['output']

    # Step 1: Generate patterns for all sizes
    generated = []
    for size in [48, 50, 52]:
        spec = SPECS[size]
        folder, name = generate_pattern(size, spec, BODY, SIM_BODY, output_base)
        generated.append((folder, name, size))

    # Step 2: Simulate each pattern on the mean_male body
    for folder, name, size in generated:
        print(f'\n--- Simulating size {size} ---')
        try:
            simulate_pattern(folder, name, output_base)
        except Exception as e:
            print(f'Simulation failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    print('\nDone! Check Logs/ for output.')
