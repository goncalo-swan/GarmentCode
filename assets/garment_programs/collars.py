import numpy as np
from scipy.spatial.transform import Rotation as R

import pygarment as pyg

from assets.garment_programs.bands import StraightBandPanel
from assets.garment_programs.circle_skirt import CircleArcPanel


# # ------ Collar shapes withough extra panels ------

def VNeckHalf(depth, width, **kwargs):
    """Simple VNeck design"""

    edges = pyg.EdgeSequence(pyg.Edge([0, 0], [width / 2, -depth]))
    return edges

def SquareNeckHalf(depth, width, **kwargs):
    """Square design"""

    edges = pyg.EdgeSeqFactory.from_verts([0, 0], [0, -depth], [width / 2, -depth])
    return edges

def TrapezoidNeckHalf(depth, width, angle=90, verbose=True, **kwargs):
    """Trapesoid neck design"""

    # Special case when angle = 180 (sin = 0)
    if (pyg.utils.close_enough(angle, 180, tol=1) 
            or pyg.utils.close_enough(angle, 0, tol=1)):
        # degrades into VNeck
        return VNeckHalf(depth, width)

    rad_angle = np.deg2rad(angle)

    bottom_x = -depth * np.cos(rad_angle) / np.sin(rad_angle)
    if bottom_x > width / 2:  # Invalid angle/depth/width combination resulted in invalid shape
        if verbose:
            print('TrapezoidNeckHalf::WARNING::Parameters are invalid and create overlap: '
                  f'{bottom_x} > {width / 2}. '
                  'The collar is reverted to VNeck')

        return VNeckHalf(depth, width)

    edges = pyg.EdgeSeqFactory.from_verts([0, 0], [bottom_x, -depth], [width / 2, -depth])
    return edges

def CurvyNeckHalf(depth, width, flip=False, **kwargs):
    """Testing Curvy Collar design"""

    sign = -1 if flip else 1
    edges = pyg.EdgeSequence(pyg.CurveEdge(
        [0, 0], [width / 2,-depth], 
        [[0.4, sign * 0.3], [0.8, sign * -0.3]]))
    
    return edges

def CircleArcNeckHalf(depth, width, angle=90, flip=False, **kwargs):
    """Collar with a side represented by a circle arc"""
    # 1/4 of a circle
    edges = pyg.EdgeSequence(pyg.CircleEdgeFactory.from_points_angle(
        [0, 0], [width / 2,-depth], arc_angle=np.deg2rad(angle),
        right=(not flip)
    ))

    return edges


def CircleNeckHalf(depth, width, **kwargs):
    """Collar that forms a perfect circle arc when halfs are stitched"""

    # Take a full desired arc and half it!
    circle = pyg.CircleEdgeFactory.from_three_points(
        [0, 0],
        [width, 0],
        [width / 2, -depth])
    subdiv = circle.subdivide_len([0.5, 0.5])
    return pyg.EdgeSequence(subdiv[0])

def Bezier2NeckHalf(depth, width, flip=False, x=0.5, y=0.3, **kwargs):
    """2d degree Bezier curve as neckline"""

    sign = 1 if flip else -1
    edges = pyg.EdgeSequence(pyg.CurveEdge(
        [0, 0], [width / 2,-depth], 
        [[x, sign*y]]))
    
    return edges

# # ------ Collars with panels ------

class NoPanelsCollar(pyg.Component):
    """Face collar class that only forms the projected shapes """
    
    def __init__(self, name, body, design) -> None:
        super().__init__(name)

        # Front
        collar_type = globals()[design['collar']['f_collar']['v']]
        f_collar = collar_type(
            design['collar']['fc_depth']['v'],
            design['collar']['width']['v'], 
            angle=design['collar']['fc_angle']['v'], 
            flip=design['collar']['f_flip_curve']['v'],
            x=design['collar']['f_bezier_x']['v'],
            y=design['collar']['f_bezier_y']['v'],
            verbose=self.verbose
        )

        # Back
        collar_type = globals()[design['collar']['b_collar']['v']]
        b_collar = collar_type(
            design['collar']['bc_depth']['v'], 
            design['collar']['width']['v'], 
            angle=design['collar']['bc_angle']['v'],
            flip=design['collar']['b_flip_curve']['v'],
            x=design['collar']['b_bezier_x']['v'],
            y=design['collar']['b_bezier_y']['v'],
            verbose=self.verbose
        )
        
        self.interfaces = {
            'front_proj': pyg.Interface(self, f_collar),
            'back_proj': pyg.Interface(self, b_collar)
        }
    
    def length(self):
        return 0


class Turtle(pyg.Component):

    def __init__(self, tag, body, design) -> None:
        super().__init__(f'Turtle_{tag}')

        depth = design['collar']['component']['depth']['v']

        # --Projecting shapes--
        f_collar = CircleNeckHalf(
            design['collar']['fc_depth']['v'],
            design['collar']['width']['v'])
        b_collar = CircleNeckHalf(
            design['collar']['bc_depth']['v'],
            design['collar']['width']['v'])
        
        self.interfaces = {
            'front_proj': pyg.Interface(self, f_collar),
            'back_proj': pyg.Interface(self, b_collar)
        }

        # -- Panels --
        length_f, length_b = f_collar.length(), b_collar.length()
        height_p = body['height'] - body['head_l'] + depth

        self.front = StraightBandPanel(
            f'{tag}_collar_front', length_f, depth).translate_by(
            [-length_f / 2, height_p, 10])
        self.back = StraightBandPanel(
            f'{tag}_collar_back', length_b, depth).translate_by(
            [-length_b / 2, height_p, -10])

        self.stitching_rules.append((
            self.front.interfaces['right'], 
            self.back.interfaces['right']
        ))

        self.interfaces.update({
            'front': self.front.interfaces['left'],
            'back': self.back.interfaces['left'],
            'bottom': pyg.Interface.from_multiple(
                self.front.interfaces['bottom'],
                self.back.interfaces['bottom']
            )
        })

    def length(self):
        return self.interfaces['back'].edges.length()


class SimpleLapelPanel(pyg.Panel):
    """A panel for the front part of simple Lapel"""
    def __init__(self, name, length, max_depth) -> None:
        super().__init__(name)

        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0], [max_depth, 0], [max_depth, -length]
        )

        self.edges.append(
            pyg.CurveEdge(
                self.edges[-1].end,
                self.edges[0].start,
                [[0.7, 0.2]]
            )
        )

        self.interfaces = {
            'to_collar': pyg.Interface(self, self.edges[0]),
            'to_bodice': pyg.Interface(self, self.edges[1]),
            'outer': pyg.Interface(self, self.edges[2]),
        }


# ------ Split-panel classes for lapel fold ------

class FrontCollarInnerPanel(pyg.Panel):
    """Inner strip of front lapel (closer to neck). Connects to back_stand.
    Fold_line runs vertically on the right side (inset to avoid corner sharing).
    """
    def __init__(self, name, width, height, gap=0.5) -> None:
        super().__init__(name)

        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0],                        # to_collar start
            [width, 0],                    # to_collar end
            [width, -gap],                 # corner tab
            [width, -(height - gap)],      # fold_line (inset)
            [width, -height],              # corner tab
            [0, -height],                  # bottom
            loop=True
        )
        self.interfaces = {
            'to_collar': pyg.Interface(self, self.edges[0]),
            'fold_line': pyg.Interface(self, self.edges[2]),
        }


class FrontCollarOuterPanel(pyg.Panel):
    """Outer strip of front lapel (the fold-over part). Connects to back_fall.
    Fold_line runs vertically on the left side (inset).
    to_bodice on the right side connects to the bodice neckline.
    """
    def __init__(self, name, width, height, gap=0.5) -> None:
        super().__init__(name)

        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0],                        # to_collar start
            [width, 0],                    # to_collar end = to_bodice start
            [width, -height],              # to_bodice end
            [0, -height],                  # bottom-left
            [0, -(height - gap)],          # corner tab
            [0, -gap],                     # fold_line (inset, going up)
            loop=True
        )
        self.interfaces = {
            'to_collar': pyg.Interface(self, self.edges[0]),
            'to_bodice': pyg.Interface(self, self.edges[1]),
            'fold_line': pyg.Interface(self, self.edges[4]),
        }


class LapelStandPanel(pyg.Panel):
    """Upper portion of front lapel — stands upright from the neckline to the fold line.

    to_bodice covers the FULL original length (not just stand_h) so the
    neckline stitch matches exactly without subdivisions. The fold_line
    is an inset edge at y=-stand_h, connected to to_bodice via a diagonal.
    """
    def __init__(self, name, stand_h, full_length, max_depth, gap=0.5) -> None:
        super().__init__(name)

        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0],                            # to_collar start
            [max_depth, 0],                    # to_collar end
            [max_depth, -full_length],         # to_bodice end (FULL length)
            [max_depth - gap, -stand_h],       # diagonal to fold_line
            [gap, -stand_h],                   # fold_line (inset)
            [0, -stand_h],                     # inner start
            loop=True
        )
        self.interfaces = {
            'to_collar': pyg.Interface(self, self.edges[0]),
            'to_bodice': pyg.Interface(self, self.edges[1]),
            'fold_line': pyg.Interface(self, self.edges[3]),
        }


class SimpleLapelFallPanel(pyg.Panel):
    """Lower portion of front lapel — folds over the stand. Carries the curved decorative edge.

    fold_line is inset to avoid corner sharing. No to_bodice — the stand
    panel covers the full neckline length.
    """
    def __init__(self, name, fall_h, max_depth, gap=0.5) -> None:
        super().__init__(name)

        self.edges = pyg.EdgeSeqFactory.from_verts(
            [gap, 0],                     # fold_line start (inset)
            [max_depth - gap, 0],         # fold_line end (inset)
            [max_depth, 0],               # corner tab
            [max_depth, -fall_h]
        )
        self.edges.append(
            pyg.CurveEdge(
                self.edges[-1].end,
                [0, 0],
                [[0.7, 0.2]]
            )
        )
        self.edges.append(pyg.EdgeSeqFactory.from_verts(
            [0, 0], [gap, 0]
        )[0])

        self.interfaces = {
            'fold_line': pyg.Interface(self, self.edges[0]),
        }


class BackCollarStandPanel(pyg.Panel):
    """Lower portion of back collar — stands upright from the neckline.

    Fold_line is inset by `gap` to avoid sharing corners with right/left edges.
    """
    def __init__(self, name, width, stand_depth, gap=0.5) -> None:
        super().__init__(name)

        # Top edge (fold line) is inset: doesn't touch right/left corners
        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0],                            # 0: right start
            [0, stand_depth],                  # 1: right end
            [gap, stand_depth],                # 2: corner tab right
            [width - gap, stand_depth],        # 3: fold_line (inset)
            [width, stand_depth],              # 4: corner tab left
            [width, 0],                        # 5: left end
            loop=True
        )

        self.interfaces = {
            'right': pyg.Interface(self, self.edges[0]),
            'fold_line': pyg.Interface(self, self.edges[2]).reverse(True),
            'left': pyg.Interface(self, self.edges[4]),
            'bottom': pyg.Interface(self, self.edges[5]),
        }

        self.top_center_pivot()
        self.center_x()


class BackCollarFallPanel(pyg.Panel):
    """Upper portion of back collar — rolls outward from the fold line.

    Fold_line (bottom) is inset by `gap` to avoid sharing corners with right/left edges.
    """
    def __init__(self, name, width, fall_depth, gap=0.5) -> None:
        super().__init__(name)

        # Bottom edge (fold line) is inset: doesn't touch right/left corners
        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0],                            # 0: right start
            [0, fall_depth],                   # 1: right end = top start
            [width, fall_depth],               # 2: top end = left start
            [width, 0],                        # 3: left end
            [width - gap, 0],                  # 4: corner tab left
            [gap, 0],                          # 5: fold_line (inset)
            loop=True
        )

        self.interfaces = {
            'right': pyg.Interface(self, self.edges[0]),
            'top': pyg.Interface(self, self.edges[1]).reverse(True),
            'left': pyg.Interface(self, self.edges[2]),
            'fold_line': pyg.Interface(self, self.edges[4]),
        }

        self.top_center_pivot()
        self.center_x()


class SimpleLapel(pyg.Component):

    def __init__(self, tag, body, design) -> None:
        super().__init__(f'Turtle_{tag}')

        depth = design['collar']['component']['depth']['v']

        # --Projecting shapes--
        # Any front one!
        collar_type = globals()[design['collar']['f_collar']['v']]
        f_collar = collar_type(
            design['collar']['fc_depth']['v'],
            design['collar']['width']['v'], 
            angle=design['collar']['fc_angle']['v'], 
            flip=design['collar']['f_flip_curve']['v'])
        
        b_collar = CircleNeckHalf(
            design['collar']['bc_depth']['v'],
            design['collar']['width']['v'])
        
        self.interfaces = {
            'front_proj': pyg.Interface(self, f_collar),
            'back_proj': pyg.Interface(self, b_collar)
        }

        # -- Panels --
        length_f, length_b = f_collar.length(), b_collar.length()
        height_p = body['height'] - body['head_l'] + depth * 2

        lapel_fold = design['collar']['component'].get('lapel_fold', {}).get('v', 0)

        if lapel_fold > 0:
            # === 8-panel: horizontal front+back splits ===
            # Front: stand (top) + fall (bottom), horizontal fold line.
            # Back: stand (bottom) + fall (top), horizontal fold line.
            # front_stand.to_collar connects to combined back right edges.
            fold_frac = design['collar']['component'].get(
                'lapel_fold_fraction', {}).get('v', 0.5)
            stand_h = length_f * fold_frac
            fall_h = length_f * (1 - fold_frac)
            stand_d = depth * fold_frac
            fall_d = depth * (1 - fold_frac)

            self.front_stand = LapelStandPanel(
                f'{tag}_collar_front_stand', stand_h, length_f, depth)
            self.front_stand.translate_by([-depth * 2, height_p, 35])

            self.front_fall = SimpleLapelFallPanel(
                f'{tag}_collar_front_fall', fall_h, depth)
            self.front_fall.translate_by([-depth * 3 - 5, height_p, 35])

            self.back_stand = BackCollarStandPanel(
                f'{tag}_collar_back_stand', length_b, stand_d)
            self.back_stand.translate_by([-length_b / 2, height_p, -10])

            self.back_fall = BackCollarFallPanel(
                f'{tag}_collar_back_fall', length_b, fall_d)
            self.back_fall.translate_by([length_b / 2 + 5, height_p, -10])

            self.back_stand.interfaces['right'].set_right_wrong(True)
            self.back_fall.interfaces['right'].set_right_wrong(True)

            for e in self.back_stand.interfaces['fold_line'].edges:
                e.label = f'{tag}_fold_line'
            for e in self.back_fall.interfaces['fold_line'].edges:
                e.label = f'{tag}_fold_line'

            # -- Stitching --
            self.stitching_rules.append((
                self.front_stand.interfaces['fold_line'],
                self.front_fall.interfaces['fold_line']
            ))
            self.stitching_rules.append((
                self.back_stand.interfaces['fold_line'],
                self.back_fall.interfaces['fold_line']
            ))
            self.stitching_rules.append((
                self.front_stand.interfaces['to_collar'],
                pyg.Interface.from_multiple(
                    self.back_stand.interfaces['right'],
                    self.back_fall.interfaces['right']
                )
            ))

            self.interfaces.update({
                'back': pyg.Interface.from_multiple(
                    self.back_stand.interfaces['left'],
                    self.back_fall.interfaces['left']
                ),
                # front_fall.to_bodice excluded from neckline to avoid
                # transitive vertex collapse through the neckline loop.
                # The fall connects only via the fold-line seam.
                'bottom': pyg.Interface.from_multiple(
                    self.front_stand.interfaces['to_bodice'].set_right_wrong(True),
                    self.back_stand.interfaces['bottom'],
                )
            })

        else:
            # === Original single-panel mode (no fold) ===
            self.front = SimpleLapelPanel(
                f'{tag}_collar_front', length_f, depth)
            self.front.translate_by([-depth * 2, height_p, 35])

            self.back = StraightBandPanel(
                f'{tag}_collar_back', length_b, depth)
            self.back.translate_by([-length_b / 2, height_p, -10])
            self.back.interfaces['right'].set_right_wrong(True)

            self.stitching_rules.append((
                self.front.interfaces['to_collar'],
                self.back.interfaces['right']
            ))

            self.interfaces.update({
                'back': self.back.interfaces['left'],
                'bottom': pyg.Interface.from_multiple(
                    self.front.interfaces['to_bodice'].set_right_wrong(True),
                    self.back.interfaces['bottom'],
                )
            })

    def length(self):
        return self.interfaces['back'].edges.length()

class HoodPanel(pyg.Panel):
    """A panel for the side of the hood"""
    def __init__(self, name, f_depth, b_depth, f_length, b_length, width, in_length, depth) -> None:
        super().__init__(name)

        width = width / 2  # Panel covers one half only
        length = in_length + width / 2  

        # Bottom-back
        bottom_back_in = pyg.CurveEdge(
            [-width, -b_depth], 
            [0, 0],
            [[0.3, -0.2], [0.6, 0.2]]
        )
        bottom_back = pyg.ops.curve_match_tangents(
            bottom_back_in.as_curve(), 
            [1, 0],  # Full opening is vertically aligned
            [1, 0],
            target_len=b_length,
            return_as_edge=True, 
            verbose=self.verbose
        )
        self.edges.append(bottom_back)

        # Bottom front
        bottom_front_in = pyg.CurveEdge(
            self.edges[-1].end, 
            [width, -f_depth],
            [[0.3, 0.2], [0.6, -0.2]]
        )
        bottom_front = pyg.ops.curve_match_tangents(
            bottom_front_in.as_curve(), 
            [1, 0],  # Full opening is vertically aligned
            [1, 0],
            target_len=f_length,
            return_as_edge=True,
            verbose=self.verbose
        )
        self.edges.append(bottom_front)

        # Front-top straight section 
        self.edges.append(pyg.EdgeSeqFactory.from_verts(
            self.edges[-1].end,
            [width * 1.2, length], [width * 1.2 - depth, length]
        ))
        # Back of the hood
        self.edges.append(
            pyg.CurveEdge(
                self.edges[-1].end, 
                self.edges[0].start, 
                [[0.2, -0.5]]
            )
        )

        self.interfaces = {
            'to_other_side': pyg.Interface(self, self.edges[-2:]),
            'to_bodice': pyg.Interface(self, self.edges[0:2]).reverse()
        }

        self.rotate_by(R.from_euler('XYZ', [0, -90, 0], degrees=True))
        self.translate_by([-width, 0, 0])

class Hood2Panels(pyg.Component):

    def __init__(self, tag, body, design) -> None:
        super().__init__(f'Hood_{tag}')

        # --Projecting shapes--
        width = design['collar']['width']['v']
        f_collar = CircleNeckHalf(
            design['collar']['fc_depth']['v'],   
            design['collar']['width']['v'])
        b_collar = CircleNeckHalf(
            design['collar']['bc_depth']['v'],   
            design['collar']['width']['v'])
        
        self.interfaces = {
            'front_proj': pyg.Interface(self, f_collar),
            'back_proj': pyg.Interface(self, b_collar)
        }

        # -- Panel --
        self.panel = HoodPanel(
            f'{tag}_hood', 
            design['collar']['fc_depth']['v'],
            design['collar']['bc_depth']['v'],
            f_length=f_collar.length(),
            b_length=b_collar.length(),
            width=width,
            in_length=body['head_l'] * design['collar']['component']['hood_length']['v'],
            depth=width / 2 * design['collar']['component']['hood_depth']['v']
        ).translate_by(
            [0, body['height'] - body['head_l'] + 10, 0])

        self.interfaces.update({
            #'front': NOTE: no front interface here
            'back': self.panel.interfaces['to_other_side'],
            'bottom': self.panel.interfaces['to_bodice']
        })

    def length(self):
        return self.panel.length()


# ------ Post-serialization spec JSON transforms ------
# These functions are applied after pattern.serialize() writes the spec JSON,
# because the panels they target are deliberately serialized flat to avoid
# stitching vertex collapse cycles.  They edit the written JSON to impose the
# 3D rotations/translations needed for the fold (collar) or backward tilt
# (hood) to behave correctly during simulation.

def apply_collar_fold_rotations(folder):
    """Rotate collar fold panels in the spec JSON after serialization.

    Panels are serialized flat to avoid stitching vertex collapse cycles.
    This applies the 3D rotations needed for the fold to work in simulation.
    Only affects panels with 'collar_front_stand', 'collar_front_fall',
    'collar_back_stand', or 'collar_back_fall' in their name.
    """
    import json
    spec_files = list(folder.glob('*_specification.json'))
    if not spec_files:
        return
    spec_file = spec_files[0]
    with open(spec_file) as f:
        spec = json.load(f)

    panels = spec.get('pattern', {}).get('panels', {})

    # First pass: collect back_stand info so back_fall fold line can be aligned
    stand_info = {}
    for pname, panel in panels.items():
        if 'collar_back_stand' in pname:
            side = pname.split('_collar_back_stand')[0]
            fold_y = max(abs(v[1]) for v in panel['vertices'])
            stand_info[side] = {
                'translation': list(panel['translation']),
                'fold_y': fold_y,
            }

    modified = False
    for pname, panel in panels.items():
        rot = panel.get('rotation', [0, 0, 0])
        if 'collar_front_stand' in pname:
            rot[0] -= 90
            panel['translation'][2] = 3
        elif 'collar_front_fall' in pname:
            rot[0] += 60
            panel['translation'][2] = 15
        elif 'collar_back_stand' in pname:
            # Vertical against neck: no X rotation (local Y → 3D Y upward)
            panel['translation'][2] = -5
        elif 'collar_back_fall' in pname:
            side = pname.split('_collar_back_fall')[0]
            # Fold backward and downward from fold line
            rot[0] = -120
            if side in stand_info:
                si = stand_info[side]
                # Align fall's fold_line (local y=0) with stand's fold_line
                # (local y=fold_y, which with 0° rotation is at stand_Y + fold_y)
                panel['translation'][0] = si['translation'][0]
                panel['translation'][1] = si['translation'][1] + si['fold_y']
            panel['translation'][2] = -5
        else:
            continue
        panel['rotation'] = rot
        modified = True

    if modified:
        with open(spec_file, 'w') as f:
            json.dump(spec, f, indent=2)


def apply_hood_down(folder):
    """Tilt hood panels backward so gravity drapes them behind the neck.

    The default hood placement extends upward over the head.  A small
    X-rotation tilts the panels backward just enough for gravity to
    pull them off the head during simulation.
    """
    import json
    spec_files = list(folder.glob('*_specification.json'))
    if not spec_files:
        return
    spec_file = spec_files[0]
    with open(spec_file) as f:
        spec = json.load(f)

    modified = False
    for pname, panel in spec.get('pattern', {}).get('panels', {}).items():
        if '_hood' not in pname:
            continue

        # Build a rotation that tilts the hood top backward (-Z) and
        # downward (-Y) instead of upward (+Y).
        # The default Ry(±90°) puts local X→Z, local Y→Y.
        # We want local Y → (0, -0.5, -0.866) (60° past vertical, behind).
        is_right = 'right' in pname or pname.startswith('r')
        sign = 1 if is_right else -1

        # local Z (panel normal) faces outward (±X)
        col_z = np.array([sign, 0, 0], dtype=float)
        # local Y (hood height) points backward and down
        col_y = np.array([0, -0.5, -0.866], dtype=float)
        # local X = Y × Z
        col_x = np.cross(col_y, col_z)
        col_x /= np.linalg.norm(col_x)

        rot_mat = np.column_stack([col_x, col_y, col_z])
        euler = R.from_matrix(rot_mat).as_euler('XYZ', degrees=True)
        panel['rotation'] = euler.tolist()
        modified = True

    if modified:
        with open(spec_file, 'w') as f:
            json.dump(spec, f, indent=2)
