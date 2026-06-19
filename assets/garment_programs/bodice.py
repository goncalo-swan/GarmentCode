from copy import deepcopy
import numpy as np

import pygarment as pyg


from assets.garment_programs.base_classes import BaseBodicePanel
from assets.garment_programs import sleeves
from assets.garment_programs import collars
from assets.garment_programs import tee
from scipy.spatial.transform import Rotation as R

class BodiceFrontHalf(BaseBodicePanel):
    def __init__(self, name, body, design) -> None:
        super().__init__(name, body, design)

        m_bust = body['bust']
        m_waist = body['waist']

        # sizes   
        bust_point = body['bust_points'] / 2
        front_frac = (body['bust'] - body['back_width']) / 2 / body['bust'] 

        self.width = front_frac * m_bust
        waist = (m_waist - body['waist_back_width']) / 2
        sh_tan = np.tan(np.deg2rad(body['_shoulder_incl']))
        shoulder_incl = sh_tan * self.width
        bottom_d_width = (self.width - waist) * 2 / 3

        adjustment = sh_tan * (self.width - body['shoulder_w'] / 2)
        max_len = body['waist_over_bust_line'] - adjustment

        # side length is adjusted due to shoulder inclination
        # for the correct sleeve fitting
        fb_diff = (front_frac - (0.5 - front_frac)) * body['bust']
        back_adjustment = sh_tan * (body['back_width'] / 2 - body['shoulder_w'] / 2)
        side_len = body['waist_line'] - back_adjustment - sh_tan * fb_diff 

        self.edges.append(pyg.EdgeSeqFactory.from_verts(
            [0, 0], 
            [-self.width, 0],
            [-self.width, max_len], 
            [0, max_len + shoulder_incl]
        ))
        self.edges.close_loop()

        # Side dart
        bust_line = body['waist_line'] - body['_bust_line']
        side_d_depth = 0.75 * (self.width - bust_point)    # NOTE: calculated value 
        side_d_width = max_len - side_len
        s_edge, side_interface = self.add_dart(
            pyg.EdgeSeqFactory.dart_shape(side_d_width, side_d_depth),
            self.edges[1], 
            offset=bust_line + side_d_width / 2)
        self.edges.substitute(1, s_edge)

        # Take some fabric from the top to match the shoulder width
        s_edge[-1].end[0] += (x_upd:=self.width - body['shoulder_w'] / 2)
        s_edge[-1].end[1] += (sh_tan * x_upd)

        # Bottom dart
        b_edge, b_interface = self.add_dart(
            pyg.EdgeSeqFactory.dart_shape(bottom_d_width, 0.9 * bust_line),
            self.edges[0], 
            offset=bust_point + bottom_d_width / 2
        )
        self.edges.substitute(0, b_edge)
        # Take some fabric from side in the bottom (!: after side dart insertion)
        b_edge[-1].end[0] = - (waist + bottom_d_width) 

        # Interfaces
        self.interfaces = {
            'outside':  pyg.Interface(self, side_interface),   # side_interface,    # pyp.Interface(self, [side_interface]),  #, self.edges[-3]]),
            'inside': pyg.Interface(self, self.edges[-1]),
            'shoulder': pyg.Interface(self, self.edges[-2]),
            'bottom': pyg.Interface(self, b_interface),

            # Reference to the corner for sleeve and collar projections
            'shoulder_corner': pyg.Interface(
                self, [self.edges[-3], self.edges[-2]]),
            'collar_corner': pyg.Interface(
                self, [self.edges[-2], self.edges[-1]])
        }
  
        # default placement
        self.translate_by([0, body['height'] - body['head_l'] - max_len - shoulder_incl, 0])


class BodiceBackHalf(BaseBodicePanel):
    """Panel for the back of basic fitted bodice block"""

    def __init__(self, name, body, design) -> None:
        super().__init__(name, body, design)

        # Overall measurements        
        self.width = body['back_width'] / 2
        waist = body['waist_back_width'] / 2  
        # NOTE: no inclination on the side, since there is not much to begin with
        waist_width = self.width if waist < self.width else waist  
        shoulder_incl = (sh_tan:=np.tan(np.deg2rad(body['_shoulder_incl']))) * self.width

        # Adjust to make sure length is measured from the shoulder
        # and not the de-fact side of the garment
        back_adjustment = sh_tan * (self.width - body['shoulder_w'] / 2)
        length = body['waist_line'] - back_adjustment

        # Base edge loop
        edge_0 = pyg.CurveEdgeFactory.curve_from_tangents(
            start=[0, shoulder_incl / 4],  # back a little shorter 
            end=[-waist_width, 0],
            target_tan0=[-1, 0]
        )
        self.edges.append(edge_0)
        self.edges.append(pyg.EdgeSeqFactory.from_verts(
            edge_0.end,
            [-self.width, body['waist_line'] - body['_bust_line']],  # from the bottom
            [-self.width, length],   
            [0, length + shoulder_incl],   # Add some fabric for the neck (inclination of shoulders)
        ))
        self.edges.close_loop()
        
        # Take some fabric from the top to match the shoulder width
        self.interfaces = {
            'outside': pyg.Interface(
                self, [self.edges[1], self.edges[2]]),
            'inside': pyg.Interface(self, self.edges[-1]),
            'shoulder': pyg.Interface(self, self.edges[-2]),
            'bottom': pyg.Interface(self, self.edges[0]),
            # Reference to the corners for sleeve and collar projections
            'shoulder_corner': pyg.Interface(
                self, pyg.EdgeSequence(self.edges[-3], self.edges[-2])),
            'collar_corner': pyg.Interface(
                self, pyg.EdgeSequence(self.edges[-2], self.edges[-1]))
        }

        # Bottom dart as cutout -- for straight line
        if waist < self.get_width(self.edges[2].end[1] - self.edges[2].start[1]):
            w_diff = waist_width - waist
            side_adj = 0  if w_diff < 4 else w_diff / 6  # NOTE: don't take from sides if the difference is too small
            bottom_d_width = w_diff - side_adj
            bottom_d_width /= 2   # double darts
            bottom_d_depth = 1. * (length - body['_bust_line'])  # calculated value
            bottom_d_position = body['bum_points'] / 2

            # TODOLOW Avoid hardcoding for matching with the bottoms?
            dist = bottom_d_position * 0.5  # Dist between darts -> dist between centers
            b_edge, b_interface = self.add_dart(
                pyg.EdgeSeqFactory.dart_shape(bottom_d_width, 0.9 * bottom_d_depth),
                self.edges[0], 
                offset=bottom_d_position + dist / 2 + bottom_d_width + bottom_d_width / 2,
            )
            b_edge, b_interface = self.add_dart(
                pyg.EdgeSeqFactory.dart_shape(bottom_d_width, bottom_d_depth),
                b_edge[0], 
                offset=bottom_d_position - dist / 2 + bottom_d_width / 2,
                edge_seq=b_edge,
                int_edge_seq=b_interface,
            )

            self.edges.substitute(0, b_edge)
            self.interfaces['bottom'] = pyg.Interface(self, b_interface)

            # Remove fabric from the sides if the diff is big enough
            b_edge[-1].end[0] += side_adj

        # default placement
        self.translate_by([0, body['height'] - body['head_l'] - length - shoulder_incl, 0])

    def get_width(self, level):
        return self.width

class BodiceHalf(pyg.Component):
    """Definition of a half of an upper garment with sleeves and collars"""

    def __init__(self, name, body, design, fitted=True) -> None:
        super().__init__(name)

        design = deepcopy(design)   # Recalculate freely!

        # Torso
        if fitted:
            self.ftorso = BodiceFrontHalf(
                f'{name}_ftorso', body, design).translate_by([0, 0, 30])
            self.btorso = BodiceBackHalf(
                f'{name}_btorso', body, design).translate_by([0, 0, -25])
        else:
            self.ftorso = tee.TorsoFrontHalfPanel(
                f'{name}_ftorso', body, design).translate_by([0, 0, 30])
            self.btorso = tee.TorsoBackHalfPanel(
                f'{name}_btorso', body, design).translate_by([0, 0, -25])

        # Interfaces
        self.interfaces.update({
            'f_bottom': self.ftorso.interfaces['bottom'],
            'b_bottom': self.btorso.interfaces['bottom'],
            'front_in': self.ftorso.interfaces['inside'],
            'back_in': self.btorso.interfaces['inside']
        })

        # Sleeves/collar cuts
        self.sleeve = None
        self.collar_comp = None
        self.eval_dep_params(body, design)

        if design['shirt']['strapless']['v'] and fitted:  # NOTE: Strapless design only for fitted tops
            self.make_strapless(body, design)
        else:
            # Sleeves and collars
            self.add_sleeves(name, body, design)
            self.add_collars(name, body, design)
            self.stitching_rules.append((
                self.ftorso.interfaces['shoulder'],
                self.btorso.interfaces['shoulder']
            ))  # tops

        # Main connectivity
        self.stitching_rules.append((
            self.ftorso.interfaces['outside'], self.btorso.interfaces['outside']))   # sides

    def eval_dep_params(self, body, design):

        # Sleeves
        # NOTE assuming the vertical side is the first argument
        max_cwidth = self.ftorso.interfaces['shoulder_corner'].edges[0].length() - 1  # cm
        min_cwidth = body['_armscye_depth']
        v = design['sleeve']['connecting_width']['v']
        design['sleeve']['connecting_width']['v'] = min(min_cwidth + min_cwidth * v, max_cwidth)

        # Collars
        # NOTE: Assuming the first is the top edge
        # Width
        # TODOLOW What if sleeve inclination is variable?
        # NOTE: Back panel is more narrow, so using it
        max_w = body['_base_sleeve_balance'] - 2  # 1 cm from default sleeve
        min_w = body['neck_w']

        cw_v = design['collar']['width']['v']
        if cw_v >= 0:
            # Manual linear interpolation: allows extrapolation beyond max_w
            # for off-shoulder / very wide necklines (v > 1).
            design['collar']['width']['v'] = width = min_w + cw_v * (max_w - min_w)
        else:
            design['collar']['width']['v'] = width = pyg.utils.lin_interpolation(0, min_w, 1 + cw_v)

        # Depth
        # Collar depth is given w.r.t. length.
        # adjust for the shoulder inclination
        tg = np.tan(np.deg2rad(body['_shoulder_incl']))
        f_depth_adj = tg * (self.ftorso.get_width(0) - width / 2)
        b_depth_adj = tg * (self.btorso.get_width(0) - width / 2)

        max_f_len = self.ftorso.interfaces['collar_corner'].edges[1].length() - tg * self.ftorso.get_width(0) - 1  # cm
        max_b_len = self.btorso.interfaces['collar_corner'].edges[1].length() - tg * self.btorso.get_width(0) - 1  # cm

        design['collar']['f_strapless_depth'] = {}
        design['collar']['f_strapless_depth']['v'] = min(
            design['collar']['fc_depth']['v'] * body['_bust_line'], 
            max_f_len)
        design['collar']['fc_depth']['v'] = design['collar']['f_strapless_depth']['v'] + f_depth_adj
        

        design['collar']['b_strapless_depth'] = {}
        design['collar']['b_strapless_depth']['v'] =  min(
            design['collar']['bc_depth']['v'] * body['_bust_line'], 
            max_b_len)
        design['collar']['bc_depth']['v'] = design['collar']['b_strapless_depth']['v'] + b_depth_adj 

    def add_sleeves(self, name, body, design):
        self.sleeve = sleeves.Sleeve(
            name, body, design, 
            front_w=self.ftorso.get_width,
            back_w=self.btorso.get_width
        )

        _, f_sleeve_int = pyg.ops.cut_corner(
            self.sleeve.interfaces['in_front_shape'].edges, 
            self.ftorso.interfaces['shoulder_corner'], 
            verbose=self.verbose
        )
        _, b_sleeve_int = pyg.ops.cut_corner(
            self.sleeve.interfaces['in_back_shape'].edges, 
            self.btorso.interfaces['shoulder_corner'],
            verbose=self.verbose
        )

        if not design['sleeve']['sleeveless']['v']:  
            # Ordering
            bodice_sleeve_int = pyg.Interface.from_multiple(
                f_sleeve_int.reverse(with_edge_dir_reverse=True),
                b_sleeve_int.reverse(),
            )    
            self.stitching_rules.append((
                self.sleeve.interfaces['in'], 
                bodice_sleeve_int
            ))

            # NOTE: This is a heuristic tuned for arm poses 30 deg-60 deg 
            # used in the dataset
            # FIXME Needs a better general solution
            gap = -1 - body['arm_pose_angle'] / 10
            self.sleeve.place_by_interface(
                self.sleeve.interfaces['in'], 
                bodice_sleeve_int, 
                gap=gap,   
                alignment='top',
            )

        # Add edge labels
        f_sleeve_int.edges.propagate_label(f'{self.name}_armhole')
        b_sleeve_int.edges.propagate_label(f'{self.name}_armhole')
    
    def add_collars(self, name, body, design):
        # Front
        collar_type = getattr(
            collars, 
            str(design['collar']['component']['style']['v']), 
            collars.NoPanelsCollar
            )
        
        self.collar_comp = collar_type(name, body, design)
        
        # Project shape
        _, fc_interface = pyg.ops.cut_corner(
            self.collar_comp.interfaces['front_proj'].edges, 
            self.ftorso.interfaces['collar_corner'],
            verbose=self.verbose
        )
        _, bc_interface = pyg.ops.cut_corner(
            self.collar_comp.interfaces['back_proj'].edges, 
            self.btorso.interfaces['collar_corner'],
            verbose=self.verbose
        )

        # Add stitches/interfaces
        if 'bottom' in self.collar_comp.interfaces:
            self.stitching_rules.append((
                pyg.Interface.from_multiple(fc_interface, bc_interface), 
                self.collar_comp.interfaces['bottom']
            ))

        # Upd front interfaces accordingly
        if 'front' in self.collar_comp.interfaces:
            self.interfaces['front_collar'] = self.collar_comp.interfaces['front']
            self.interfaces['front_in'] = pyg.Interface.from_multiple(
                self.ftorso.interfaces['inside'], self.interfaces['front_collar']
            )
        if 'back' in self.collar_comp.interfaces:
            self.interfaces['back_collar'] = self.collar_comp.interfaces['back']
            self.interfaces['back_in'] = pyg.Interface.from_multiple(
                self.btorso.interfaces['inside'], self.interfaces['back_collar']
            )
        
        # Add edge labels.
        # NOTE: only the FRONT collar edge gets the `{name}_collar` label.
        # The back collar edge intentionally does not — when both halves were
        # labelled, garment.py's X-pull attachment (right→-neck_w/2,
        # left→+neck_w/2) pulled the back-center seam vertices in opposite
        # directions and bunched the seam. The front-collar X-pull is the one
        # we actually need for lapel positioning; the back collar settles
        # cleanly via cloth dynamics + the fold-line Y-pin.
        fc_interface.edges.propagate_label(f'{self.name}_collar')

        # Label free lapel edges for fold attachment constraint.
        # Only label edges NOT involved in stitching (to avoid label conflicts).
        if hasattr(self.collar_comp, 'front_fall'):
            # Split-panel fold mode: label free edges of the fall panel
            fall_panel = self.collar_comp.front_fall
            stitched = set()
            for iface_name in ('fold_line', 'to_bodice'):
                if iface_name in fall_panel.interfaces:
                    for e in fall_panel.interfaces[iface_name].edges:
                        stitched.add(id(e))
            for e in fall_panel.edges:
                if id(e) not in stitched:
                    e.label = f'{self.name}_lapel'
        elif hasattr(self.collar_comp, 'front'):
            # Lapel-style components expose a front Panel with `to_collar` /
            # `to_bodice` interfaces. Other components that just happen to
            # have a `front` attribute (e.g. Turtle's StraightBandPanel) use
            # different interface names and must NOT get `_lapel` labels —
            # doing so overwrites collar-stitch labels (e.g. `front.left`)
            # already set by propagate_label above, causing a stitch mismatch.
            front_panel = self.collar_comp.front
            if ('to_collar' in front_panel.interfaces
                    or 'to_bodice' in front_panel.interfaces):
                stitched = set()
                if 'to_collar' in front_panel.interfaces:
                    for e in front_panel.interfaces['to_collar'].edges:
                        stitched.add(id(e))
                if 'to_bodice' in front_panel.interfaces:
                    for e in front_panel.interfaces['to_bodice'].edges:
                        stitched.add(id(e))
                for e in front_panel.edges:
                    if id(e) not in stitched:
                        e.label = f'{self.name}_lapel'

    def make_strapless(self, body, design):

        out_depth = design['sleeve']['connecting_width']['v']
        f_in_depth = design['collar']['f_strapless_depth']['v']
        b_in_depth = design['collar']['b_strapless_depth']['v']

        # Shoulder adjustment for the back
        # TODOLOW Shoulder adj evaluation should be a function
        shoulder_angle = np.deg2rad(body['_shoulder_incl'])
        sleeve_balance = body['_base_sleeve_balance'] / 2
        back_w = self.btorso.get_width(0)
        shoulder_adj = np.tan(shoulder_angle) * (back_w - sleeve_balance)
        out_depth -= shoulder_adj

        # Upd back
        self._adjust_top_level(self.btorso, out_depth, b_in_depth)

        # Front depth determined by ~compensating for lenght difference
        len_back = self.btorso.interfaces['outside'].edges.length()
        len_front = self.ftorso.interfaces['outside'].edges.length()
        self._adjust_top_level(self.ftorso, out_depth, f_in_depth, target_remove=(len_front - len_back))
        
        # Placement
        # NOTE: The commented line places the top a bit higher, increasing the chanced of correct drape
        # Surcumvented by attachment constraint, so removed for nicer alignment in asymmetric garments
        # self.translate_by([0, out_depth - body['_armscye_depth'] * 0.75, 0])   # adjust for better localisation

        # Add a label
        self.ftorso.interfaces['shoulder'].edges.propagate_label('strapless_top')
        self.btorso.interfaces['shoulder'].edges.propagate_label('strapless_top')


    def _adjust_top_level(self, panel, out_level, in_level, target_remove=None):
        """Crops the top of the bodice front/back panel for strapless style

            * out_length_diff -- if set, determined the length difference that should be compensates
            after cutting the depth
        """
        # TODOLOW Should this be the panel's function?

        panel_top = panel.interfaces['shoulder'].edges[0]
        min_y = min(panel_top.start[1], panel_top.end[1])  

        # Order vertices
        ins, out = panel_top.start, panel_top.end
        if panel_top.start[1] < panel_top.end[1]:
            ins, out = out, ins
    
        # Inside is a simple vertical line and can be adjusted by chaning Y value
        ins[1] = min_y - in_level

        # Outside could be inclined, so needs further calculations
        outside_edge = panel.interfaces['outside'].edges[-1]
        bot, top = outside_edge.start, outside_edge.end
        if bot is out:
            bot, top = top, bot
        
        if target_remove is not None:
            # Adjust the depth to remove this length exactly
            angle_sin = abs(out[1] - bot[1]) / outside_edge.length()   
            curr_remove = out_level / angle_sin
            length_diff = target_remove - curr_remove
            adjustment = length_diff * angle_sin
            out_level += adjustment
        
        angle_cotan = abs(out[0] - bot[0]) / abs(out[1] - bot[1])
        out[0] -= out_level * angle_cotan  
        out[1] = min_y - out_level


    def length(self):
        return self.btorso.length()

class Shirt(pyg.Component):
    """Panel for the front of upper garments with darts to properly fit it to
    the shape"""

    def __init__(self, body, design, fitted=False) -> None:
        name_with_params = f"{self.__class__.__name__}"
        super().__init__(name_with_params)

        design = self.eval_dep_params(design)

        self.right = BodiceHalf(f'right', body, design, fitted=fitted)
        self.left = BodiceHalf(
            f'left', body, 
            design['left'] if design['left']['enable_asym']['v'] else design, 
            fitted=fitted).mirror()

        self.stitching_rules.append((self.right.interfaces['front_in'],
                                     self.left.interfaces['front_in']))
        self.stitching_rules.append((self.right.interfaces['back_in'],
                                     self.left.interfaces['back_in']))

        # Adjust interface ordering for correct connectivity
        self.interfaces = {   # Bottom connection
            'bottom': pyg.Interface.from_multiple(
                self.right.interfaces['f_bottom'].reverse(),
                self.left.interfaces['f_bottom'],
                self.left.interfaces['b_bottom'].reverse(),
                self.right.interfaces['b_bottom'],)
        }

    def eval_dep_params(self, design):
        # NOTE: Support for full collars with partially strapless top
        # or combination of paneled collar styles
        # requres further development
        # TODOLOW enable this one to work
        if design['left']['enable_asym']['v']:
            # Force no collars since they are not compatible with each other
            design = deepcopy(design)
            design['collar']['component']['style']['v'] = None
            design['left']['collar']['component'] = dict(style=dict(v=None))
            
            # Left-right design compatibility 
            design['left']['shirt'].update(length={})
            design['left']['shirt']['length']['v'] = design['shirt']['length']['v']
            
            design['left']['collar'].update(fc_depth={}, bc_depth={})
            design['left']['collar']['fc_depth']['v'] = design['collar']['fc_depth']['v']
            design['left']['collar']['bc_depth']['v'] = design['collar']['bc_depth']['v']

        return design

    def length(self):
        return self.right.length()

class FittedShirt(Shirt):
    """Creates fitted shirt

        NOTE: Separate class is used for selection convenience.
        Even though most of the processing is the same
        (hence implemented with the same components except for panels),
        design parametrization differs significantly.
        With that, we decided to separate the top level names
    """
    def __init__(self, body, design) -> None:
        super().__init__(body, design, fitted=True)


class ButtonDownShirt(Shirt):
    """Upper garment with open front (button-down style).

    The center-front edges are NOT stitched together, creating a
    front opening. Optional placket_width adds overlap fabric.
    """
    def __init__(self, body, design, fitted=False) -> None:
        name_with_params = f"{self.__class__.__name__}"
        pyg.Component.__init__(self, name_with_params)

        design = self.eval_dep_params(design)

        self.right = BodiceHalf(f'right', body, design, fitted=fitted)
        self.left = BodiceHalf(
            f'left', body,
            design['left'] if design['left']['enable_asym']['v'] else design,
            fitted=fitted).mirror()

        # Placket extension — shift center-front edge outward for overlap
        pw = design['shirt']['placket_width']['v']
        if pw > 0:
            for edge in self.right.ftorso.interfaces['inside'].edges:
                edge.start[0] += pw
                edge.end[0] += pw
            for edge in self.left.ftorso.interfaces['inside'].edges:
                edge.start[0] -= pw
                edge.end[0] -= pw

        # Back seam — always stitched
        self.stitching_rules.append((self.right.interfaces['back_in'],
                                     self.left.interfaces['back_in']))

        # Front closure: 0.0 = fully open, 1.0 = fully closed
        closure = design['shirt']['closure']['v']
        if closure >= 1.0:
            self.stitching_rules.append((self.right.interfaces['front_in'],
                                         self.left.interfaces['front_in']))
        elif closure > 0:
            # Optional middle-band closure (e.g. a blazer buttoned only at the
            # waist, lapel + hem open): closure_start shifts the closed band up
            # from the hem. Default 0 = bottom-up.
            closure_start = design['shirt'].get('closure_start', {}).get('v', 0.0)
            right_partial = self._partial_front_in(self.right, closure, closure_start)
            left_partial = self._partial_front_in(self.left, closure, closure_start)
            if right_partial is not None and left_partial is not None:
                self.stitching_rules.append((right_partial, left_partial))

        self.interfaces = {
            'bottom': pyg.Interface.from_multiple(
                self.right.interfaces['f_bottom'].reverse(),
                self.left.interfaces['f_bottom'],
                self.left.interfaces['b_bottom'].reverse(),
                self.right.interfaces['b_bottom'],)
        }

    @staticmethod
    def _partial_front_in(half, closure, closure_start=0.0):
        """Return an Interface covering a band of front_in measured from the
        bottom (hem): the band spans [closure_start, closure_start+closure] of
        the total front length.

        closure_start=0 closes the bottom `closure` fraction (the original
        behaviour). closure_start>0 closes a middle band, leaving both the
        bottom (hem) and the top (lapel/neckline) open — e.g. a blazer
        buttoned only at the waist.
        """
        front_in = half.interfaces['front_in']
        total_len = front_in.edges.length()
        band_lo = total_len * max(closure_start, 0.0)
        band_hi = total_len * min(closure_start + closure, 1.0)

        # Detect direction: compare Y of first and last vertex
        first_y = front_in.edges[0].start[1]
        last_y = front_in.edges[-1].end[1]
        bottom_is_at_end = first_y > last_y  # top→bottom: bottom at end

        # Snapshot (edge, panel, c0, c1, len) in BOTTOM→TOP order so we can
        # mutate front_in.edges (subdivision) without invalidating indices.
        order = list(range(len(front_in.edges)))
        if bottom_is_at_end:
            order = order[::-1]
        spans = []
        cumulative = 0.0
        for i in order:
            edge = front_in.edges[i]
            panel = front_in.panel[i]
            L = edge.length()
            spans.append((edge, panel, cumulative, cumulative + L, L))
            cumulative += L

        eps = 1e-3
        collected = []  # (panel, edge)
        for edge, panel, c0, c1, L in spans:
            lo = max(c0, band_lo)
            hi = min(c1, band_hi)
            if hi - lo <= 0.1:
                continue  # edge outside the band
            # edge-local fractions in bottom→top direction
            f_lo = (lo - c0) / L
            f_hi = (hi - c0) / L
            # map to the edge's own start→end direction
            if bottom_is_at_end:
                e_lo, e_hi = 1.0 - f_hi, 1.0 - f_lo
            else:
                e_lo, e_hi = f_lo, f_hi

            if e_lo < eps and e_hi > 1.0 - eps:
                collected.append((panel, edge))
                continue
            if e_lo < eps:
                parts = edge.subdivide_len([e_hi, 1.0 - e_hi]); keep = parts[0]
            elif e_hi > 1.0 - eps:
                parts = edge.subdivide_len([e_lo, 1.0 - e_lo]); keep = parts[1]
            else:
                parts = edge.subdivide_len([e_lo, e_hi - e_lo, 1.0 - e_hi]); keep = parts[1]
            panel.edges.substitute(edge, parts)
            front_in.substitute(edge, parts, [panel] * len(parts))
            collected.append((panel, keep))

        if not collected:
            return None
        if len(collected) == 1:
            return pyg.Interface(collected[0][0], collected[0][1])
        return pyg.Interface.from_multiple(*[pyg.Interface(p, e) for p, e in collected])

    def length(self):
        return self.right.length()


class FittedButtonDownShirt(ButtonDownShirt):
    """Fitted variant of the button-down shirt with darts."""
    def __init__(self, body, design) -> None:
        super().__init__(body, design, fitted=True)
