from copy import deepcopy
import numpy as np

import pygarment as pyg
from assets.garment_programs.base_classes import BaseBottoms
from assets.garment_programs import bands


class PantPanel(pyg.Panel):
    def __init__(
            self, name, body, design,
            length,
            waist,
            hips,
            hips_depth,
            crotch_width,
            dart_position,
            match_top_int_to=None,
            hipline_ext=1,
            double_dart=False,
            thigh_width=None,
            knee_width=None,
            knee_y=None,
            thigh_y=None) -> None:
        """
            Basic pant panel with option to be fitted (with darts)
        """
        super().__init__(name)

        flare = body['leg_circ'] * (design['flare']['v']  - 1) / 4
        hips_depth = hips_depth * hipline_ext

        hip_side_incl = np.deg2rad(body['_hip_inclination'])
        dart_depth = hips_depth * 0.8

        # Crotch cotrols
        crotch_depth_diff =  body['crotch_hip_diff']
        crotch_extention = crotch_width

        # eval pants shape
        # TODO Return ruffle opportunity?

        # amount of extra fabric at waist
        w_diff = hips - waist   # Assume its positive since waist is smaller then hips
        # We distribute w_diff among the side angle and a dart
        hw_shift = np.tan(hip_side_incl) * hips_depth
        # Small difference
        if hw_shift > w_diff:
            hw_shift = w_diff

        # Check if we should use thigh/knee subcurves
        use_subcurves = (thigh_width is not None and knee_width is not None
                         and knee_y is not None and thigh_y is not None)

        # --- Edges definition ---
        if use_subcurves:
            # --- Outseam with 3 subcurves: ankle→knee, knee→thigh, thigh→hip ---
            ankle_pt = [-flare, 0]
            hip_pt = [0, length]

            # Outseam X at intermediate heights
            knee_out_x = -flare * (1 - knee_y / length)
            # Thigh outseam: set so panel width at crotch_y = thigh_width.
            # At crotch_y the inseam is at hips+crotch_ext (crotch endpoint),
            # and outseam ≈ thigh_out_x (only 2cm above on the Bezier).
            thigh_out_x = (hips + crotch_extention) - thigh_width

            knee_pt_out = [knee_out_x, knee_y]
            thigh_pt_out = [thigh_out_x, thigh_y]

            # Tangent at junctions: catmull-rom style (direction from prev to next point)
            knee_out_tan = np.array([thigh_pt_out[0] - ankle_pt[0],
                                     thigh_pt_out[1] - ankle_pt[1]])
            knee_out_tan = knee_out_tan / np.linalg.norm(knee_out_tan)

            thigh_out_tan = np.array([hip_pt[0] - knee_pt_out[0],
                                      hip_pt[1] - knee_pt_out[1]])
            thigh_out_tan = thigh_out_tan / np.linalg.norm(thigh_out_tan)

            right_ankle_to_knee = pyg.CurveEdgeFactory.curve_from_tangents(
                ankle_pt, knee_pt_out,
                target_tan1=knee_out_tan,
                initial_guess=[0.5, 0]
            )
            right_knee_to_thigh = pyg.CurveEdgeFactory.curve_from_tangents(
                knee_pt_out, thigh_pt_out,
                target_tan0=knee_out_tan,
                target_tan1=thigh_out_tan,
                initial_guess=[0.5, 0]
            )
            right_thigh_to_hip = pyg.CurveEdgeFactory.curve_from_tangents(
                thigh_pt_out, hip_pt,
                target_tan0=thigh_out_tan,
                target_tan1=np.array([0, 1]),  # vertical at hip (original behavior)
                initial_guess=[0.5, 0]
            )

            right_top = pyg.CurveEdgeFactory.curve_from_tangents(
                right_thigh_to_hip.end,
                [hw_shift, length + hips_depth],
                target_tan0=np.array([0, 1]),
                initial_guess=[0.5, 0]
            )
        else:
            # Original single-curve outseam
            if pyg.utils.close_enough(design['flare']['v'], 1):  # skip optimization
                right_bottom = pyg.Edge(
                    [-flare, 0],
                    [0, length]
                )
            else:
                right_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
                    [-flare, 0],
                    [0, length],
                    target_tan1=np.array([0, 1]),
                    # initial guess places control point closer to the hips
                    initial_guess=[0.75, 0]
                )
            right_top = pyg.CurveEdgeFactory.curve_from_tangents(
                right_bottom.end,
                [hw_shift, length + hips_depth],
                target_tan0=np.array([0, 1]),
                initial_guess=[0.5, 0]
            )

        top = pyg.Edge(
            right_top.end,
            [w_diff + waist, length + hips_depth]
        )

        crotch_top = pyg.Edge(
            top.end,
            [hips, length + 0.45 * hips_depth]  # A bit higher than hip line
            # NOTE: The point should be lower than the minimum rise value (0.5)
        )
        crotch_bottom = pyg.CurveEdgeFactory.curve_from_tangents(
            crotch_top.end,
            [hips + crotch_extention, length - crotch_depth_diff],
            target_tan0=np.array([0, -1]),
            target_tan1=np.array([1, 0]),
            initial_guess=[0.5, -0.5]
        )

        if use_subcurves:
            # --- Inseam with 3 subcurves: crotch→thigh, thigh→knee, knee→ankle ---
            # Inseam X at intermediate heights = outseam_x + target_width
            thigh_in_x = thigh_out_x + thigh_width
            knee_in_x = knee_out_x + knee_width

            thigh_pt_in = [thigh_in_x, thigh_y]
            knee_pt_in = [knee_in_x, knee_y]

            # Ankle inseam point (same formula as original)
            y_ankle = min(0, length - crotch_depth_diff * 1.5)
            ankle_in_x = crotch_bottom.end[0] - 2 + flare
            ankle_pt_in = [ankle_in_x, y_ankle]

            # Inseam tangent at knee junction (catmull-rom: prev→next)
            knee_in_tan = np.array([ankle_pt_in[0] - crotch_bottom.end[0],
                                    ankle_pt_in[1] - crotch_bottom.end[1]])
            knee_in_tan = knee_in_tan / np.linalg.norm(knee_in_tan)

            # 3 inseam subcurves: crotch→thigh, thigh→knee, knee→ankle
            # Crotch→thigh is always a STRAIGHT edge: a Bezier here bumps
            # above the crotch line because the catmull-rom tangent (pointing
            # steeply downward toward the knee) forces the control point way
            # above the start, causing self-intersection with the crotch curve.
            left_crotch_to_thigh = pyg.Edge(
                crotch_bottom.end, thigh_pt_in)

            # Tangent at thigh: direction of the straight crotch→thigh edge
            thigh_edge_dir = np.array([thigh_pt_in[0] - crotch_bottom.end[0],
                                       thigh_pt_in[1] - crotch_bottom.end[1]])
            thigh_edge_len = np.linalg.norm(thigh_edge_dir)
            thigh_in_tan = (thigh_edge_dir / thigh_edge_len
                            if thigh_edge_len > 0.01 else np.array([0, -1]))

            left_thigh_to_knee = pyg.CurveEdgeFactory.curve_from_tangents(
                thigh_pt_in, knee_pt_in,
                target_tan0=thigh_in_tan,
                target_tan1=knee_in_tan,
                initial_guess=[0.5, 0]
            )
            left_knee_to_ankle = pyg.CurveEdgeFactory.curve_from_tangents(
                knee_pt_in, ankle_pt_in,
                target_tan0=knee_in_tan,
                target_tan1=[flare, y_ankle - crotch_bottom.end[1]],
                initial_guess=[0.3, 0]
            )
            inseam_edges = pyg.EdgeSequence(
                left_crotch_to_thigh, left_thigh_to_knee, left_knee_to_ankle)

            self.edges = pyg.EdgeSequence(
                right_ankle_to_knee, right_knee_to_thigh, right_thigh_to_hip,
                right_top, top, crotch_top, crotch_bottom,
                *inseam_edges
            ).close_loop()
            bottom = self.edges[-1]

            # Default placement
            self.set_pivot(crotch_bottom.end)
            self.translation = [-0.5, - hips_depth - crotch_depth_diff + 5, 0]

            # Out interfaces
            self.interfaces = {
                'outside': pyg.Interface(
                    self,
                    pyg.EdgeSequence(right_ankle_to_knee, right_knee_to_thigh,
                                     right_thigh_to_hip, right_top),
                    ruffle=[1, 1, 1, hipline_ext]),
                'crotch': pyg.Interface(self, pyg.EdgeSequence(crotch_top, crotch_bottom)),
                'inside': pyg.Interface(self, inseam_edges),
                'bottom': pyg.Interface(self, bottom)
            }
        else:
            left = pyg.CurveEdgeFactory.curve_from_tangents(
                crotch_bottom.end,
                [
                    # NOTE "Magic value" (-2 cm) which we use to define default width:
                    #   just a little behing the crotch point
                    # NOTE: Ensuring same distance from the crotch point in both
                    #   front and back for matching curves
                    crotch_bottom.end[0] - 2 + flare,
                    # NOTE: The inside edge either matches the length of the outside (0, normal case)
                    # or when the inteded length is smaller than crotch depth,
                    # inside edge covers of the inside leg a bit below the crotch (panties-like shorts)
                    y:=min(0, length - crotch_depth_diff * 1.5)
                ],
                target_tan1=[flare, y - crotch_bottom.end[1]],
                initial_guess=[0.3, 0]
            )

            self.edges = pyg.EdgeSequence(
                right_bottom, right_top, top, crotch_top, crotch_bottom, left
                ).close_loop()
            bottom = self.edges[-1]

            # Default placement
            self.set_pivot(crotch_bottom.end)
            self.translation = [-0.5, - hips_depth - crotch_depth_diff + 5, 0]

            # Out interfaces (easier to define before adding a dart)
            self.interfaces = {
                'outside': pyg.Interface(
                    self,
                    pyg.EdgeSequence(right_bottom, right_top),
                    ruffle=[1, hipline_ext]),
                'crotch': pyg.Interface(self, pyg.EdgeSequence(crotch_top, crotch_bottom)),
                'inside': pyg.Interface(self, left),
                'bottom': pyg.Interface(self, bottom)
            }

        # Add top dart
        # NOTE: Ruffle indicator to match to waistline proportion for correct balance line matching
        dart_width = w_diff - hw_shift  
        if w_diff > hw_shift:
            top_edges, int_edges = self.add_darts(
                top, dart_width, dart_depth, dart_position, double_dart=double_dart)
            self.interfaces['top'] = pyg.Interface(
                self, int_edges, 
                ruffle=waist / match_top_int_to if match_top_int_to is not None else 1.
            ) 
            self.edges.substitute(top, top_edges)
        else:
            self.interfaces['top'] = pyg.Interface(
                self, top, 
                ruffle=waist / match_top_int_to if match_top_int_to is not None else 1.
        ) 
        
        

    def add_darts(self, top, dart_width, dart_depth, dart_position, double_dart=False):
        
        if double_dart:
            # TODOLOW Avoid hardcoding for matching with the top?
            dist = dart_position * 0.5  # Dist between darts -> dist between centers
            offsets_mid = [
                - (dart_position + dist / 2 + dart_width / 2 + dart_width / 4),   
                - (dart_position - dist / 2) - dart_width / 4,
            ]

            darts = [
                pyg.EdgeSeqFactory.dart_shape(dart_width / 2, dart_depth * 0.9), # smaller
                pyg.EdgeSeqFactory.dart_shape(dart_width / 2, dart_depth)
            ]
        else:
            offsets_mid = [
                - dart_position - dart_width / 2,
            ]
            darts = [
                pyg.EdgeSeqFactory.dart_shape(dart_width, dart_depth)
            ]
        top_edges, int_edges = pyg.EdgeSequence(top), pyg.EdgeSequence(top)

        for off, dart in zip(offsets_mid, darts):
            left_edge_len = top_edges[-1].length()
            top_edges, int_edges = self.add_dart(
                dart,
                top_edges[-1],
                offset=left_edge_len + off,
                edge_seq=top_edges, 
                int_edge_seq=int_edges
            )

        return top_edges, int_edges
        

class PantsHalf(BaseBottoms):
    def __init__(self, tag, body, design, rise=None) -> None:
        super().__init__(body, design, tag, rise=rise)
        design = design['pants']
        self.rise = design['rise']['v'] if rise is None else rise
        waist, hips_depth, waist_back = self.eval_rise(self.rise)

        # NOTE: min value = full sum > leg curcumference
        # Max: pant leg falls flat from the back
        # Mostly from the back side
        # => This controls the foundation width of the pant
        # width_v scales the entire panel hip width (not just crotch extension)
        width_v = design['width']['v']
        min_ext = body['leg_circ'] - body['hips'] / 2  + 5  # 2 inch ease: from pattern making book
        front_hip = (body['hips'] - body['hip_back_width']) / 2 * width_v
        crotch_extention = min_ext * width_v
        if 'front_crotch_fraction' in design and design['front_crotch_fraction'].get('v') is not None:
            front_extention = crotch_extention * float(design['front_crotch_fraction']['v'])
        else:
            front_extention = front_hip / 4    # From pattern making book
        back_extention = crotch_extention - front_extention

        length, cuff_len = design['length']['v'], design['cuff']['cuff_len']['v']
        if design['cuff']['type']['v']:
            if length - cuff_len < design['length']['range'][0]:   # Min length from paramss
                # Cannot be longer then a pant
                cuff_len = length - design['length']['range'][0]
            # Include the cuff into the overall length,
            # unless the requested length is too short to fit the cuff
            # (to avoid negative length)
            length -= cuff_len
        length *= body['_leg_length']
        cuff_len *= body['_leg_length']

        # For rise > 1.0 (high-rise), offset panel upward so waistband
        # sits above the natural waist line (e.g. belly button level)
        rise_offset = max(0, (self.rise - 1.0)) * body['hips_line']

        # Thigh/knee width controls (if body measurements available)
        back_hip = body['hip_back_width'] / 2 * width_v
        leg_shape_kwargs = {}
        if 'thigh_circ' in body and 'knee_circ' in body:
            thigh_v = design.get('thigh', {}).get('v', 1.0)
            knee_v = design.get('knee', {}).get('v', 1.0)

            thigh_circ = body['thigh_circ'] * thigh_v
            knee_circ = body['knee_circ'] * knee_v

            # Ensure crotch extension is wide enough for the thigh.
            # When thigh_circ > per-leg crotch width, the outseam must
            # bulge past the hip line, creating a spike. Widen the crotch
            # extension to match so the outseam stays monotonic.
            per_leg_crotch = front_hip + back_hip + crotch_extention
            if thigh_circ > per_leg_crotch:
                ext_ratio = (front_extention / crotch_extention
                             if crotch_extention > 0 else 0.5)
                crotch_extention = thigh_circ - front_hip - back_hip
                front_extention = crotch_extention * ext_ratio
                back_extention = crotch_extention - front_extention

            # Crotch-to-knee: prefer production override from design, then body YAML
            design_ctk = design.get('crotch_to_knee', {})
            if isinstance(design_ctk, dict) and design_ctk.get('v') is not None:
                crotch_to_knee = design_ctk['v']
            elif 'crotch_to_knee' in body:
                crotch_to_knee = body['crotch_to_knee']
            else:
                crotch_to_knee = 32.0

            # Y positions in panel coords (Y=0 at bottom, Y=length at hip)
            crotch_y = length - body['crotch_hip_diff']
            # Thigh point placed 2cm below crotch: just enough vertical
            # clearance for the straight crotch→thigh edge, while keeping
            # the control point near the thigh measurement cross-section.
            thigh_y = crotch_y - 2
            knee_y = crotch_y - crotch_to_knee

            # Only use subcurves if knee is above the ankle (pants long enough)
            if knee_y > 1:
                # Split thigh circumference using crotch-extension ratio:
                # At crotch_y the inseam X = hips + crotch_ext (fixed by
                # anatomy), so each panel's thigh share must match its
                # crotch-end width for the cross-section to hit the target.
                crotch_front = front_hip + front_extention
                crotch_back = back_hip + back_extention
                crotch_ratio = crotch_front / (crotch_front + crotch_back)

                # Knee uses hip ratio (knee measurement isn't affected by
                # crotch extensions since it's well below the crotch).
                total_hip_per_leg = front_hip + back_hip
                knee_front_ratio = front_hip / total_hip_per_leg

                leg_shape_kwargs = {
                    'knee_y': knee_y,
                    'thigh_y': thigh_y,
                    'front_thigh': thigh_circ * crotch_ratio,
                    'front_knee': knee_circ * knee_front_ratio,
                    'back_thigh': thigh_circ * (1 - crotch_ratio),
                    'back_knee': knee_circ * (1 - knee_front_ratio),
                }

        self.front = PantPanel(
            f'pant_f_{tag}', body, design,
            length=length,
            waist=(waist - waist_back) / 2,
            hips=front_hip,
            hips_depth=hips_depth,
            dart_position = body['bust_points'] / 2,
            crotch_width=front_extention,
            match_top_int_to=(body['waist'] - body['waist_back_width']) / 2,
            thigh_width=leg_shape_kwargs.get('front_thigh'),
            knee_width=leg_shape_kwargs.get('front_knee'),
            knee_y=leg_shape_kwargs.get('knee_y'),
            thigh_y=leg_shape_kwargs.get('thigh_y'),
            ).translate_by([0, body['_waist_level'] - 5 + rise_offset, 25])
        self.back = PantPanel(
            f'pant_b_{tag}', body, design,
            length=length,
            waist=waist_back / 2,
            hips=back_hip,
            hips_depth=hips_depth,
            hipline_ext=1.1,
            dart_position = body['bum_points'] / 2,
            crotch_width=back_extention,
            match_top_int_to=body['waist_back_width'] / 2,
            double_dart=True,
            thigh_width=leg_shape_kwargs.get('back_thigh'),
            knee_width=leg_shape_kwargs.get('back_knee'),
            knee_y=leg_shape_kwargs.get('knee_y'),
            thigh_y=leg_shape_kwargs.get('thigh_y'),
            ).translate_by([0, body['_waist_level'] - 5 + rise_offset, -20])

        self.stitching_rules = pyg.Stitches(
            (self.front.interfaces['outside'], self.back.interfaces['outside']),
            (self.front.interfaces['inside'], self.back.interfaces['inside'])
        )

        # add a cuff
        # TODOLOW This process is the same for sleeves -- make a function?
        if design['cuff']['type']['v']:
            
            pant_bottom = pyg.Interface.from_multiple(
                self.front.interfaces['bottom'],
                self.back.interfaces['bottom'])

            # Copy to avoid editing original design dict
            cdesign = deepcopy(design)
            cdesign['cuff']['b_width'] = {}
            cdesign['cuff']['b_width']['v'] = pant_bottom.edges.length() / design['cuff']['top_ruffle']['v']
            cdesign['cuff']['cuff_len']['v'] = cuff_len

            # Init
            cuff_class = getattr(bands, cdesign['cuff']['type']['v'])
            self.cuff = cuff_class(f'pant_{tag}', cdesign)

            # Position
            self.cuff.place_by_interface(
                self.cuff.interfaces['top'],
                pant_bottom,
                gap=5,
                alignment='left'
            )

            # Stitch
            self.stitching_rules.append((
                pant_bottom,
                self.cuff.interfaces['top'])
            )

        self.interfaces = {
            'crotch_f': self.front.interfaces['crotch'],
            'crotch_b': self.back.interfaces['crotch'],
            'top_f': self.front.interfaces['top'],
            'top_b': self.back.interfaces['top']
        }



    def length(self):
        if self.design['pants']['cuff']['type']['v']:
            return self.front.length() + self.cuff.length()
        
        return self.front.length()

class Pants(BaseBottoms):
    def __init__(self, body, design, rise=None) -> None:
        super().__init__(body, design)

        self.right = PantsHalf('r', body, design, rise)
        self.left = PantsHalf('l', body, design, rise).mirror()

        self.stitching_rules = pyg.Stitches(
            (self.right.interfaces['crotch_f'], self.left.interfaces['crotch_f']),
            (self.right.interfaces['crotch_b'], self.left.interfaces['crotch_b']),
        )

        self.interfaces = {
            'top_f': pyg.Interface.from_multiple(
                self.right.interfaces['top_f'], self.left.interfaces['top_f']),
            'top_b': pyg.Interface.from_multiple(
                self.right.interfaces['top_b'], self.left.interfaces['top_b']),
            # Some are reversed for correct connection
            'top': pyg.Interface.from_multiple(   # around the body starting from front right
                self.right.interfaces['top_f'].flip_edges(),
                self.left.interfaces['top_f'].reverse(with_edge_dir_reverse=True),
                self.left.interfaces['top_b'].flip_edges(),
                self.right.interfaces['top_b'].reverse(with_edge_dir_reverse=True), # Flips the edges and restores the direction
            )
        }

    def get_rise(self):
        return self.right.get_rise()
    
    def length(self):
        return self.right.length()

