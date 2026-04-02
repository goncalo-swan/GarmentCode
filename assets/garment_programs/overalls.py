from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R

import pygarment as pyg

from assets.garment_programs.base_classes import BaseBottoms
from assets.garment_programs.pants import Pants


class BibPanel(pyg.Panel):
    """Rectangular bib panel for overalls with strap attachment sub-edges.

    Top edge is subdivided into 3 segments:
      top_left_strap | top_center | top_right_strap
    so straps can be stitched to the corner segments.
    """
    def __init__(self, name, width, height, strap_width) -> None:
        super().__init__(name)

        sw = min(strap_width, width / 2 - 1)

        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0],                     # 0: bottom-left
            [0, height],                # 1: top-left
            [sw, height],               # 2: left strap end
            [width - sw, height],       # 3: right strap start
            [width, height],            # 4: top-right
            [width, 0],                 # 5: bottom-right
            loop=True
        )

        self.interfaces = {
            'bottom': pyg.Interface(self, self.edges[5]),
            'left': pyg.Interface(self, self.edges[0]),
            'top_left_strap': pyg.Interface(self, self.edges[1]),
            'top_center': pyg.Interface(self, self.edges[2]),
            'top_right_strap': pyg.Interface(self, self.edges[3]),
            'right': pyg.Interface(self, self.edges[4]),
            'top': pyg.Interface(self, self.edges[1:4]),
        }

        self.top_center_pivot()
        self.center_x()


class StrapPanel(pyg.Panel):
    """Narrow rectangular strap for overalls."""
    def __init__(self, name, width, length) -> None:
        super().__init__(name)

        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0], [0, length], [width, length], [width, 0], loop=True
        )

        self.interfaces = {
            'bottom': pyg.Interface(self, self.edges[3]),
            'top': pyg.Interface(self, self.edges[1]),
        }

        self.top_center_pivot()
        self.center_x()


class Overalls(BaseBottoms):
    """Overalls: pants with front bib, back bib, and shoulder straps.

    Both bibs stitch to pants waist (front and back). Straps
    connect front bib top corners to back bib top corners,
    going over the shoulders. All panels are stitched — no
    free-hanging edges.
    """
    def __init__(self, body, design, rise=None) -> None:
        super().__init__(body, design, rise=rise)

        od = design['overalls']

        # Create the pants
        self.pants = Pants(body, design, rise=rise if rise is not None else 1.)

        # Dimensions
        front_waist = body['waist'] - body['waist_back_width']
        back_waist = body['waist_back_width']
        bib_height = body['waist_line'] * od['bib_height']['v']
        strap_width = od['strap_width']['v']

        # Scale bib widths to match pants top interfaces
        bib_w_frac = od['bib_width']['v']
        front_bib_w = front_waist * bib_w_frac
        back_bib_w = back_waist * bib_w_frac

        waist_y = body['_waist_level']

        # Front bib — close to body surface (Z=5)
        self.front_bib = BibPanel('front_bib', front_bib_w, bib_height, strap_width)
        self.front_bib.translate_by([0, waist_y, 5])

        # Back bib
        self.back_bib = BibPanel('back_bib', back_bib_w, bib_height, strap_width)
        self.back_bib.translate_by([0, waist_y, -5])

        # Straps: go over the shoulders from front bib to back bib.
        # Length = tight fit over-shoulder distance (front gap + shoulder arc + back gap)
        shoulder_y = body['height'] - body['head_l']
        front_gap = shoulder_y - (waist_y + bib_height)
        strap_length = front_gap * 2 + 8  # 8cm shoulder arc — tight fit
        shoulder_offset = body['shoulder_w'] / 4

        # Right strap — positioned over right shoulder
        self.strap_r = StrapPanel('strap_r', strap_width, strap_length)
        self.strap_r.translate_by([shoulder_offset, shoulder_y, 0])
        self.strap_r.rotate_by(R.from_euler('XYZ', [90, 0, 0], degrees=True))

        # Left strap — positioned over left shoulder
        self.strap_l = StrapPanel('strap_l', strap_width, strap_length)
        self.strap_l.translate_by([-shoulder_offset, shoulder_y, 0])
        self.strap_l.rotate_by(R.from_euler('XYZ', [90, 0, 0], degrees=True))

        # Stitching:
        # Combined bib bottom wrapping the full waist (matching pants 'top' interface
        # ordering: front-right → front-left → back-left → back-right).
        # Using the combined interface avoids same-panel vertex conflicts.
        combined_bib_bottom = pyg.Interface.from_multiple(
            self.front_bib.interfaces['bottom'].reverse(),
            self.back_bib.interfaces['bottom'],
        )
        self.stitching_rules = pyg.Stitches(
            (combined_bib_bottom, self.pants.interfaces['top']),
        )

        # Straps: top → front bib corner, bottom → back bib corner
        self.stitching_rules.append(
            (self.strap_r.interfaces['top'], self.front_bib.interfaces['top_right_strap'])
        )
        self.stitching_rules.append(
            (self.strap_r.interfaces['bottom'], self.back_bib.interfaces['top_right_strap'])
        )
        self.stitching_rules.append(
            (self.strap_l.interfaces['top'], self.front_bib.interfaces['top_left_strap'])
        )
        self.stitching_rules.append(
            (self.strap_l.interfaces['bottom'], self.back_bib.interfaces['top_left_strap'])
        )

        # Reuse recognized attachment labels so the sim pins bibs/straps.
        # 'strapless_top' → vertical attachment under armscye (holds bibs up)
        # 'right_collar'/'left_collar' → horizontal attachment near neck (holds straps)
        self.front_bib.interfaces['top_center'].edges.propagate_label('strapless_top')
        self.back_bib.interfaces['top_center'].edges.propagate_label('strapless_top')
        self.strap_r.edges[0].label = 'right_collar'
        self.strap_r.edges[2].label = 'right_collar'
        self.strap_l.edges[0].label = 'left_collar'
        self.strap_l.edges[2].label = 'left_collar'

        # Expose top interface for MetaGarment compatibility
        self.interfaces = {
            'top': self.front_bib.interfaces['top_center'],
        }

    def get_rise(self):
        return self.pants.get_rise()

    def length(self):
        return self.pants.length() + self.front_bib.edges[0].length()
