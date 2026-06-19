""" Panels for a straight upper garment (T-shirt)
    Note that the code is very similar to Bodice.
"""
import numpy as np
import pygarment as pyg

from assets.garment_programs.base_classes import BaseBodicePanel


def _collar_width_cm(body, design):
    """Resolve the design's collar width parameter to a cm value, matching
    bodice.py's branched lin_interpolation (positive v: neck_w..max_w;
    negative v: 0..neck_w via (1+v)*neck_w).
    """
    v = design.get('collar', {}).get('width', {}).get('v', 0.2)
    neck_w = body['neck_w']
    if v >= 0:
        max_w = body['_base_sleeve_balance'] - 2
        return neck_w + v * (max_w - neck_w)
    return (1.0 + v) * neck_w


_EXTRA_HPS_LIFT_CM = 0.0   # bonus lift so the panel rides slightly above body HPS


def _hps_y_offset(body, design, panel_half_width):
    """Y compensation: after the collar is cut, the panel's max-Y vertex is
    the shoulder-neck junction (NOT the pre-cut HPS corner). Without
    compensation, translate_by lands the (cut) panel ~tan(slope) * (cw/2) cm
    below the body's HPS. We also add _EXTRA_HPS_LIFT_CM so the panel rides
    visibly above HPS — observed renders sat too low at strict 0 cm gap.
    """
    sh_tan = np.tan(np.deg2rad(body['_shoulder_incl']))
    cw = _collar_width_cm(body, design)
    # Clamp neck half-width to panel half-width to avoid weird artefacts
    nhw = min(cw / 2.0, max(panel_half_width - 1e-3, 0.0))
    return sh_tan * nhw + _EXTRA_HPS_LIFT_CM


class TorsoFrontHalfPanel(BaseBodicePanel):
    """Half of a simple non-fitted upper garment (e.g. T-Shirt)
    
        Fits to the bust size
    """
    def __init__(self, name, body, design) -> None:
        """ Front = True, provides the adjustments necessary for the front panel
        """
        super().__init__(name, body, design)

        self._design_root = design
        design = design['shirt']

        # width
        m_width = design['width']['v'] * body['bust']
        b_width = design['flare']['v'] * m_width

        # sizes
        body_width = (body['bust'] - body['back_width']) / 2
        frac = body_width / body['bust']
        self.width = frac * m_width
        b_width = frac * b_width

        sh_tan = np.tan(np.deg2rad(body['_shoulder_incl']))
        shoulder_incl = sh_tan * self.width
        length = design['length']['v'] * body['waist_line']

        # length in the front panel is adjusted due to shoulder inclination
        # for the correct sleeve fitting
        fb_diff = (frac - (0.5 - frac)) * body['bust']
        length = length - sh_tan * fb_diff

        # Optional waist nip: insert a side vertex pulled inward at the waist
        # line so the side seam squeezes at the waist and flares to the hem
        # (hourglass). Gated by shirt.waist_fitted so plain tees are unchanged.
        # waist_x scales the bust side-width by the waist/bust ratio; the hem
        # (b_width) already carries the flare from the mapper.
        waist_fitted = design.get('waist_fitted', {}).get('v', False)
        waist_y = length - body['waist_line']
        if waist_fitted and waist_y > 2:
            waist_x = self.width * (body['waist'] / m_width)
            self.edges = pyg.EdgeSeqFactory.from_verts(
                [0, 0],
                [-b_width, 0],
                [-waist_x, waist_y],
                [-self.width, length],
                [0, length + shoulder_incl],
                loop=True
            )
            outside_int = pyg.Interface(self, [self.edges[1], self.edges[2]])
        else:
            self.edges = pyg.EdgeSeqFactory.from_verts(
                [0, 0],
                [-b_width, 0],
                [-self.width, length],
                [0, length + shoulder_incl],
                loop=True
            )
            outside_int = pyg.Interface(self, self.edges[1])

        # Interfaces
        self.interfaces = {
            'outside':  outside_int,
            'inside': pyg.Interface(self, self.edges[-1]),
            'shoulder': pyg.Interface(self, self.edges[-2]),
            'bottom': pyg.Interface(self, self.edges[0], ruffle=self.edges[0].length() / ((body['waist'] - body['waist_back_width']) / 2)),

            # Reference to the corner for sleeve and collar projections
            'shoulder_corner': pyg.Interface(self, [self.edges[-3], self.edges[-2]]),
            'collar_corner': pyg.Interface(self, [self.edges[-2], self.edges[-1]])
        }

        # Default placement: shoulder-neck junction at body HPS. The original
        # formula assumed the pre-cut HPS corner was the panel top, but the
        # collar cut later removes that corner — actual panel top is the
        # shoulder-neck junction, tan(slope) * (cw/2) cm below the HPS corner.
        # Compensate so the post-cut top lands at body HPS.
        y_off = _hps_y_offset(body, self._design_root, self.width)
        self.translate_by([0,
                           body['height'] - body['head_l']
                           - length - shoulder_incl + y_off, 0])

    def get_width(self, level):
        return super().get_width(level) + self.width - self.body['shoulder_w'] / 2


class TorsoBackHalfPanel(BaseBodicePanel):
    """Half of a simple non-fitted upper garment (e.g. T-Shirt)
    
        Fits to the bust size
    """
    def __init__(self, name, body, design) -> None:
        """ Front = True, provides the adjustments necessary for the front panel
        """
        super().__init__(name, body, design)

        self._design_root = design
        design = design['shirt']
        # account for ease in basic measurements
        m_width = design['width']['v'] * body['bust']
        b_width = design['flare']['v'] * m_width

        # sizes 
        body_width = body['back_width'] / 2
        frac = body_width / body['bust'] 
        self.width = frac * m_width
        b_width = frac * b_width

        shoulder_incl = (np.tan(np.deg2rad(body['_shoulder_incl']))) * self.width
        length = design['length']['v'] * body['waist_line']

        # Optional waist nip (see TorsoFrontHalfPanel).
        waist_fitted = design.get('waist_fitted', {}).get('v', False)
        waist_y = length - body['waist_line']
        if waist_fitted and waist_y > 2:
            waist_x = self.width * (body['waist'] / m_width)
            self.edges = pyg.EdgeSeqFactory.from_verts(
                [0, 0],
                [-b_width, 0],
                [-waist_x, waist_y],
                [-self.width, length],
                [0, length + shoulder_incl],
                loop=True
            )
            outside_int = pyg.Interface(self, [self.edges[1], self.edges[2]])
        else:
            self.edges = pyg.EdgeSeqFactory.from_verts(
                [0, 0],
                [-b_width, 0],
                [-self.width, length],
                [0, length + shoulder_incl],
                loop=True
            )
            outside_int = pyg.Interface(self, self.edges[1])

        # Interfaces
        self.interfaces = {
            'outside':  outside_int,
            'inside': pyg.Interface(self, self.edges[-1]),
            'shoulder': pyg.Interface(self, self.edges[-2]),
            'bottom': pyg.Interface(self, self.edges[0], ruffle=self.edges[0].length() / (body['waist_back_width'] / 2)),

            # Reference to the corner for sleeve and collar projections
            'shoulder_corner': pyg.Interface(self, [self.edges[-3], self.edges[-2]]),
            'collar_corner': pyg.Interface(self, [self.edges[-2], self.edges[-1]])
        }

        # See TorsoFrontHalfPanel for the y_off rationale.
        y_off = _hps_y_offset(body, self._design_root, self.width)
        self.translate_by([0,
                           body['height'] - body['head_l']
                           - length - shoulder_incl + y_off, 0])

    def get_width(self, level):
        return super().get_width(level) + self.width - self.body['shoulder_w'] / 2
