import pygarment as pyg
from assets.garment_programs.circle_skirt import CircleArcPanel
from assets.garment_programs import skirt_paneled
from assets.garment_programs.base_classes import BaseBand

class StraightBandPanel(pyg.Panel):
    """One panel for a panel skirt"""

    def __init__(self, name, width, depth, match_int_proportion=None) -> None:
        super().__init__(name)

        # define edge loop
        self.edges = pyg.EdgeSeqFactory.from_verts(
            [0, 0], [0, depth], [width, depth], [width, 0], loop=True)

        # define interface
        self.interfaces = {
            'right': pyg.Interface(self, self.edges[0]),
            'top': pyg.Interface(self, 
                                 self.edges[1], 
                                 ruffle=width / match_int_proportion if match_int_proportion is not None else 1.
                                 ).reverse(True),
            'left': pyg.Interface(self, self.edges[2]),
            'bottom': pyg.Interface(self, 
                                    self.edges[3], 
                                    ruffle=width / match_int_proportion if match_int_proportion is not None else 1.
                                    )
        }

        # Default translation
        self.top_center_pivot()
        self.center_x()


class StraightWB(BaseBand):
    """Simple 2 panel waistband"""
    def __init__(self, body, design, rise=1.) -> None:
        """Simple 2 panel waistband

            * rise -- the rise value of the bottoms that the WB is attached to 
                Adapts the shape of the waistband to sit tight on top 
                of the given rise level (top measurement). If 1. or anything less than waistband width, 
                the rise is ignored and the StraightWB is created to sit well on the waist
        
        """
        super().__init__(body, design, rise=rise)

        # Measurements
        self.waist = design['waistband']['waist']['v'] * body['waist']
        self.waist_back_frac = body['waist_back_width'] / body['waist']
        self.hips = body['hips'] * design['waistband']['waist']['v']
        self.hips_back_frac = body['hip_back_width'] / body['hips']

        # Params
        self.width = design['waistband']['width']['v']
        self.rise = rise

        # Clamp interpolation factor at 1.0: for high-rise (rise > 1.0),
        # waistband width stays at waist measurement (narrowest)
        interp_factor = min(self.rise + self.width, 1.0)

        self.top_width = pyg.utils.lin_interpolation(
            self.hips, self.waist, interp_factor)
        self.top_back_fraction = pyg.utils.lin_interpolation(
            self.hips_back_frac, self.waist_back_frac, interp_factor)

        self.width = self.width * body['hips_line']

        # Elastic gathered waistband (opt-in via design flag). Fully isolated:
        # when off, the standard rectangle waistband below runs unchanged.
        if design['waistband'].get('elastic_gather', {}).get('v', False):
            self._build_elastic_gathered(body)
            return

        self.define_panels()

        # For rise > 1.0 (high-rise), offset waistband above natural waist
        rise_offset = max(0, (self.rise - 1.0)) * body['hips_line']
        self.front.translate_by([0, body['_waist_level'] + rise_offset, 20])
        self.back.translate_by([0, body['_waist_level'] + rise_offset, -15])

        self.stitching_rules = pyg.Stitches(
            (self.front.interfaces['right'], self.back.interfaces['right']),
            (self.front.interfaces['left'], self.back.interfaces['left'])
        )

        self.interfaces = {
            'bottom_f': self.front.interfaces['bottom'],
            'bottom_b': self.back.interfaces['bottom'],

            'top_f': self.front.interfaces['top'],
            'top_b': self.back.interfaces['top'],

            'bottom': pyg.Interface.from_multiple(
                self.front.interfaces['bottom'],
                self.back.interfaces['bottom']),
            'top': pyg.Interface.from_multiple(
                self.front.interfaces['top'],
                self.back.interfaces['top']),
        }

    def _build_elastic_gathered(self, body):
        """Elastic gathered waistband: a full-width fabric band (= config Waist)
        whose top edge is gathered into a narrow elastic band (= body waist).
        The fabric/elastic length mismatch at the gather seam forces the cloth
        to bunch into folds (same mechanism as the balloon-leg cuff).

        Builds two stacked straight tubes (fabric + thin elastic), 4 panels.
        Only ever called when the elastic_gather flag is set, so the standard
        waistband path is wholly unaffected.
        """
        rise_offset = max(0, (self.rise - 1.0)) * body['hips_line']
        y0 = body['_waist_level'] + rise_offset

        fabric = self.top_width            # config-Waist fabric circumference
        elastic = body['waist']            # body-waist elastic (the cinch)
        fab_back = self.top_back_fraction
        el_back = self.waist_back_frac
        e_depth = min(3.0, self.width * 0.5)   # thin elastic band height

        # Fabric tube (full width, full band height) — the bunched fabric.
        self.front = StraightBandPanel(
            'wb_front', fabric * (1 - fab_back), self.width,
            match_int_proportion=self.body['waist'] - self.body['waist_back_width'])
        self.back = StraightBandPanel(
            'wb_back', fabric * fab_back, self.width,
            match_int_proportion=self.body['waist_back_width'])
        self.front.translate_by([0, y0, 20])
        self.back.translate_by([0, y0, -15])

        # Elastic tube (narrow), stacked just above the fabric top.
        self.elastic_front = StraightBandPanel(
            'wb_elastic_front', elastic * (1 - el_back), e_depth)
        self.elastic_back = StraightBandPanel(
            'wb_elastic_back', elastic * el_back, e_depth)
        self.elastic_front.translate_by([0, y0 + self.width + 2, 20])
        self.elastic_back.translate_by([0, y0 + self.width + 2, -15])

        fabric_top = pyg.Interface.from_multiple(
            self.front.interfaces['top'], self.back.interfaces['top'])
        elastic_bottom = pyg.Interface.from_multiple(
            self.elastic_front.interfaces['bottom'], self.elastic_back.interfaces['bottom'])

        self.stitching_rules = pyg.Stitches(
            # fabric tube side seams
            (self.front.interfaces['right'], self.back.interfaces['right']),
            (self.front.interfaces['left'], self.back.interfaces['left']),
            # elastic tube side seams
            (self.elastic_front.interfaces['right'], self.elastic_back.interfaces['right']),
            (self.elastic_front.interfaces['left'], self.elastic_back.interfaces['left']),
            # GATHER seam: 90.5 fabric top eased into 71.2 elastic bottom -> folds
            (fabric_top, elastic_bottom),
        )

        self.interfaces = {
            'bottom_f': self.front.interfaces['bottom'],
            'bottom_b': self.back.interfaces['bottom'],
            'bottom': pyg.Interface.from_multiple(
                self.front.interfaces['bottom'], self.back.interfaces['bottom']),
            'top_f': self.elastic_front.interfaces['top'],
            'top_b': self.elastic_back.interfaces['top'],
            'top': pyg.Interface.from_multiple(
                self.elastic_front.interfaces['top'], self.elastic_back.interfaces['top']),
        }

    def define_panels(self):
        back_width = self.top_width * self.top_back_fraction

        self.front = StraightBandPanel(
            'wb_front',
            self.top_width - back_width,
            self.width,
            match_int_proportion=self.body['waist'] - self.body['waist_back_width']
        )

        self.back = StraightBandPanel(
            'wb_back',
            back_width,
            self.width,
            match_int_proportion=self.body['waist_back_width']
        )


class FittedWB(StraightWB):
    """Also known as Yoke: a waistband that ~follows the body curvature,
            and hence sits tight
        Made out of two circular arc panels
    """
    def __init__(self, body, design, rise=1.) -> None:
        """A waistband that ~follows the body curvature, and hence sits tight
        
            * rise -- the rise value of the bottoms that the WB is attached to 
                Adapts the shape of the waistband to sit tight on top 
                of the given rise level. If 1. or anything less than waistband width, 
                the rise is ignored and the FittedWB is created to sit well on the waist
        """
        self.bottom_width = None
        self.bottom_back_fraction = None
        super().__init__(body, design, rise)

    def define_panels(self):
        self.bottom_width = pyg.utils.lin_interpolation(
            self.hips, self.waist, self.rise)
        self.bottom_back_fraction = pyg.utils.lin_interpolation(
            self.hips_back_frac, self.waist_back_frac, self.rise)
        
        self.front = CircleArcPanel.from_all_length(
            'wb_front', 
            self.width, 
            self.top_width * (1 - self.top_back_fraction), 
            self.bottom_width * (1 - self.bottom_back_fraction),
            match_top_int_proportion=self.body['waist'] - self.body['waist_back_width'],
            match_bottom_int_proportion=self.body['waist'] - self.body['waist_back_width']
        )
        
        self.back = CircleArcPanel.from_all_length(
            'wb_back', 
            self.width, 
            self.top_width * self.top_back_fraction, 
            self.bottom_width * self.bottom_back_fraction,
            match_top_int_proportion=self.body['waist_back_width'],
            match_bottom_int_proportion=self.body['waist_back_width']
        )


class CuffBand(BaseBand):
    """ Cuff class for sleeves or pants
        band-like piece of fabric with optional "skirt"
    """
    def __init__(self, tag, design, length=None) -> None:
        super().__init__(body=None, design=design, tag=tag)

        self.design = design['cuff']

        if length is None:
            length = self.design['cuff_len']['v']

        self.front = StraightBandPanel(
            f'{tag}_cuff_f', self.design['b_width']['v'] / 2, length)
        self.front.translate_by([0, 0, 15])  
        self.back = StraightBandPanel(
            f'{tag}_cuff_b', self.design['b_width']['v'] / 2, length)
        self.back.translate_by([0, 0, -15])  

        self.stitching_rules = pyg.Stitches(
            (self.front.interfaces['right'], self.back.interfaces['right']),
            (self.front.interfaces['left'], self.back.interfaces['left'])
        )

        self.interfaces = {
            'bottom': pyg.Interface.from_multiple(
                self.front.interfaces['bottom'], 
                self.back.interfaces['bottom']),
            'top_front': self.front.interfaces['top'],
            'top_back': self.back.interfaces['top'],
            'top': pyg.Interface.from_multiple(
                self.front.interfaces['top'], 
                self.back.interfaces['top']),
        }


class CuffSkirt(BaseBand):
    """A skirt-like flared cuff """

    def __init__(self, tag, design, length=None) -> None:
        super().__init__(body=None, design=design, tag=tag)

        self.design = design['cuff']
        width = self.design['b_width']['v']
        flare_diff = (self.design['skirt_flare']['v'] - 1) * width / 2

        if length is None:
            length = self.design['cuff_len']['v']

        self.front = skirt_paneled.SkirtPanel(
            f'{tag}_cuff_skirt_f', ruffles=self.design['skirt_ruffle']['v'], 
            waist_length=width / 2, length=length, 
            flare=flare_diff)
        self.front.translate_by([0, 0, 15])  
        self.back = skirt_paneled.SkirtPanel(
            f'{tag}_cuff_skirt_b', ruffles=self.design['skirt_ruffle']['v'], 
            waist_length=width / 2, length=length, 
            flare=flare_diff)
        self.back.translate_by([0, 0, -15])  

        self.stitching_rules = pyg.Stitches(
            (self.front.interfaces['right'], self.back.interfaces['right']),
            (self.front.interfaces['left'], self.back.interfaces['left'])
        )

        self.interfaces = {
            'top': pyg.Interface.from_multiple(
                self.front.interfaces['top'], self.back.interfaces['top']),
            'top_front': self.front.interfaces['top'],
            'top_back': self.back.interfaces['top'],
            'bottom': pyg.Interface.from_multiple(
                self.front.interfaces['bottom'], 
                self.back.interfaces['bottom']),
        }


class CuffBandSkirt(pyg.Component):
    """ Cuff class for sleeves or pants
        band-like piece of fabric with optional "skirt"
    """
    def __init__(self, tag, design) -> None:
        super().__init__(self.__class__.__name__)

        self.cuff = CuffBand(
            tag, 
            design, 
            length=design['cuff']['cuff_len']['v'] * (1 - design['cuff']['skirt_fraction']['v'])
        )
        self.skirt = CuffSkirt(
            tag, 
            design, 
            length=design['cuff']['cuff_len']['v'] * design['cuff']['skirt_fraction']['v']
        )

        # Align
        self.skirt.place_below(self.cuff)

        self.stitching_rules = pyg.Stitches(
            (self.cuff.interfaces['bottom'], self.skirt.interfaces['top']),
        )

        self.interfaces = {
            'top': self.cuff.interfaces['top'],
            'top_front': self.cuff.interfaces['top_front'],
            'top_back': self.cuff.interfaces['top_back'],
            'bottom': self.skirt.interfaces['bottom']
        }

    def length(self):
        return self.cuff.length() + self.skirt.length()