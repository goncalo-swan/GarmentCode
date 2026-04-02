import pygarment as pyg

from assets.garment_programs.bodice import ButtonDownShirt, BodiceHalf


class Jacket(ButtonDownShirt):
    """Jacket: open-front upper garment with wider placket.

    Extends ButtonDownShirt with wider default overlap and
    compatibility with structured collar types (NotchedLapelCollar,
    PeakLapelCollar).
    """
    def __init__(self, body, design, fitted=False) -> None:
        # Jacket uses its own placket_width if specified, falling back to shirt's
        jacket_d = design.get('jacket', {})
        if 'placket_width' in jacket_d:
            design['shirt']['placket_width']['v'] = jacket_d['placket_width']['v']

        # Double-breasted doubles the placket width
        if jacket_d.get('double_breasted', {}).get('v', False):
            design['shirt']['placket_width']['v'] *= 2

        # Propagate jacket closure to shirt closure
        if 'closure' in jacket_d:
            design['shirt']['closure']['v'] = jacket_d['closure']['v']

        super().__init__(body, design, fitted=fitted)

    def length(self):
        return self.right.length()


class FittedJacket(Jacket):
    """Fitted jacket with darts."""
    def __init__(self, body, design) -> None:
        super().__init__(body, design, fitted=True)


class Coat(Jacket):
    """Long coat extending below the waist.

    Same construction as Jacket but allows longer length
    via the coat.length design parameter.
    """
    def __init__(self, body, design, fitted=False) -> None:
        # Apply coat length override if specified
        coat_d = design.get('coat', {})
        if 'length' in coat_d:
            design['shirt']['length']['v'] = coat_d['length']['v']

        super().__init__(body, design, fitted=fitted)


class FittedCoat(Coat):
    """Fitted coat with darts."""
    def __init__(self, body, design) -> None:
        super().__init__(body, design, fitted=True)
