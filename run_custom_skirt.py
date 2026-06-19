"""
Custom Skirt Generation & Simulation Pipeline.

Mirrors the pants/shirt pipelines: production_data flows through design
knobs only — no mutation of the body object. Per-size sizing is achieved
via `target_waist` / `target_hips` knobs on the relevant design section,
which the skirt classes (PencilSkirt, Skirt2, SkirtCircle, ...) consume
through `apply_design_body_targets` (assets/garment_programs/base_classes.py)
to build a scaled body view internally.

Supported production_data fields (cm):
  - Waist  : full waist circumference  -> design.<section>.target_waist
  - Low_Hip: full hip circumference    -> design.<section>.target_hips
                                          (PencilSkirt only — others ignore)
  - Hem    : full hem circumference    -> per-style flare/suns derivation
  - Length : back-center waist→hem     -> per-style length.v derivation

Skirt style is selected by the yaml's top-level `skirt_style:` field
(default PencilSkirt). Currently mapped:
  - PencilSkirt      : fitted darted panel
  - Skirt2           : flat 2-panel A-line
  - SkirtCircle      : circle skirt (suns from Hem/Waist/Length)
  - AsymmSkirtCircle : circle skirt with asymmetric front/back length
  - SkirtManyPanels  : N-panel round skirt (n_panels via design_overrides)

GodetSkirt and SkirtLevels are not wired yet — both wrap a base skirt
class, so per-style mappers can be added incrementally.
"""
from pathlib import Path
from datetime import datetime
import math
import yaml

# Reuse the shirt-pipeline simulate_pattern verbatim — sim is garment-agnostic.
from run_custom_tshirt import simulate_pattern  # noqa: F401  (re-exported)


_DEFAULT_DESIGN_YAML = './assets/design_params/default.yaml'


def _load_default_design():
    with open(_DEFAULT_DESIGN_YAML) as f:
        return yaml.safe_load(f)['design']


def _set_target_dimensions(section, prod, include_hips=False):
    """Set target_waist (and optionally target_hips) on a design section
    from production_data. Mirrors how the pants/shirt mappers feed design
    knobs — body is never mutated.
    """
    if 'Waist' in prod and prod['Waist'] > 0:
        section.setdefault('target_waist', {})['v'] = float(prod['Waist'])
    if include_hips and 'Low_Hip' in prod and prod['Low_Hip'] > 0:
        section.setdefault('target_hips', {})['v'] = float(prod['Low_Hip'])


# ─────────────────────── per-style design mappers ─────────────────────── #
# Each mapper takes (prod, body, design) and mutates the relevant design
# section in place. Body is read-only.

def _map_pencil_skirt(prod, body, design):
    """PencilSkirt — fitted darted panel."""
    section = design['pencil-skirt']
    _set_target_dimensions(section, prod, include_hips=True)

    back_hipline_ext = 1.05   # FittedSkirtPanel uses 1.05 for the back panel
    rise = section['rise']['v']
    adj_hips_depth_back = body['hips_line'] * rise * back_hipline_ext

    if 'Length' in prod:
        section['length']['v'] = float(
            (prod['Length'] - adj_hips_depth_back) / body['_leg_length'])

    if 'Hem' in prod and 'Low_Hip' in prod and prod['Low_Hip'] > 0:
        # FittedSkirtPanel: hem total − hip total = body.hips × (flare − 1);
        # with body.hips overridden (via target_hips) to Low_Hip:
        section['flare']['v'] = float(
            (prod['Hem'] - prod['Low_Hip']) / prod['Low_Hip'] + 1.0)


def _map_skirt2(prod, body, design):
    """Skirt2 — flat 2-panel A-line, no darts. Optional ruffle/gather."""
    section = design['skirt']
    _set_target_dimensions(section, prod, include_hips=False)

    if 'Length' in prod:
        section['length']['v'] = float(
            (prod['Length'] - body['hips_line']) / body['_leg_length'])

    if 'Hem' in prod and 'Waist' in prod:
        # SkirtPanel: hem = waist + 4×flare (flare per side, cm)
        section['flare']['v'] = float((prod['Hem'] - prod['Waist']) / 4.0)

    # Default: no gather. Override via design_overrides if you want one.
    section['ruffle']['v'] = 1.0


def _map_circle(prod, body, design, asymm=False):
    """SkirtCircle / AsymmSkirtCircle — circle skirt.
    Hem geometry: hem = waist + 2π × length × suns.
    """
    section = design['flare-skirt']
    _set_target_dimensions(section, prod, include_hips=False)

    length_cm = None
    if 'Length' in prod:
        length_cm = prod['Length'] - body['hips_line']
        section['length']['v'] = float(length_cm / body['_leg_length'])

    if ('Hem' in prod and 'Waist' in prod
            and length_cm is not None and length_cm > 0):
        suns_v = (prod['Hem'] - prod['Waist']) / (2.0 * math.pi * length_cm)
        # default.yaml range [0.1, 1.95]; soft-clamp to keep it valid.
        section['suns']['v'] = float(max(0.1, min(1.95, suns_v)))


def _map_many_panels(prod, body, design):
    """SkirtManyPanels — uses flare-skirt section + n_panels override."""
    _map_circle(prod, body, design, asymm=False)


_STYLE_MAPPERS = {
    'PencilSkirt':      _map_pencil_skirt,
    'Skirt2':           _map_skirt2,
    'SkirtCircle':      lambda p, b, d: _map_circle(p, b, d, asymm=False),
    'AsymmSkirtCircle': lambda p, b, d: _map_circle(p, b, d, asymm=True),
    'SkirtManyPanels':  _map_many_panels,
}


def map_production_to_design(prod, body_yaml_path, skirt_style='PencilSkirt'):
    """Build a design dict from production_data for a skirt of given style."""
    from assets.bodies.body_params import BodyParameters

    design = _load_default_design()
    body = BodyParameters(body_yaml_path)

    # Common meta: skirt-only, no upper, no separate waistband by default.
    design['meta']['upper']['v'] = None
    design['meta']['bottom']['v'] = skirt_style
    design['meta']['wb']['v'] = None

    mapper = _STYLE_MAPPERS.get(skirt_style)
    if mapper is None:
        print(f'  WARNING: no production mapper for skirt_style={skirt_style!r}; '
              'using design defaults — knobs will need design_overrides.')
    else:
        mapper(prod, body, design)

    return design


def generate_pattern(size, design, body_yaml_path, output_base,
                     garment_prefix='skirt', **_unused):
    """Build skirt pattern. Body is never modified — production sizing
    flows through design knobs (target_waist / target_hips), which the
    skirt class consumes via apply_design_body_targets internally.
    """
    from assets.garment_programs.meta_garment import MetaGarment
    from assets.bodies.body_params import BodyParameters

    garment_name = f'{garment_prefix}_size{size}'

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
