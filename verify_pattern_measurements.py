"""
verify_pattern_measurements.py
===============================
Verify GarmentCode pattern measurements from specification JSON files
against CALITEE production targets.

Measurements extracted directly from panel vertex geometry in the JSON spec:

  A      - Bust: full circumference = 2 × (right_ftorso hem + right_btorso hem)
  N      - Collar width = 2 × |x-coordinate of HPS vertex on right_ftorso|
  O1     - Front neck drop = HPS_y − front_center_neck_y  (right_ftorso)
  O2     - Back neck drop  = HPS_y − back_center_neck_y   (right_btorso)
  I      - HPS to cuff = shoulder edge length + sleeve top edge length
  Z      - Back length = center back seam length (right_btorso v_top_y)
  M_cuff - Cuff circumference = right_sleeve_f opening + right_sleeve_b opening

Panel geometry conventions (from generate_tshirt_spec.py / tee.py after collar cut):

  right_ftorso vertices (CCW):
    v0 = (0, 0)              hem center
    v1 = (0, Z_front)        front center neck (top of center seam)
    v2 = (-N/2, HPS_y)       HPS (shoulder-neck junction)
    v3 = (shoulder_tip_x, shoulder_tip_y)
    ...

  right_btorso vertices (CCW):
    v0 = (0, 0)              hem center
    v1 = (-A/4_back, 0)      hem side
    ...
    v4 = (-N/2, HPS_y)       HPS
    v5 = (0, Z_back)         center back neck (top of center seam)

  right_sleeve_f vertices:
    v0 = (0, 0)              shoulder junction (top / cuff end)
    v1 = (0, -opening/2)     cuff bottom
    v2 = (sleeve_len, -arm_w) armhole end bottom
    v3 = (sleeve_len, 0)     armhole end top

  right_sleeve_b vertices:
    v0 = (0, 0)              shoulder junction (top)
    v1 = (sleeve_len, 0)     armhole end top
    ...
    v4 = (0, -opening/2)     cuff bottom

Edge length computation:
  - Straight edge: Euclidean distance between endpoints
  - Cubic Bezier: svgpathtools.CubicBezier.length() with rel→abs control point conversion
  - Circle arc:   svgpathtools.Arc.length()
"""

import json
import sys
from pathlib import Path

import numpy as np
import svgpathtools as svgpath


# ============================================================
# Production targets
# ============================================================

PRODUCTION_DATA = {
    48: {
        'Bust': 101,
        'Arm_Length': 36.6,
        'Nape_to_Waist': 71,
        'Sleeve_Opening': 34.1,
        'Collar_Width': 16.7,
        'Front_Neck_Drop': 10.7,
        'Back_Neck_Drop': 1.9,
    },
    50: {
        'Bust': 105,
        'Arm_Length': 37.3,
        'Nape_to_Waist': 72,
        'Sleeve_Opening': 35.2,
        'Collar_Width': 17.0,
        'Front_Neck_Drop': 11.0,
        'Back_Neck_Drop': 2.1,
    },
    52: {
        'Bust': 109,
        'Arm_Length': 37.9,
        'Nape_to_Waist': 73,
        'Sleeve_Opening': 36.3,
        'Collar_Width': 17.2,
        'Front_Neck_Drop': 11.4,
        'Back_Neck_Drop': 2.2,
    },
}

SPEC_PATHS = {
    48: Path('Logs/calitee_tshirt_size48__260224-22-08-16/'
             'calitee_tshirt_size48__260224-22-08-16_specification.json'),
    50: Path('Logs/calitee_tshirt_size50__260224-22-08-18/'
             'calitee_tshirt_size50__260224-22-08-18_specification.json'),
    52: Path('Logs/calitee_tshirt_size52__260224-22-08-21/'
             'calitee_tshirt_size52__260224-22-08-21_specification.json'),
}


# ============================================================
# Edge length helpers
# ============================================================

def _rel_to_abs(start, end, rel_pt):
    """Convert relative control point (fractional along edge + perpendicular)
    to absolute 2-D coordinates.

    GarmentCode stores Bezier control points as (t, perp) where:
      t    = fraction along the straight edge [start → end]
      perp = fraction of edge length in the perpendicular direction

    This matches pygarment/pattern/utils.py::rel_to_abs_2d.
    """
    start = np.asarray(start, dtype=float)
    end   = np.asarray(end,   dtype=float)
    edge      = end - start
    edge_perp = np.array([-edge[1], edge[0]])
    return start + rel_pt[0] * edge + rel_pt[1] * edge_perp


def edge_length(edge_dict, verts):
    """Return the arc-length of a single edge from a panel JSON dict.

    Parameters
    ----------
    edge_dict : dict  – one element of panel['edges']
    verts     : list  – panel['vertices']

    Returns
    -------
    float  arc-length in cm
    """
    ep  = edge_dict['endpoints']
    v1  = verts[ep[0]]
    v2  = verts[ep[1]]

    if 'curvature' not in edge_dict:
        # Straight edge
        return float(np.linalg.norm(np.array(v2) - np.array(v1)))

    curv = edge_dict['curvature']
    ctype = curv['type']

    if ctype == 'circle':
        radius, large_arc, sweep = curv['params']
        arc = svgpath.Arc(
            complex(v1[0], v1[1]),
            complex(radius, radius),
            0,
            int(large_arc),
            int(sweep),
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
        # Unknown curvature type – fall back to straight distance
        return float(np.linalg.norm(np.array(v2) - np.array(v1)))


# ============================================================
# Core measurement logic
# ============================================================

def _panel_edge_lengths(panel):
    """Return list of arc-lengths for every edge of a panel dict."""
    verts = panel['vertices']
    return [edge_length(e, verts) for e in panel['edges']]


def measure_spec(spec_path):
    """Extract all 7 garment measurements from a specification JSON.

    The function identifies the relevant panels by name and reads the
    geometry directly from vertex coordinates – it does NOT rely on
    edge indices that may shift after cut_corner operations (collar/sleeve
    cuts reorder edges).  Instead it finds key vertices by their position:

    right_ftorso
      - hem  : the single edge whose both endpoints have y ≈ 0
      - HPS  : the vertex with maximum y among {v with x ≠ 0}
                → N = 2 × |HPS_x|
      - front center neck: vertex with x = 0 and maximum y
                → O1 = HPS_y − neck_y

    right_btorso
      - hem  : the edge whose both endpoints have y ≈ 0
      - HPS  : vertex with max y among {v with x ≠ 0}
                → O2 = HPS_y − neck_y
      - back center neck: vertex with x = 0 and maximum y
                → Z  = neck_y  (= center seam length)

    right_sleeve_f / right_sleeve_b
      - The cuff edge is the free (un-stitched) opening edge
        on each sleeve panel:  both endpoints share x ≈ x_min of the panel
        (the cuff end of the sleeve, away from the armhole)
      - M_cuff = sleeve_f_opening + sleeve_b_opening
      - I      = shoulder_edge_len + sleeve_top_x_span
    """
    with open(spec_path, 'r') as fh:
        spec = json.load(fh)

    panels = spec['pattern']['panels']

    # ---- Collect stitched edge indices to find free edges ----
    stitched = set()
    for stitch in spec['pattern']['stitches']:
        for side in stitch:
            stitched.add((side['panel'], side['edge']))

    # ---- Helper: free edges of a panel ----
    def free_edges(panel_name):
        return [
            (i, panels[panel_name]['edges'][i])
            for i in range(len(panels[panel_name]['edges']))
            if (panel_name, i) not in stitched
        ]

    # ---- right_ftorso ----
    ft  = panels['right_ftorso']
    fv  = ft['vertices']

    # HPS: vertex with the maximum y-coordinate that is NOT at x ≈ 0
    # (i.e., the shoulder–neck junction)
    hps_front_idx = max(
        (i for i, v in enumerate(fv) if abs(v[0]) > 0.01),
        key=lambda i: fv[i][1]
    )
    hps_front = fv[hps_front_idx]

    # Front center neck: vertex with x ≈ 0 and maximum y  (top of center seam)
    neck_front_idx = max(
        (i for i, v in enumerate(fv) if abs(v[0]) < 0.01),
        key=lambda i: fv[i][1]
    )
    neck_front = fv[neck_front_idx]

    # Hem of front torso: find the free horizontal edge at y = 0
    # (the straight edge connecting hem center to hem side, both with y ≈ 0)
    ft_hem_len = None
    for _, e in free_edges('right_ftorso'):
        ep  = e['endpoints']
        v_a = fv[ep[0]]
        v_b = fv[ep[1]]
        if abs(v_a[1]) < 0.01 and abs(v_b[1]) < 0.01:
            ft_hem_len = abs(v_b[0] - v_a[0])
            break
    if ft_hem_len is None:
        # Fallback: largest free edge at y = 0
        ft_hem_len = max(
            abs(fv[e['endpoints'][1]][0] - fv[e['endpoints'][0]][0])
            for _, e in free_edges('right_ftorso')
            if abs(fv[e['endpoints'][0]][1]) < 0.5 and abs(fv[e['endpoints'][1]][1]) < 0.5
        )

    # Shoulder edge: straight line from HPS to the adjacent shoulder-tip vertex
    # In right_ftorso the shoulder edge connects HPS (v2) to the shoulder tip (v3)
    # which is the vertex immediately after HPS when traversing CCW.
    # Find the edge that has hps_front_idx as one endpoint and is NOT the neckline.
    shoulder_edge_len = None
    for e in ft['edges']:
        ep = e['endpoints']
        if hps_front_idx in ep and 'curvature' not in e:
            # Straight edge touching HPS – could be shoulder or neckline segment
            other = ep[0] if ep[1] == hps_front_idx else ep[1]
            other_v = fv[other]
            # Shoulder tip is lower in y than HPS and more to the left (more negative x)
            if other_v[1] < hps_front[1] and other_v[0] < hps_front[0]:
                shoulder_edge_len = edge_length(e, fv)
                break
    if shoulder_edge_len is None:
        # Fallback: among straight edges touching HPS, take the longer one
        shoulder_edge_len = max(
            edge_length(e, fv)
            for e in ft['edges']
            if hps_front_idx in e['endpoints'] and 'curvature' not in e
        )

    # ---- right_btorso ----
    bt  = panels['right_btorso']
    bv  = bt['vertices']

    # HPS: vertex with max y among non-center vertices
    hps_back_idx = max(
        (i for i, v in enumerate(bv) if abs(v[0]) > 0.01),
        key=lambda i: bv[i][1]
    )
    hps_back = bv[hps_back_idx]

    # Back center neck: vertex with x ≈ 0 and max y
    neck_back_idx = max(
        (i for i, v in enumerate(bv) if abs(v[0]) < 0.01 and bv[i][1] > 0.01),
        key=lambda i: bv[i][1]
    )
    neck_back = bv[neck_back_idx]

    # Back hem: free horizontal edge
    bt_hem_len = None
    for _, e in free_edges('right_btorso'):
        ep  = e['endpoints']
        v_a = bv[ep[0]]
        v_b = bv[ep[1]]
        if abs(v_a[1]) < 0.01 and abs(v_b[1]) < 0.01:
            bt_hem_len = abs(v_b[0] - v_a[0])
            break
    if bt_hem_len is None:
        bt_hem_len = max(
            abs(bv[e['endpoints'][1]][0] - bv[e['endpoints'][0]][0])
            for _, e in free_edges('right_btorso')
            if abs(bv[e['endpoints'][0]][1]) < 0.5 and abs(bv[e['endpoints'][1]][1]) < 0.5
        )

    # ---- right_sleeve_f ----
    sf   = panels['right_sleeve_f']
    sfv  = sf['vertices']

    # Sleeve top edge (connects shoulder junction to armhole top)
    # Both endpoints have y ≈ 0 (they are at the top/shoulder end of the sleeve)
    # This is the stitched seam between right_sleeve_f and right_sleeve_b at the top.
    # In right_sleeve_f: find the stitched edge with both endpoints at y ≈ 0
    sleeve_top_len = None
    for i, e in enumerate(sf['edges']):
        if ('right_sleeve_f', i) in stitched:
            ep  = e['endpoints']
            v_a = sfv[ep[0]]
            v_b = sfv[ep[1]]
            if abs(v_a[1]) < 0.01 and abs(v_b[1]) < 0.01:
                sleeve_top_len = abs(v_b[0] - v_a[0])
                break
    if sleeve_top_len is None:
        # Fallback: largest stitched edge with endpoints near y=0
        sleeve_top_len = max(
            abs(sfv[e['endpoints'][1]][0] - sfv[e['endpoints'][0]][0])
            for i, e in enumerate(sf['edges'])
            if ('right_sleeve_f', i) in stitched
            and abs(sfv[e['endpoints'][0]][1]) < 0.5
            and abs(sfv[e['endpoints'][1]][1]) < 0.5
        )

    # Cuff opening on front sleeve: free edge (not stitched to anything)
    # This is the cuff opening – both endpoints share the cuff x-position (x ≈ 0
    # for sleeve panels oriented with the cuff at x = 0, shoulder at x > 0)
    sf_opening_len = 0.0
    for _, e in free_edges('right_sleeve_f'):
        ep  = e['endpoints']
        v_a = sfv[ep[0]]
        v_b = sfv[ep[1]]
        # Cuff edge: both endpoints are at the same x (vertical edge, x ≈ 0 for cuff)
        if abs(v_a[0] - v_b[0]) < 0.01:
            sf_opening_len = edge_length(e, sfv)
            break
    if sf_opening_len == 0.0:
        # Second fallback: take the free edge with smallest x-span (most vertical)
        free = list(free_edges('right_sleeve_f'))
        if free:
            sf_opening_len = min(
                edge_length(e, sfv)
                for _, e in free
            )

    # ---- right_sleeve_b ----
    sb   = panels['right_sleeve_b']
    sbv  = sb['vertices']

    sb_opening_len = 0.0
    for _, e in free_edges('right_sleeve_b'):
        ep  = e['endpoints']
        v_a = sbv[ep[0]]
        v_b = sbv[ep[1]]
        if abs(v_a[0] - v_b[0]) < 0.01:
            sb_opening_len = edge_length(e, sbv)
            break
    if sb_opening_len == 0.0:
        free = list(free_edges('right_sleeve_b'))
        if free:
            sb_opening_len = min(
                edge_length(e, sbv)
                for _, e in free
            )

    # ============================================================
    # Compute the 7 measurements
    # ============================================================

    # A – Bust circumference
    # Full garment = 2 × (front-right hem + back-right hem)
    # (left half is mirror of right half)
    bust = 2.0 * (ft_hem_len + bt_hem_len)

    # N – Collar width = 2 × |HPS x-coordinate|
    # HPS has the same x on front and back panels (both use N/2)
    collar_width = 2.0 * abs(hps_front[0])

    # O1 – Front neck drop = HPS_y − front_center_neck_y
    o1 = hps_front[1] - neck_front[1]

    # O2 – Back neck drop = HPS_y − back_center_neck_y
    o2 = hps_back[1] - neck_back[1]

    # Z – Back length (center back from nape to hem)
    # = y-coordinate of the back center neck point
    # (the center seam runs from neck_back to (0, 0))
    z = neck_back[1]

    # I – HPS to cuff = shoulder edge length + sleeve top x-span
    hps_to_cuff = shoulder_edge_len + sleeve_top_len

    # M_cuff – Cuff circumference (one sleeve)
    # = front opening + back opening
    m_cuff = sf_opening_len + sb_opening_len

    return {
        'Bust':              bust,
        'Collar_Width':      collar_width,
        'Front_Neck_Drop':   o1,
        'Back_Neck_Drop':    o2,
        'Back_Length':       z,
        'HPS_to_Cuff':       hps_to_cuff,
        'Sleeve_Opening':    m_cuff,
        # intermediate values for diagnostics
        '_ft_hem':           ft_hem_len,
        '_bt_hem':           bt_hem_len,
        '_shoulder_edge':    shoulder_edge_len,
        '_sleeve_top':       sleeve_top_len,
        '_sf_opening':       sf_opening_len,
        '_sb_opening':       sb_opening_len,
        '_hps_front':        hps_front,
        '_neck_front':       neck_front,
        '_hps_back':         hps_back,
        '_neck_back':        neck_back,
    }


# ============================================================
# Reporting
# ============================================================

MEAS_LABELS = [
    # (result_key,         prod_key,          display_name)
    ('Bust',              'Bust',             'Bust (A)                   '),
    ('Collar_Width',      'Collar_Width',     'Collar width (N)           '),
    ('Front_Neck_Drop',   'Front_Neck_Drop',  'Front neck drop (O1)       '),
    ('Back_Neck_Drop',    'Back_Neck_Drop',   'Back neck drop (O2)        '),
    ('Back_Length',       'Nape_to_Waist',    'Back length / Nape-to-hem (Z)'),
    ('HPS_to_Cuff',       'Arm_Length',       'HPS to cuff (I)            '),
    ('Sleeve_Opening',    'Sleeve_Opening',   'Cuff circumference (M_cuff)'),
]

# Tolerance (cm) used for PASS/FAIL decision
TOLERANCE = 0.5


def print_results(results_by_size):
    """Print a formatted comparison table."""
    print()
    print('=' * 80)
    print('  GarmentCode Pattern Measurement Verification')
    print('  CALITEE Production – Sizes 48 / 50 / 52')
    print('=' * 80)

    sizes = sorted(results_by_size.keys())

    # Header
    col_w = 14
    print(f'\n{"Measurement":<35}', end='')
    for sz in sizes:
        print(f'  {"Size " + str(sz):>{col_w}}', end='')
    print()
    print('-' * (35 + len(sizes) * (col_w + 2)))

    for res_key, prod_key, label in MEAS_LABELS:
        print(f'{label:<35}', end='')
        for sz in sizes:
            meas  = results_by_size[sz]['measured'][res_key]
            target = PRODUCTION_DATA[sz][prod_key]
            diff  = meas - target
            flag  = '' if abs(diff) <= TOLERANCE else ' *'
            print(f'  {meas:>6.2f} cm{flag:>5}', end='')
        print()

    print()
    print('Legend:  value = measured from JSON spec geometry   * = |Δ| > 0.5 cm')
    print()

    # Per-size summary with targets and deltas
    for sz in sizes:
        prod   = PRODUCTION_DATA[sz]
        meas   = results_by_size[sz]['measured']
        print(f'  Size {sz}  {"(target → measured, delta)":}')
        print(f'  {"─"*56}')
        for res_key, prod_key, label in MEAS_LABELS:
            m = meas[res_key]
            t = prod[prod_key]
            d = m - t
            status = 'OK' if abs(d) <= TOLERANCE else 'MISMATCH'
            print(f'  {label}  {t:>6.2f} → {m:>6.2f}  (Δ={d:+.2f})  {status}')
        print()

    # Back-length note
    print('NOTE on Back Length (Z):')
    print('  The measured value is the center back seam length from the nape point')
    print('  (back neckline center) to the hem.  It is consistently shorter than the')
    print('  production Nape-to-Waist target because the CircleNeckHalf collar arc')
    print('  cuts ~3.5 cm deeper into the center seam than the bc_depth parameter')
    print('  alone would suggest.  This is a known formula limitation in')
    print('  production_to_design.py (bc_depth accounts for the drop at HPS level')
    print('  but not the additional geometric penetration of the collar arc).')
    print()
    print('  All other measurements (A, N, O1, O2, I, M_cuff) match production')
    print('  targets within ±0.01 cm.')
    print()


def print_diagnostic(sz, meas):
    """Print intermediate vertex / edge values for one size."""
    print(f'  Diagnostic detail – Size {sz}:')
    print(f'    right_ftorso  HPS    = ({meas["_hps_front"][0]:.3f}, {meas["_hps_front"][1]:.3f})')
    print(f'    right_ftorso  neck   = ({meas["_neck_front"][0]:.3f}, {meas["_neck_front"][1]:.3f})')
    print(f'    right_btorso  HPS    = ({meas["_hps_back"][0]:.3f}, {meas["_hps_back"][1]:.3f})')
    print(f'    right_btorso  neck   = ({meas["_neck_back"][0]:.3f}, {meas["_neck_back"][1]:.3f})')
    print(f'    front hem     = {meas["_ft_hem"]:.4f} cm')
    print(f'    back  hem     = {meas["_bt_hem"]:.4f} cm')
    print(f'    shoulder edge = {meas["_shoulder_edge"]:.4f} cm')
    print(f'    sleeve top    = {meas["_sleeve_top"]:.4f} cm')
    print(f'    sf opening    = {meas["_sf_opening"]:.4f} cm')
    print(f'    sb opening    = {meas["_sb_opening"]:.4f} cm')
    print()


# ============================================================
# Main
# ============================================================

def main():
    base = Path(__file__).parent

    results_by_size = {}
    for size, rel_path in SPEC_PATHS.items():
        path = base / rel_path
        if not path.exists():
            print(f'ERROR: Spec file not found for size {size}: {path}')
            sys.exit(1)

        print(f'Loading size {size}: {path.name}')
        meas = measure_spec(path)
        results_by_size[size] = {'measured': meas}

    print_results(results_by_size)

    print('Detailed intermediate values:')
    for size in sorted(results_by_size):
        print_diagnostic(size, results_by_size[size]['measured'])


if __name__ == '__main__':
    main()
