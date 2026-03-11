"""
Custom Pants Generation & Simulation Pipeline
Uses the BUILT-IN GarmentCode parametric system (MetaGarment + Pants + StraightWB).

Production data: HM Wide High Waist Jeans
Body: global_women_size36_apose.obj
Simulation body mesh: global_women_size36_apose.obj (custom SMPL, 168cm)

Measurements from techpack:
  - Waist: "Waist top of WB"
  - Low Hip: "Seat"
  - Inseam: "Inseam"
  - Thigh: "Thigh"
  - Knee: "Knee"
  - Ankle: "Bottom leg"
  - Front rise: "Front rise"
  - Back rise: "Back rise"
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from pathlib import Path
from datetime import datetime
import json
import xml.etree.ElementTree as ET

import numpy as np
import yaml
import cairosvg
import svgpathtools as svgpath
from scipy.spatial.transform import Rotation as Rotation3D

import pygarment as pyg
from pygarment.data_config import Properties
from production_to_design import ProductionToDesign


# ============================================================
# Production measurements per garment size (full circumference, cm)
# ============================================================
GARMENT_SIZES = [32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54]
SIZES_TO_RUN  = [38]   # subset to generate/simulate; set to GARMENT_SIZES for all

PRODUCTION_DATA = {
    32: {
        'Waist': 64.8,
        'Low_Hip': 86.5,
        'Inseam': 76.9,
        'Thigh': 54.5,
        'Knee': 46.8,
        'Ankle': 51,
        'Front Rise': 22.6,
        'Back Rise': 30.2,
        'Hip_from_waist': 13.9,   # bottom of waistband → hip line (cm)
        'Knee_from_crotch': 32.1, # crotch (= thigh line) → knee line (cm)
    },
    34: {
        'Waist': 67.8,
        'Low_Hip': 90.5,
        'Inseam': 77.6,
        'Thigh': 57.3,
        'Knee': 48.7,
        'Ankle': 52.5,
        'Front Rise': 23.3,
        'Back Rise': 31.7,
        'Hip_from_waist': 14.1,
        'Knee_from_crotch': 32.4,
    },
    36: {
        'Waist': 71.5,
        'Low_Hip': 94.5,
        'Inseam': 78.3,
        'Thigh': 60.1,
        'Knee': 50.6,
        'Ankle': 54,
        'Front Rise': 24,
        'Back Rise': 33.2,
        'Hip_from_waist': 14.3,
        'Knee_from_crotch': 32.7,
    },
    38: {
        'Waist': 76,
        'Low_Hip': 98.5,
        'Inseam': 79,
        'Thigh': 62.8,
        'Knee': 52.5,
        'Ankle': 55.5,
        'Front Rise': 24.7,
        'Back Rise': 34.7,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 33.0,
    },
    40: {
        'Waist': 79.8,
        'Low_Hip': 101.5,
        'Inseam': 79,
        'Thigh': 64.8,
        'Knee': 53.9,
        'Ankle': 56.4,
        'Front Rise': 25.2,
        'Back Rise': 35.8,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 33.0,
    },
    42: {
        'Waist': 83.6,
        'Low_Hip': 104.5,
        'Inseam': 79,
        'Thigh': 66.8,
        'Knee': 55.3,
        'Ankle': 57.1,
        'Front Rise': 25.7,
        'Back Rise': 36.9,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 33.0,
    },
    44: {
        'Waist': 87.5,
        'Low_Hip': 107.5,
        'Inseam': 78.5,
        'Thigh': 68.8,
        'Knee': 56.7,
        'Ankle': 57.8,
        'Front Rise': 26.2,
        'Back Rise': 38,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 32.5,
    },
    46: {
        'Waist': 93.2,
        'Low_Hip': 110.5,
        'Inseam': 78,
        'Thigh': 70.8,
        'Knee': 57.9,
        'Ankle': 58.6,
        'Front Rise': 26.7,
        'Back Rise': 39.1,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 32.0,
    },
    48: {
        'Waist': 98.9,
        'Low_Hip': 115.1,
        'Inseam': 77.3,
        'Thigh': 73.5,
        'Knee': 58.9,
        'Ankle': 59.4,
        'Front Rise': 27.6,
        'Back Rise': 40.2,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 31.3,
    },
    50: {
        'Waist': 104.6,
        'Low_Hip': 119.7,
        'Inseam': 76.6,
        'Thigh': 76.1,
        'Knee': 59.9,
        'Ankle': 60.2,
        'Front Rise': 28.5,
        'Back Rise': 41.3,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 30.6,
    },
    52: {
        'Waist': 110.3,
        'Low_Hip': 124.3,
        'Inseam': 75.8,
        'Thigh': 78.8,
        'Knee': 61.5,
        'Ankle': 61,
        'Front Rise': 29.4,
        'Back Rise': 42.4,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 29.9,
    },
    54: {
        'Waist': 116,
        'Low_Hip': 128.9,
        'Inseam': 75,
        'Thigh': 81.4,
        'Knee': 63.1,
        'Ankle': 61.8,
        'Front Rise': 30.3,
        'Back Rise': 43.6,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 29.2,
    },
    56: {
        'Waist': 121.7,
        'Low_Hip': 133.5,
        'Inseam': 74.3,
        'Thigh': 83.9,
        'Knee': 64.7,
        'Ankle': 62.7,
        'Front Rise': 31.2,
        'Back Rise': 44.8,
        'Hip_from_waist': 14.5,
        'Knee_from_crotch': 28.5,
    },
}

BODY_YAML = './assets/bodies/global_women_size36_apose.yaml'
BODY_NAME = 'global_women_size36_apose'


def _find_panel_edge(panel, stitches, role):
    """
    Find edge index in a panel by structural role.
    role:
      'wb_top'    – free edge at max-y (waistband top)
      'pant_hem'  – free edge at min-y (hem / ankle)
      'inseam'    – (unused; inseam edge found dynamically in annotate_pattern_svg)
      'f_rise'    – stitched to pant_b_l e11  (always e1 for pant_f_l)
      'b_rise'    – stitched to pant_f_l e1   (always e11 for pant_b_l)
    """
    panel_name = None
    for pn, pd in {'wb_front': panel, 'pant_f_l': panel, 'pant_b_l': panel}.items():
        pass  # just duck-type by using panel dict directly

    verts = panel['vertices']
    edges = panel['edges']
    ys = [v[1] for v in verts]
    min_y, max_y = min(ys), max(ys)

    # Build stitched set for this panel (passed via stitches list)
    stitched = {}  # edge_idx -> partner (panel_name, edge_idx)
    for s in stitches:
        if len(s) == 2:
            for a, b in [(s[0], s[1]), (s[1], s[0])]:
                stitched[(a['panel'], a['edge'])] = (b['panel'], b['edge'])

    if role == 'wb_top':
        # Free edge with both endpoints at max y
        for i, e in enumerate(edges):
            ep = e['endpoints']
            if (abs(verts[ep[0]][1] - max_y) < 0.5 and
                    abs(verts[ep[1]][1] - max_y) < 0.5):
                return i
        # Fallback: any free edge not at y=0
        for panel_name_key in ['wb_front', 'wb_back']:
            for i, e in enumerate(edges):
                ep = e['endpoints']
                if (abs(verts[ep[0]][1] - max_y) < 1.0 or
                        abs(verts[ep[1]][1] - max_y) < 1.0):
                    return i

    elif role == 'pant_hem':
        # Free edge with both endpoints at min y
        for i, e in enumerate(edges):
            ep = e['endpoints']
            if (abs(verts[ep[0]][1] - min_y) < 0.5 and
                    abs(verts[ep[1]][1] - min_y) < 0.5):
                return i

    elif role == 'f_rise':
        # Edge stitched to pant_b_l, with NO endpoint at min_y
        for i, e in enumerate(edges):
            partner = stitched.get(('pant_f_l', i))
            if partner and partner[0] == 'pant_b_l':
                ep = e['endpoints']
                if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                        abs(verts[ep[1]][1] - min_y) > 1.0):
                    return i

    elif role == 'b_rise':
        # Edge on pant_b_l stitched to pant_f_l, with NO endpoint at min_y
        for i, e in enumerate(edges):
            partner = stitched.get(('pant_b_l', i))
            if partner and partner[0] == 'pant_f_l':
                ep = e['endpoints']
                if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                        abs(verts[ep[1]][1] - min_y) > 1.0):
                    return i

    elif role == 'br_inner_rise':
        # Edge on pant_b_r stitched to pant_f_r, with NO endpoint at min_y
        for i, e in enumerate(edges):
            partner = stitched.get(('pant_b_r', i))
            if partner and partner[0] == 'pant_f_r':
                ep = e['endpoints']
                if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                        abs(verts[ep[1]][1] - min_y) > 1.0):
                    return i

    return None


def _svg_line(parent, ns, p1, p2, color, width):
    line = ET.SubElement(parent, f'{{{ns}}}line')
    line.set('x1', f'{p1[0]:.3f}')
    line.set('y1', f'{p1[1]:.3f}')
    line.set('x2', f'{p2[0]:.3f}')
    line.set('y2', f'{p2[1]:.3f}')
    line.set('stroke', color)
    line.set('stroke-width', width)


def _panel_verts_to_svg(panel):
    """Transform panel local vertices to SVG coordinate space."""
    verts = np.array(panel['vertices'], dtype=float).copy()
    t = np.array(panel['translation'][:2], dtype=float)
    verts[:, 1] *= -1
    t[1] *= -1
    offset = verts.min(axis=0)
    verts -= offset
    t += offset
    rot3d = Rotation3D.from_euler('XYZ', panel['rotation'], degrees=True)
    res = rot3d.apply([0, 1, 0])
    v2 = res[:2]
    v2_len = np.linalg.norm(v2)
    if v2_len > 1e-6:
        v2 /= v2_len
    cos_a = float(np.clip(np.dot([0.0, 1.0], v2), -1.0, 1.0))
    cross = float(np.cross([0.0, 1.0], v2))
    flat_rot = -float(np.arccos(cos_a) * np.sign(cross) if abs(cross) > 1e-5 else 0.0)
    flat_rot = np.rad2deg(flat_rot)
    if abs(flat_rot) > 0.01:
        origin = verts[0].copy()
        c, s = np.cos(np.deg2rad(flat_rot)), np.sin(np.deg2rad(flat_rot))
        R_mat = np.array([[c, -s], [s, c]])
        verts = (R_mat @ (verts - origin).T).T + origin
    verts += t
    return verts


def _local_to_svg(panel, local_pt):
    """Convert a single local (x, y) panel point to SVG space."""
    verts_all = np.array(panel['vertices'], dtype=float).copy()
    pt = np.array([[local_pt[0], local_pt[1]]], dtype=float)
    t = np.array(panel['translation'][:2], dtype=float)
    # Flip Y
    verts_all[:, 1] *= -1
    pt[:, 1] *= -1
    t[1] *= -1
    offset = verts_all.min(axis=0)
    pt -= offset
    t += offset
    # Rotation
    rot3d = Rotation3D.from_euler('XYZ', panel['rotation'], degrees=True)
    res = rot3d.apply([0, 1, 0])
    v2 = res[:2]
    v2_len = np.linalg.norm(v2)
    if v2_len > 1e-6:
        v2 /= v2_len
    cos_a = float(np.clip(np.dot([0.0, 1.0], v2), -1.0, 1.0))
    cross = float(np.cross([0.0, 1.0], v2))
    flat_rot = -float(np.arccos(cos_a) * np.sign(cross) if abs(cross) > 1e-5 else 0.0)
    flat_rot = np.rad2deg(flat_rot)
    if abs(flat_rot) > 0.01:
        verts_for_origin = verts_all.copy()
        verts_for_origin -= offset
        origin = verts_for_origin[0]
        c, s = np.cos(np.deg2rad(flat_rot)), np.sin(np.deg2rad(flat_rot))
        R_mat = np.array([[c, -s], [s, c]])
        pt = (R_mat @ (pt - origin).T).T + origin
    pt += t
    return pt[0]


def _sample_edge_pts(v1, v2, edge, n=200):
    """Return n+1 points sampled along an edge (straight or curved)."""
    if 'curvature' not in edge:
        return np.array([v1 + (v2 - v1) * t for t in np.linspace(0, 1, n + 1)])
    curv = edge['curvature']
    ctype = curv['type']
    ts = np.linspace(0, 1, n + 1)
    if ctype == 'quadratic':
        cp = _rel_to_abs(v1, v2, curv['params'][0])
        return np.array([
            (1 - t)**2 * v1 + 2 * (1 - t) * t * cp + t**2 * v2
            for t in ts
        ])
    elif ctype == 'cubic':
        cp1 = _rel_to_abs(v1, v2, curv['params'][0])
        cp2 = _rel_to_abs(v1, v2, curv['params'][1])
        return np.array([
            (1 - t)**3 * v1 + 3*(1-t)**2*t * cp1 + 3*(1-t)*t**2 * cp2 + t**3 * v2
            for t in ts
        ])
    else:  # circle or unknown — sample linearly as approximation
        return np.array([v1 + (v2 - v1) * t for t in ts])


def _panel_x_range_at_y(panel, y_local):
    """Return (x_min, x_max) of the panel boundary at y=y_local (local space)."""
    verts = np.array(panel['vertices'], dtype=float)
    xs = []
    for edge in panel['edges']:
        ep = edge['endpoints']
        v1, v2 = verts[ep[0]], verts[ep[1]]
        pts = _sample_edge_pts(v1, v2, edge)
        for i in range(len(pts) - 1):
            ya, yb = pts[i][1], pts[i + 1][1]
            if min(ya, yb) <= y_local <= max(ya, yb) and abs(yb - ya) > 1e-9:
                t = (y_local - ya) / (yb - ya)
                xs.append(pts[i][0] + t * (pts[i + 1][0] - pts[i][0]))
    return (min(xs), max(xs)) if len(xs) >= 2 else None


def _cross_section_circumference(panels, y_local, above_crotch=False):
    """
    Measure garment circumference at height y_local (local panel coords, y=0 at hem).

    Below crotch (thigh, knee, ankle):
      One leg circumference = pant_f_l width + pant_b_l width at y.

    Above crotch (hip, waist):
      Full garment circumference = 2 × (pant_f_l + pant_b_l) width at y,
      because both left and right panels (mirror images) contribute equally.
    """
    total = 0.0
    for pname in ('pant_f_l', 'pant_b_l'):
        panel = panels.get(pname)
        if panel is None:
            continue
        xr = _panel_x_range_at_y(panel, y_local)
        if xr is not None:
            total += xr[1] - xr[0]
    return total * (2.0 if above_crotch else 1.0)


def _rise_seam_arc_length(panels, stitches, panel_name, partner_name):
    """Total arc length of the rise seam on panel_name stitched to partner_name."""
    stitched = {}
    for s in stitches:
        if len(s) == 2:
            for a, b in [(s[0], s[1]), (s[1], s[0])]:
                stitched[(a['panel'], a['edge'])] = (b['panel'], b['edge'])
    panel = panels[panel_name]
    verts = panel['vertices']
    min_y = min(v[1] for v in verts)
    total = 0.0
    for i, e in enumerate(panel['edges']):
        partner = stitched.get((panel_name, i))
        if partner and partner[0] == partner_name:
            ep = e['endpoints']
            if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                    abs(verts[ep[1]][1] - min_y) > 1.0):
                total += edge_length(e, verts)
    return total


def _find_panel_rise_seam_endpoints(panels, stitches, panel_name, partner_name):
    """
    Find the top and bottom LOCAL-space points of the rise seam on `panel_name`
    that is stitched to `partner_name`, excluding edges that touch the hem (min_y).

    Used to span F.Rise (pant_f_l ↔ pant_f_r, center-front seam)
    and B.Rise (pant_b_l ↔ pant_b_r, center-back seam).

    Returns (top_local, bottom_local) as np.float64 arrays, or (None, None).
    """
    stitched = {}
    for s in stitches:
        if len(s) == 2:
            for a, b in [(s[0], s[1]), (s[1], s[0])]:
                stitched[(a['panel'], a['edge'])] = (b['panel'], b['edge'])

    panel = panels[panel_name]
    verts = np.array(panel['vertices'], dtype=float)
    min_y = verts[:, 1].min()

    pts = []
    for i, e in enumerate(panel['edges']):
        partner = stitched.get((panel_name, i))
        if partner and partner[0] == partner_name:
            ep = e['endpoints']
            if (abs(verts[ep[0]][1] - min_y) > 1.0 and
                    abs(verts[ep[1]][1] - min_y) > 1.0):
                pts.extend([verts[ep[0]], verts[ep[1]]])

    if not pts:
        return None, None
    pts = np.array(pts)
    top    = pts[pts[:, 1].argmax()]   # highest local y → waist area
    bottom = pts[pts[:, 1].argmin()]   # lowest  local y → crotch extension tip
    return top, bottom


def _compute_back_shift(panels):
    """Compute the x-shift applied to back panels in the SVG."""
    front_max_x = -np.inf
    back_min_x = np.inf
    for p in panels.values():
        sv = _panel_verts_to_svg(p)
        if p['translation'][2] >= 0:
            front_max_x = max(front_max_x, sv[:, 0].max())
        else:
            back_min_x = min(back_min_x, sv[:, 0].min())
    if front_max_x == -np.inf or back_min_x == np.inf:
        return 0.0
    return front_max_x - back_min_x + 10.0


def _draw_dim_annotation(root, ns, p1, p2, label, offset_dist, fontsize, color,
                         centroid=None, forced_normal=None):
    """Draw a dimension line annotation between two SVG points."""
    direction = p2 - p1
    dlen = np.linalg.norm(direction)
    if dlen < 0.1:
        return
    dir_unit = direction / dlen
    normal_ccw = np.array([-dir_unit[1], dir_unit[0]])
    mid = (p1 + p2) / 2

    if forced_normal is not None:
        normal = np.array(forced_normal, dtype=float)
        n_len = np.linalg.norm(normal)
        if n_len > 1e-9:
            normal /= n_len
    elif centroid is not None:
        outward = mid - centroid
        normal = normal_ccw if np.dot(normal_ccw, outward) >= 0 else -normal_ccw
    else:
        normal = normal_ccw

    a1 = p1 + normal * offset_dist
    a2 = p2 + normal * offset_dist
    a_mid = mid + normal * offset_dist
    tick = 2.5

    g = ET.SubElement(root, f'{{{ns}}}g')
    _svg_line(g, ns, a1, a2, color, '0.3')
    for ap in [a1, a2]:
        _svg_line(g, ns, ap - dir_unit * tick / 2, ap + dir_unit * tick / 2, color, '0.3')

    angle_deg = np.rad2deg(np.arctan2(direction[1], direction[0]))
    if angle_deg > 90 or angle_deg < -90:
        angle_deg += 180

    text = ET.SubElement(g, f'{{{ns}}}text')
    text.set('x', f'{a_mid[0]:.3f}')
    text.set('y', f'{a_mid[1]:.3f}')
    text.set('fill', color)
    text.set('font-size', str(fontsize))
    text.set('font-weight', 'bold')
    text.set('text-anchor', 'middle')
    text.set('dominant-baseline', 'middle')
    text.set('transform', f'rotate({angle_deg:.2f},{a_mid[0]:.3f},{a_mid[1]:.3f})')
    text.text = label


def annotate_pattern_svg(svg_path, spec_path, measurements, offset_dist=4.0, fontsize=3.5):
    """
    Add dimension line annotations for garment measurements to the pattern SVG.

    Edge annotations (production targets):
      wb_front top edge              → Waist
      pant_f_l 'left' edge (inseam) → Inseam (left side of front panel)
      pant_f_l hem edge             → Ankle (right side)
      pant_f_l↔pant_f_r seam chain → F.Rise (left side of front panel)
      pant_b_l↔pant_b_r seam chain → B.Rise (right side of back panel)
        Both rise annotations span from the waist-seam inner corner to the
        crotch-extension tip, matching the actual arc path and measurement point.

    Cross-section annotations (ACTUAL measured values from Bézier geometry):
      y = fl_max_y − Hip_from_waist → Hip  (×2, both legs)
      y = crotch_y                  → Thigh (one leg)
      y = crotch_y − Knee_from_crotch → Knee (one leg)
    """
    with open(spec_path) as f:
        spec = json.load(f)
    panels = spec['pattern']['panels']

    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    ET.register_namespace('ev', 'http://www.w3.org/2001/xml-events')
    tree = ET.parse(str(svg_path))
    root = tree.getroot()
    ns = 'http://www.w3.org/2000/svg'
    color = 'rgb(0,100,160)'

    # Expand SVG viewBox to accommodate annotations
    margin = offset_dist + fontsize + 3
    vb = root.get('viewBox').split()
    vb_x, vb_y, vb_w, vb_h = float(vb[0]), float(vb[1]), float(vb[2]), float(vb[3])
    vb_x -= margin;  vb_y -= margin;  vb_w += 2 * margin;  vb_h += 2 * margin
    root.set('viewBox', f'{vb_x} {vb_y} {vb_w} {vb_h}')
    root.set('width', f'{vb_w}cm')
    root.set('height', f'{vb_h}cm')

    back_shift = _compute_back_shift(panels)
    stitches = spec['pattern']['stitches']

    # ── Resolve edge indices dynamically (dart count varies by size) ─────────
    wb_top_idx = _find_panel_edge(panels['wb_front'], stitches, 'wb_top')
    hem_idx    = _find_panel_edge(panels['pant_f_l'], stitches, 'pant_hem')

    # Find inseam edge(s) on pant_f_l: the 'inside' interface in pants.py.
    # With subcurves this can be 3 edges (crotch→thigh, thigh→knee, knee→ankle).
    # Inseam edges are on the RIGHT side (high X); outseam on the LEFT (low X).
    _stitched_map = {}
    for _s in stitches:
        if len(_s) == 2:
            for _a, _b in [(_s[0], _s[1]), (_s[1], _s[0])]:
                _stitched_map[(_a['panel'], _a['edge'])] = (_b['panel'], _b['edge'])
    _fl = panels['pant_f_l']
    _fl_verts = _fl['vertices']
    _fl_centroid_x = np.mean([v[0] for v in _fl_verts])
    _inseam_edge_indices = []
    for _i, _e in enumerate(_fl['edges']):
        _partner = _stitched_map.get(('pant_f_l', _i))
        if _partner and _partner[0] == 'pant_b_l':
            _ep = _e['endpoints']
            _v1, _v2 = _fl_verts[_ep[0]], _fl_verts[_ep[1]]
            _mean_x = (_v1[0] + _v2[0]) / 2
            _dy = abs(_v1[1] - _v2[1])
            if _mean_x > _fl_centroid_x and _dy > 1.0:
                _inseam_edge_indices.append(_i)
    inseam_edge_idx = _inseam_edge_indices[0] if _inseam_edge_indices else 0

    # ── Waist / Inseam / Ankle edge annotations ──────────────────────────────
    # Waist and Ankle are annotated on BOTH front and back panels with actual
    # edge lengths (front ≠ back). Inseam is identical on both sides (stitched
    # edges always match), so only annotated once on pant_f_l.
    edge_annots = []

    # Waist: wb_front and wb_back top edges
    for wb_name in ('wb_front', 'wb_back'):
        wb_panel = panels.get(wb_name)
        if wb_panel is None:
            continue
        wb_idx = _find_panel_edge(wb_panel, stitches, 'wb_top')
        if wb_idx is not None:
            w = edge_length(wb_panel['edges'][wb_idx], wb_panel['vertices'])
            edge_annots.append((wb_name, wb_idx, f"Waist: {w:.1f}cm ({2*w:.1f}cm)"))

    # Inseam: annotation spanning full inseam chain on pant_f_l.
    # With subcurves this is multiple edges; draw from ankle to crotch endpoints.
    if len(_inseam_edge_indices) == 1:
        edge_annots.append(('pant_f_l', _inseam_edge_indices[0],
                            f"Inseam: {measurements['Inseam']:.1f}cm"))
    elif _inseam_edge_indices:
        # Find the overall span: lowest and highest Y inseam vertices
        _inseam_pts_y = []
        for _idx in _inseam_edge_indices:
            _ep = _fl['edges'][_idx]['endpoints']
            _inseam_pts_y.append((_fl_verts[_ep[0]][1], _ep[0]))
            _inseam_pts_y.append((_fl_verts[_ep[1]][1], _ep[1]))
        _bottom_vi = min(_inseam_pts_y, key=lambda x: x[0])[1]
        _top_vi = max(_inseam_pts_y, key=lambda x: x[0])[1]
        # Draw directly (bypassing edge_annots loop which expects single edge idx)
        _fl_svg_verts = _panel_verts_to_svg(_fl)
        _p1_is = _fl_svg_verts[_bottom_vi]
        _p2_is = _fl_svg_verts[_top_vi]
        _fl_centroid = _fl_svg_verts.mean(axis=0)
        _draw_dim_annotation(root, ns, _p1_is, _p2_is,
                             f"Inseam: {measurements['Inseam']:.1f}cm",
                             offset_dist, fontsize, color, centroid=_fl_centroid)

    # Ankle: pant_f_l and pant_b_l hem edges
    hem_b_idx = _find_panel_edge(panels['pant_b_l'], stitches, 'pant_hem')
    for hem_panel_name, h_idx in (('pant_f_l', hem_idx), ('pant_b_l', hem_b_idx)):
        if h_idx is None:
            continue
        p = panels[hem_panel_name]
        a = edge_length(p['edges'][h_idx], p['vertices'])
        edge_annots.append((hem_panel_name, h_idx, f"Ankle: {a:.1f}cm ({2*a:.1f}cm)"))

    for panel_name, edge_idx, label in edge_annots:
        panel = panels.get(panel_name)
        if panel is None:
            continue
        is_back = panel['translation'][2] < 0
        sx = back_shift if is_back else 0.0
        verts = _panel_verts_to_svg(panel)
        verts[:, 0] += sx
        edge = panel['edges'][edge_idx]
        i, j = edge['endpoints']
        p1, p2 = verts[i], verts[j]
        centroid = verts.mean(axis=0)
        _draw_dim_annotation(root, ns, p1, p2, label, offset_dist, fontsize, color,
                             centroid=centroid)

    # ── F.Rise: center-front seam (pant_f_l ↔ pant_f_r) ────────────────────
    # Spans from the waist-seam inner corner down to the crotch-extension tip.
    # Placed to the LEFT of the front panel.
    # Label shows ACTUAL arc length, not production target.
    fr_top, fr_bot = _find_panel_rise_seam_endpoints(
        panels, stitches, 'pant_f_l', 'pant_f_r')
    if fr_top is not None:
        actual_f_rise = _rise_seam_arc_length(panels, stitches, 'pant_f_l', 'pant_f_r')
        p1_svg = _local_to_svg(panels['pant_f_l'], fr_top)
        p2_svg = _local_to_svg(panels['pant_f_l'], fr_bot)
        fl_centroid = _panel_verts_to_svg(panels['pant_f_l']).mean(axis=0)
        _draw_dim_annotation(root, ns, p1_svg, p2_svg,
                             f"F.Rise: {actual_f_rise:.1f}cm",
                             offset_dist, fontsize, color, centroid=fl_centroid)

    # ── B.Rise: center-back seam (pant_b_l ↔ pant_b_r) ─────────────────────
    # Spans from the waist-seam inner corner down to the crotch-extension tip.
    # Placed to the RIGHT of the back panel.
    # Label shows ACTUAL arc length, not production target.
    br_top, br_bot = _find_panel_rise_seam_endpoints(
        panels, stitches, 'pant_b_l', 'pant_b_r')
    if br_top is not None:
        actual_b_rise = _rise_seam_arc_length(panels, stitches, 'pant_b_l', 'pant_b_r')
        p1_svg = _local_to_svg(panels['pant_b_l'], br_top)
        p1_svg[0] += back_shift
        p2_svg = _local_to_svg(panels['pant_b_l'], br_bot)
        p2_svg[0] += back_shift
        bl_verts_svg = _panel_verts_to_svg(panels['pant_b_l']).copy()
        bl_verts_svg[:, 0] += back_shift
        _draw_dim_annotation(root, ns, p1_svg, p2_svg,
                             f"B.Rise: {actual_b_rise:.1f}cm",
                             offset_dist, fontsize, color,
                                 centroid=bl_verts_svg.mean(axis=0))

    # ── Cross-section annotations (Hip, Thigh, Knee) on BOTH panels ─────────
    # Each panel is annotated with ITS OWN width at that height.
    # (Showing the total circumference on a single panel's span would be misleading.)
    #
    # Crotch height = U bottom (tip of center-front seam, fr_bot[1] = 78.3cm).
    fl_panel = panels.get('pant_f_l')
    section_heights = []   # list of (y_local, label_prefix)
    if fl_panel is not None:
        fl_verts_loc = np.array(fl_panel['vertices'], dtype=float)
        fl_min_y = fl_verts_loc[:, 1].min()
        fl_max_y = fl_verts_loc[:, 1].max()
        if fr_bot is not None:
            crotch_y = float(fr_bot[1])
        else:
            e0 = fl_panel['edges'][0]
            ep0 = e0['endpoints']
            crotch_y = max(fl_verts_loc[ep0[0]][1], fl_verts_loc[ep0[1]][1])

        section_heights.append((crotch_y, 'Thigh'))

        kfc = measurements.get('Knee_from_crotch')
        if kfc:
            knee_y = crotch_y - kfc
            if knee_y > fl_min_y:
                section_heights.append((knee_y, 'Knee'))

        hfw = measurements.get('Hip_from_waist')
        if hfw:
            hip_y = fl_max_y - hfw
            if hip_y > crotch_y:
                section_heights.append((hip_y, 'Hip'))

    # In SVG coords y increases downward (ankle ≈ +14, waist ≈ -93).
    # Force a consistent annotation side so front and back panels align visually:
    #   below-crotch (Thigh, Knee) → annotate toward ankle  = SVG (0, +1)
    #   above-crotch (Hip)         → annotate toward waist   = SVG (0, -1)
    _forced_normals = {'Thigh': (0, 1), 'Knee': (0, 1), 'Hip': (0, -1)}

    for y_local, prefix in section_heights:
        fnorm = _forced_normals.get(prefix)
        for panel_name in ('pant_f_l', 'pant_b_l'):
            panel = panels.get(panel_name)
            if panel is None:
                continue
            xr = _panel_x_range_at_y(panel, y_local)
            if xr is None:
                continue
            panel_width = xr[1] - xr[0]
            label = f"{prefix}: {panel_width:.1f}cm ({2 * panel_width:.1f}cm)"
            sx = back_shift if panel['translation'][2] < 0 else 0.0
            p1 = _local_to_svg(panel, [xr[0], y_local]).copy()
            p2 = _local_to_svg(panel, [xr[1], y_local]).copy()
            p1[0] += sx
            p2[0] += sx
            _draw_dim_annotation(root, ns, p1, p2, label, offset_dist, fontsize, color,
                                 forced_normal=fnorm)

    svg_bytes = ET.tostring(root, encoding='unicode', xml_declaration=False)
    with open(str(svg_path), 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="utf-8" ?>\n')
        f.write(svg_bytes)

    # Regenerate PNG from annotated SVG
    png_path = Path(str(svg_path).replace('_pattern.svg', '_pattern.png'))
    if png_path != Path(svg_path) and png_path.exists():
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), dpi=2.54 * 3)


def _fit_rise_parameters(prod, body_yaml_path, base_garment_measurements):
    """Numerically fit (rise_v, front_crotch_fraction) to match production F/B rise seam arcs.

    Uses alternating 1D bisection:
      1. Bisect front_crotch_fraction so F.Rise arc == target_f  (front ext controls front arc)
      2. Bisect rise_v                  so B.Rise arc == target_b  (vertical rise controls both)
      3. Repeat until both converge.

    Returns: (rise_v, front_crotch_fraction)
    """
    from assets.garment_programs.meta_garment import MetaGarment
    from assets.bodies.body_params import BodyParameters

    target_f = prod['Front Rise']
    target_b = prod['Back Rise']

    body_params = BodyParameters(body_yaml_path)
    mapper = ProductionToDesign(body_yaml_path)

    def measure(rise_v, front_fraction):
        gm = dict(base_garment_measurements)
        gm['rise'] = rise_v
        design = mapper.map_pants(gm)
        if front_fraction is not None:
            design['pants']['front_crotch_fraction'] = {'v': float(front_fraction)}
        garment = MetaGarment('_fit', body_params, design)
        pattern = garment.assembly()
        panels = pattern.pattern['panels']
        stitches = pattern.pattern['stitches']
        f = _rise_seam_arc_length(panels, stitches, 'pant_f_l', 'pant_f_r')
        b = _rise_seam_arc_length(panels, stitches, 'pant_b_l', 'pant_b_r')
        return f, b

    # Initial rise_v from average formula (used as starting point)
    body = mapper.body
    avg_rise = (target_f + target_b) / 2
    rise_v = float(np.clip((avg_rise - body['crotch_hip_diff']) / body['hips_line'], 0.3, 1.8))
    front_fraction = None  # use pants.py default on first pass

    print(f'  Fitting rise: target F={target_f:.1f}cm, B={target_b:.1f}cm')

    # Minimum viable front_crotch_fraction: small values make the front crotch
    # extension nearly zero, which causes degenerate Bézier curve fitting.
    # 0.10 gives ~1–2cm front extension, which is the practical lower bound.
    FRAC_MIN = 0.10

    prev_fraction = None
    for outer in range(8):
        # --- Step 1: bisect front_crotch_fraction to match F.Rise arc ---
        # Higher fraction → larger front_extention → longer F.Rise arc
        lo, hi = FRAC_MIN, 0.99
        f_lo, _ = measure(rise_v, lo)
        f_hi, _ = measure(rise_v, hi)
        if f_lo > target_f:
            front_fraction = lo  # clamp to minimum
        elif f_hi < target_f:
            front_fraction = hi  # clamp to maximum
        else:
            for _ in range(18):
                mid = (lo + hi) / 2
                f, _ = measure(rise_v, mid)
                if f > target_f:
                    hi = mid
                else:
                    lo = mid
            front_fraction = (lo + hi) / 2

        # --- Step 2: bisect rise_v to match B.Rise arc ---
        # Higher rise_v → taller panel → longer B.Rise arc
        lo, hi = 0.3, 1.8
        _, b_lo = measure(lo, front_fraction)
        _, b_hi = measure(hi, front_fraction)
        if b_lo > target_b:
            rise_v = lo
        elif b_hi < target_b:
            rise_v = hi
        else:
            for _ in range(18):
                mid = (lo + hi) / 2
                _, b = measure(mid, front_fraction)
                if b > target_b:
                    hi = mid
                else:
                    lo = mid
            rise_v = (lo + hi) / 2

        f_now, b_now = measure(rise_v, front_fraction)
        print(f'  iter {outer + 1}: rise_v={rise_v:.4f}, frac={front_fraction:.4f} '
              f'→ F={f_now:.2f}cm (Δ{f_now - target_f:+.2f}), '
              f'B={b_now:.2f}cm (Δ{b_now - target_b:+.2f})')
        if abs(f_now - target_f) < 0.05 and abs(b_now - target_b) < 0.05:
            print(f'  Converged in {outer + 1} iterations.')
            break
        # If fraction hit the lower bound and didn't change, further iterations won't help
        if front_fraction == FRAC_MIN and prev_fraction == FRAC_MIN:
            print(f'  NOTE: F.Rise target {target_f:.1f}cm not achievable by this model '
                  f'(best: {f_now:.1f}cm); constrained by crotch shape.')
            break
        prev_fraction = front_fraction

    return rise_v, front_fraction


def map_production_to_design(prod, body_yaml_path):
    """Map production measurements to GarmentCode design parameters."""
    mapper = ProductionToDesign(body_yaml_path)
    body = mapper.body

    # Panel length ≈ inseam + crotch_hip_diff
    panel_length = prod['Inseam'] + body['crotch_hip_diff']

    garment_measurements = {
        'waist_circumference': prod['Waist'],
        'hip_circumference': prod['Low_Hip'],
        'length': panel_length,
        'leg_opening': prod['Ankle'],
        'rise': 1.0,  # placeholder; overridden below
        'thigh_circumference': prod.get('Thigh'),
        'knee_circumference': prod.get('Knee'),
        'crotch_to_knee': prod.get('Knee_from_crotch'),
    }

    if 'Front Rise' in prod and 'Back Rise' in prod:
        # Fit rise_v and front_crotch_fraction to match production seam arc targets
        rise_v, front_fraction = _fit_rise_parameters(prod, body_yaml_path, garment_measurements)
        garment_measurements['rise'] = rise_v
        design = mapper.map_pants(garment_measurements)
        design['pants']['front_crotch_fraction'] = {'v': float(front_fraction)}
    else:
        # Fall back: use average of available rise values
        rises = [v for k, v in prod.items() if 'Rise' in k and v is not None]
        rise_v = (sum(rises) / len(rises) - body['crotch_hip_diff']) / body['hips_line'] if rises else 1.0
        garment_measurements['rise'] = float(np.clip(rise_v, 0.5, 1.5))
        design = mapper.map_pants(garment_measurements)

    return design


def generate_pattern(size, design, body_yaml_path, output_base, name_prefix=''):
    """Generate pattern using built-in MetaGarment system."""
    from assets.garment_programs.meta_garment import MetaGarment
    from assets.bodies.body_params import BodyParameters

    garment_name = f'{name_prefix}hm_pants_size{size}' if name_prefix else f'hm_pants_size{size}'

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


def simulate_pattern(pattern_folder, garment_name, output_base):
    """Run physics simulation on A-pose body."""
    from pygarment.meshgen.boxmeshgen import BoxMesh
    from pygarment.meshgen.simulation import run_sim
    from pygarment.meshgen.sim_config import PathCofig

    sim_config_path = './assets/Sim_props/pants_sim_props.yaml'
    props = Properties(sim_config_path)
    props.set_section_stats(
        'sim', fails={}, sim_time={}, spf={},
        fin_frame={}, body_collisions={}, self_collisions={}
    )
    props.set_section_stats('render', render_time={})

    spec_files = list(pattern_folder.glob('*_specification.json'))
    if not spec_files:
        print(f'  ERROR: No specification file found in {pattern_folder}')
        return None
    spec_file = spec_files[0]
    in_name = spec_file.stem.replace('_specification', '')

    paths = PathCofig(
        in_element_path=pattern_folder,
        out_path=output_base,
        in_name=in_name,
        body_name=BODY_NAME,
        smpl_body=True,
        add_timestamp=True
    )

    print(f'  Generating box mesh for {in_name}...')
    resolution_scale = props['sim']['config']['resolution_scale']
    garment_box_mesh = BoxMesh(paths.in_g_spec, resolution_scale)
    garment_box_mesh.load()
    garment_box_mesh.serialize(
        paths, store_panels=False,
        uv_config=props['render']['config']['uv_texture']
    )

    props.serialize(paths.element_sim_props)

    print(f'  Running simulation for {in_name}...')
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
    print(f'  Simulation complete: {in_name} -> {paths.out_el}')
    return paths.out_el


# ============================================================
# Measurement verification from spec JSON
# ============================================================

def _rel_to_abs(start, end, rel_pt):
    """Convert relative Bezier control point to absolute 2D coordinates."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    edge = end - start
    edge_perp = np.array([-edge[1], edge[0]])
    return start + rel_pt[0] * edge + rel_pt[1] * edge_perp


def edge_length(edge_dict, verts):
    """Return the arc-length of a single edge from a panel JSON dict."""
    ep = edge_dict['endpoints']
    v1 = verts[ep[0]]
    v2 = verts[ep[1]]

    if 'curvature' not in edge_dict:
        return float(np.linalg.norm(np.array(v2) - np.array(v1)))

    curv = edge_dict['curvature']
    ctype = curv['type']

    if ctype == 'circle':
        radius, large_arc, sweep = curv['params']
        arc = svgpath.Arc(
            complex(v1[0], v1[1]),
            complex(radius, radius),
            0, int(large_arc), int(sweep),
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
        return float(np.linalg.norm(np.array(v2) - np.array(v1)))


def _panel_edge_lengths(panel):
    """Return list of arc-lengths for every edge of a panel."""
    verts = panel['vertices']
    return [edge_length(e, verts) for e in panel['edges']]


def _free_edges(panels, stitches, panel_name):
    """Return indices and edge dicts of free (unstitched) edges."""
    stitched = set()
    for stitch in stitches:
        for side in stitch:
            stitched.add((side['panel'], side['edge']))
    return [
        (i, panels[panel_name]['edges'][i])
        for i in range(len(panels[panel_name]['edges']))
        if (panel_name, i) not in stitched
    ]


def verify_measurements(spec_path, size, prod):
    """Extract pants measurements from spec JSON and compare to production targets.

    Panel structure (from stitching analysis):
      pant_f_r: e0=bottom(hem), e1=inseam, e2=crotch_bottom, e3=crotch_top,
                e4=top(waist), e5=outside_top, e6=outside_bottom
      pant_b_r: e0=outside_bottom, e1=outside_top, e2..e9=top(with darts),
                e10=crotch_top, e11=inseam, e12=bottom(hem)
      wb_front: lower_interface on e3 (top edge)
      wb_back:  lower_interface on e1 (top edge)
    """
    with open(spec_path, 'r') as f:
        spec = json.load(f)

    panels = spec['pattern']['panels']
    stitches = spec['pattern']['stitches']

    # Build stitching map for edge identification
    stitched = set()
    stitch_map = {}  # (panel, edge) -> (other_panel, other_edge)
    for stitch in stitches:
        for side in stitch:
            stitched.add((side['panel'], side['edge']))
        if len(stitch) == 2:
            a, b = stitch
            stitch_map[(a['panel'], a['edge'])] = (b['panel'], b['edge'])
            stitch_map[(b['panel'], b['edge'])] = (a['panel'], a['edge'])

    results = {}

    # --- Waist circumference ---
    # Find edges with 'lower_interface' label on waistband panels
    waist_circ = 0.0
    for pname in ['wb_front', 'wb_back']:
        panel = panels.get(pname)
        if not panel:
            continue
        for e in panel['edges']:
            if e.get('label') == 'lower_interface':
                waist_circ += edge_length(e, panel['vertices'])
    results['Waist'] = waist_circ if waist_circ > 0 else None

    # --- Leg opening (ankle circumference) ---
    # Free (unstitched) edges on pant panels at y ≈ 0 (hem level)
    # For right leg: pant_f_r and pant_b_r free edges at bottom
    leg_opening = 0.0
    for pname in ['pant_f_r', 'pant_b_r']:
        panel = panels.get(pname)
        if not panel:
            continue
        verts = panel['vertices']
        for i, e in enumerate(panel['edges']):
            if (pname, i) in stitched:
                continue
            ep = e['endpoints']
            v1, v2 = verts[ep[0]], verts[ep[1]]
            # Bottom edge: both endpoints near y = min_y of panel
            min_y = min(v[1] for v in verts)
            if abs(v1[1] - min_y) < 1.0 and abs(v2[1] - min_y) < 1.0:
                leg_opening += edge_length(e, verts)
    # front_bottom + back_bottom = full circumference of one leg opening (no ×2)
    results['Ankle'] = leg_opening if leg_opening > 0 else None

    # --- Inseam ---
    # The inseam is the stitched edge(s) between front and back panels on the inside.
    # With subcurves this can be 3 edges (crotch→thigh, thigh→knee, knee→ankle).
    # Inseam edges are on the RIGHT side (high X) of the panel; outseam on the LEFT
    # (low X). The crotch U-curve is also stitched but is horizontal at crotch level.
    # We identify inseam by: stitched to pant_b_r, mean X > panel centroid X,
    # and NOT horizontal (i.e., has vertical extent — excludes crotch curve).
    inseam = 0.0
    pant_fr = panels.get('pant_f_r')
    pant_br = panels.get('pant_b_r')
    if pant_fr and pant_br:
        fr_verts = pant_fr['vertices']
        centroid_x = np.mean([v[0] for v in fr_verts])
        for i, e in enumerate(pant_fr['edges']):
            partner = stitch_map.get(('pant_f_r', i))
            if partner and partner[0] == 'pant_b_r':
                ep = e['endpoints']
                v1, v2 = fr_verts[ep[0]], fr_verts[ep[1]]
                mean_x = (v1[0] + v2[0]) / 2
                dy = abs(v1[1] - v2[1])
                # Inseam: on the right (high-X) side and has vertical extent
                if mean_x > centroid_x and dy > 1.0:
                    inseam += edge_length(e, fr_verts)
    results['Inseam'] = inseam if inseam > 0 else None

    # --- Cross-section circumferences (Thigh, Knee, Hip) ---
    # Crotch height = U bottom = tip of center-front seam (pant_f_l ↔ pant_f_r).
    # This is the assembly crotch point, crotch_hip_diff (8cm) BELOW the inseam
    # endpoint in the flat pattern.
    fl_panel = panels.get('pant_f_l')
    if fl_panel:
        fl_verts = fl_panel['vertices']
        fl_min_y = min(v[1] for v in fl_verts)
        fl_max_y = max(v[1] for v in fl_verts)
        # Crotch y: bottom of center-front seam = actual crotch seam tip
        _fr_top, _fr_bot = _find_panel_rise_seam_endpoints(panels, stitches, 'pant_f_l', 'pant_f_r')
        if _fr_bot is not None:
            crotch_y = float(_fr_bot[1])
        else:
            e0 = fl_panel['edges'][0]
            ep0 = e0['endpoints']
            crotch_y = max(fl_verts[ep0[0]][1], fl_verts[ep0[1]][1])

        # Thigh: at crotch level (one leg circumference)
        results['Thigh'] = _cross_section_circumference(panels, crotch_y, above_crotch=False)

        # Knee: crotch_y minus the thigh-to-knee distance
        if prod.get('Knee_from_crotch'):
            knee_y = crotch_y - prod['Knee_from_crotch']
            if knee_y > fl_min_y:
                results['Knee'] = _cross_section_circumference(panels, knee_y, above_crotch=False)

        # Hip: fl_max_y (waist seam = bottom of waistband) minus hip_from_waist
        if prod.get('Hip_from_waist'):
            hip_y = fl_max_y - prod['Hip_from_waist']
            if hip_y > crotch_y:
                results['Hip'] = _cross_section_circumference(panels, hip_y, above_crotch=True)

    # --- Rise arc lengths ---
    results['Front Rise'] = _rise_seam_arc_length(panels, stitches, 'pant_f_l', 'pant_f_r')
    results['Back Rise']  = _rise_seam_arc_length(panels, stitches, 'pant_b_l', 'pant_b_r')

    with open(BODY_YAML) as f:
        body_data = yaml.safe_load(f)['body']

    # --- Print results ---
    print(f'\n  {"="*60}')
    print(f'  Measurement Verification – Size {size}')
    print(f'  {"="*60}')
    print(f'  {"Measurement":<25} {"Target":>10} {"Measured":>10} {"Delta":>10}')
    print(f'  {"-"*60}')

    hfw = prod.get('Hip_from_waist')
    kfc = prod.get('Knee_from_crotch')
    checks = [
        ('Waist',      'Waist',      'WB lower_interface edge length'),
        ('Hip',        'Low_Hip',    f'Cross-section at {hfw:.1f}cm below waist seam (×2)' if hfw else 'Cross-section at hip height (×2)'),
        ('Thigh',      'Thigh',      'Cross-section at crotch level (one leg)'),
        ('Knee',       'Knee',       f'Cross-section at {kfc:.1f}cm below crotch (one leg)' if kfc else 'Cross-section at knee height (one leg)'),
        ('Inseam',     'Inseam',     'Inseam edge arc length'),
        ('Ankle',      'Ankle',      'Free hem edges (one leg circumference)'),
        ('Front Rise', 'Front Rise', 'Center-front seam arc length'),
        ('Back Rise',  'Back Rise',  'Center-back seam arc length'),
    ]

    for meas_key, prod_key, note in checks:
        measured = results.get(meas_key)
        target = prod.get(prod_key)
        if target is None:
            continue
        if measured is not None:
            delta = measured - target
            flag = '  <-- CHECK' if abs(delta) > 1.5 else ''
            print(f'  {meas_key:<25} {target:>10.1f} {measured:>10.1f} {delta:>+10.1f}{flag}')
            print(f'    {note}')
        else:
            print(f'  {meas_key:<25} {target:>10.1f} {"N/A":>10}')
            print(f'    {note}')

    waist_body = body_data.get('waist', '?')
    print(f'\n  Note: Waist target ({prod["Waist"]}) is production spec; '
          f'body waist is {waist_body}cm (elastic/stretch fit).')

    return results


def save_combined_mesh(sim_folder,
                       body_obj_path=f'./assets/bodies/{BODY_NAME}.obj'):
    """Create combined body + garment mesh."""
    import trimesh

    sim_folder = Path(sim_folder)
    garment_files = list(sim_folder.glob('*_sim.obj'))
    if not garment_files:
        print(f'  No garment mesh found in {sim_folder}')
        return

    garment_path = garment_files[0]
    garment = trimesh.load(str(garment_path), process=False)
    body = trimesh.load(str(body_obj_path), process=False)

    if body.vertices.max() < 3.0:
        body.vertices = body.vertices * 100.0
    min_y = body.vertices[:, 1].min()
    if min_y < 0:
        body.vertices[:, 1] += abs(min_y)

    body_v = np.array(body.vertices)
    body_f = np.array(body.faces)
    garm_v = np.array(garment.vertices)
    garm_f = np.array(garment.faces)
    garm_f_offset = garm_f + len(body_v)

    output_path = sim_folder / 'combined.obj'
    with open(output_path, 'w') as f:
        f.write("# Combined body + garment mesh\n\n")
        for v in body_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for v in garm_v:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        f.write("\ng body\n")
        for face in body_f:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
        f.write("\ng garment\n")
        for face in garm_f_offset:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    print(f'  Combined mesh saved to {output_path}')

    body.visual = trimesh.visual.ColorVisuals(
        mesh=body, face_colors=np.tile([80, 80, 80, 255], (len(body.faces), 1)))
    garment.visual = trimesh.visual.ColorVisuals(
        mesh=garment, face_colors=np.tile([220, 220, 220, 255], (len(garment.faces), 1)))
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name='body')
    scene.add_geometry(garment, node_name='garment')
    glb_path = sim_folder / 'combined.glb'
    scene.export(str(glb_path))
    print(f'  Combined GLB saved to {glb_path}')


# ============================================================
# Main pipeline
# ============================================================

if __name__ == '__main__':
    sys_props = Properties('./system.json')
    output_base = sys_props['output']

    print("=" * 60)
    print("  Custom Pants Pipeline (Built-in Parametric System)")
    print(f"  Simulation body: {BODY_NAME} (A-pose)")
    print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 60)

    # Step 1: Generate patterns for all sizes
    print("\n--- Step 1: Generating patterns ---")
    generated = []
    for size in SIZES_TO_RUN:
        prod = PRODUCTION_DATA[size]
        print(f'\nSize {size}: Waist={prod["Waist"]}, Hip={prod["Low_Hip"]}, '
              f'Inseam={prod["Inseam"]}, Ankle={prod["Ankle"]}')
        design = map_production_to_design(prod, BODY_YAML)
        print(f'  Design: width={design["pants"]["width"]["v"]:.3f}, '
              f'length={design["pants"]["length"]["v"]:.3f}, '
              f'flare={design["pants"]["flare"]["v"]:.3f}, '
              f'thigh={design["pants"]["thigh"]["v"]:.3f}, '
              f'knee={design["pants"]["knee"]["v"]:.3f}')
        folder, name = generate_pattern(size, design, BODY_YAML, output_base)
        generated.append((folder, name, size))

        # Annotate pattern SVG with key measurements
        svg_files = [f for f in folder.glob('*_pattern.svg') if '_print_' not in f.name]
        spec_files = list(folder.glob('*_specification.json'))
        if svg_files and spec_files:
            annot_meas = {
                'Waist': prod['Waist'],
                'Inseam': prod['Inseam'],
                'Ankle': prod['Ankle'],
                'Hip': prod['Low_Hip'],
                'Thigh': prod['Thigh'],
                'Knee': prod['Knee'],
                'Front Rise': prod['Front Rise'],
                'Back Rise': prod['Back Rise'],
            }
            # Optional vertical position data from techpack:
            # Hip_from_waist: distance from waistband seam down to hip measurement line
            # Knee_from_crotch: distance from crotch level down to knee measurement line
            if 'Hip_from_waist' in prod:
                annot_meas['Hip_from_waist'] = prod['Hip_from_waist']
            if 'Knee_from_crotch' in prod:
                annot_meas['Knee_from_crotch'] = prod['Knee_from_crotch']
            annotate_pattern_svg(svg_files[0], spec_files[0], annot_meas)

    # Step 2: Verify measurements from spec JSONs
    print("\n--- Step 2: Verifying pattern measurements ---")
    for folder, name, size in generated:
        spec_files = list(folder.glob('*_specification.json'))
        if spec_files:
            verify_measurements(spec_files[0], size, PRODUCTION_DATA[size])
        else:
            print(f'  No spec file found for size {size}')

    # Step 3: Simulate each pattern on A-pose body
    print("\n--- Step 3: Running simulations (A-pose) ---")
    sim_results = []
    for folder, name, size in generated:
        print(f'\nSimulating size {size}...')
        try:
            sim_folder = simulate_pattern(folder, name, output_base)
            if sim_folder:
                sim_results.append((sim_folder, size))
        except Exception as e:
            print(f'  Simulation failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    # Step 4: Create combined meshes
    print("\n--- Step 4: Creating combined body+garment meshes ---")
    for sim_folder, size in sim_results:
        print(f'\nCombining size {size}...')
        try:
            save_combined_mesh(sim_folder)
        except Exception as e:
            print(f'  Combined mesh failed for size {size}: {e}')
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("=" * 60)
    print("\nGenerated patterns:")
    for folder, name, size in generated:
        print(f"  Size {size}: {folder}")
    print("\nSimulation results:")
    for sim_folder, size in sim_results:
        print(f"  Size {size}: {sim_folder}")
    print(f"\nAll output in: {output_base}")
