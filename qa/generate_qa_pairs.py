#!/usr/bin/env python

import argparse
import json
import math
import random
from dataclasses import dataclass
from typing import Any

# ===========================================================
# Config
# ===========================================================

@dataclass
class QAConfig:
    """Configuration for QA generation."""
    use_vertical_relations: bool = False  # whether to use up/down relations
    dataset: str = "nuscenes"  # dataset type to adjust distance scales

# ===========================================================
# IO helpers
# ===========================================================

def load_scene_graphs(path: str) -> list[dict[str, Any]]:
    """
    Load scene graphs.

    Default: the file is a JSON list, where each element is one scene graph.

    If you use JSONL, replace json.load(...) with:
        data = [json.loads(line) for line in f if line.strip()]
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def load_templates(path: str) -> list[dict[str, Any]]:
    """Load QA templates from a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ===========================================================
# Scene graph helpers
# ===========================================================

def build_object_dict(scene: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return a mapping from node_id to object dict."""
    return {obj["node_id"]: obj for obj in scene["objects"]}


def build_relation_index(scene: dict[str, Any]) -> dict[tuple[str, str], dict[str, int]]:
    """
    Build an index from (from_id, to_id) to relation dict.
    relation dict example:
      {"left":0/1, "right":0/1, "in front of":0/1, "behind":0/1, "up":0/1, "down":0/1}
    """
    idx = {}
    for e in scene.get("edges", []):
        idx[(e["from"], e["to"])] = e["relation"]
    return idx


def invert_relation(rel_B_relative_to_A: dict[str, int]) -> dict[str, int]:
    """
    Given relation of B relative to A, compute relation of A relative to B.
    """
    return {
        "left":       rel_B_relative_to_A.get("right", 0),
        "right":      rel_B_relative_to_A.get("left", 0),
        "in front of": rel_B_relative_to_A.get("behind", 0),
        "behind":     rel_B_relative_to_A.get("in front of", 0),
        "up":         rel_B_relative_to_A.get("down", 0),
        "down":       rel_B_relative_to_A.get("up", 0),
    }


def get_rel_AB(
    obj_a: str,
    obj_b: str,
    rel_index: dict[tuple[str, str], dict[str, int]]
) -> dict[str, int] | None:
    """
    Return relation of B relative to A, combining A->B and B->A edges.

    If (A,B) exists, relation already encodes B relative to A.
    If only (B,A) exists, invert it.
    """
    if (obj_a, obj_b) in rel_index:
        return rel_index[(obj_a, obj_b)]

    if (obj_b, obj_a) in rel_index:
        rel_BA = rel_index[(obj_b, obj_a)]  # A relative to B
        rel_AB = invert_relation(rel_BA)    # B relative to A
        return rel_AB

    return None

def get_center_node_id(scene: dict[str, Any]) -> str | None:
    """
    Identify the center node ID from the scene graph.
    Assumes a star topology where edges originate from the center.
    """
    edges = scene.get("edges", [])
    if not edges:
        # If no edges, fallback or return None depending on logic.
        # Here we return None, causing generators to skip if edges are strictly required for ID.
        return None
    # The 'from' node of the first edge is assumed to be the center.
    return edges[0]["from"]


# ===========================================================
# Text helpers
# ===========================================================

def get_object_text_for_question(obj: dict[str, Any]) -> str:
    """
    Text used to replace <OBJ_A>/<OBJ_B> in the question.

    Priority:
        1. local id
        2. category_name
    """
    attrs = obj.get("attributes", {})
    local_id = obj.get("local_id")
    if local_id:
        return "object " + str(local_id)
    cat = attrs.get("category_name", "object")
    return cat


def coarse_class_name(category_name: str) -> str:
    """
    Extract a coarse class name for <CLASS>:

    - "human.pedestrian.adult" -> "pedestrian"
    - "vehicle.car" -> "car"
    - "human" -> "human"
    """
    parts = category_name.split(".")
    if len(parts) >= 2:
        return parts[1]
    return parts[0]


# ===========================================================
# Geometry helpers
# ===========================================================

def distance_to_camera(obj: dict[str, Any]) -> float:
    """
    Euclidean distance in camera coordinate frame:
      sqrt(x^2 + y^2 + z_cam^2)
    """
    t = obj["attributes"]["translation"]
    x, y, z = t["x"], t["y"], t["z_cam"]
    return math.sqrt(x * x + y * y + z * z)


def volume(obj: dict[str, Any]) -> float:
    """Compute object volume w * l * h."""
    s = obj["attributes"]["size"]
    return s["w"] * s["l"] * s["h"]


def height(obj: dict[str, Any]) -> float:
    """Return object height h."""
    return obj["attributes"]["size"]["h"]


def width(obj: dict[str, Any]) -> float:
    """Return object width w."""
    return obj["attributes"]["size"]["w"]

# Helper: get heading angle in radians in camera frame (X=Right, Z=Front)
# Returns angle in (-pi, pi], where 0 is Front (Z), pi/2 is Right (+X).
def get_heading(origin_obj, target_obj):
    """Return heading angle (radians) from origin to target in camera frame."""
    t_o = origin_obj["attributes"]["translation"]
    t_t = target_obj["attributes"]["translation"]
    dx = t_t["x"] - t_o["x"]
    dz = t_t["z_cam"] - t_o["z_cam"]
    # atan2(y, x) -> here we map (Front=Z) to 0 degrees.
    # Standard math angle from +X axis: atan2(y, x).
    return math.atan2(dx, dz)

# Helper: Normalize angle to (-pi, pi]
def normalize_angle(theta):
    """Normalize angle to the range (-pi, pi]."""
    while theta > math.pi:
        theta -= 2 * math.pi
    while theta <= -math.pi:
        theta += 2 * math.pi
    return theta

def get_8_sector_direction(angle_diff):
    """Map relative angle to one of 8 directions."""
    ad = angle_diff
    abs_d = abs(ad)
    PI_8 = math.pi / 8

    if abs_d <= PI_8:
        return "Front"
    elif abs_d >= 7 * PI_8:
        return "Behind"

    if ad > 0: # Right side
        if PI_8 < ad <= 3 * PI_8:
            return "Front-Right"
        elif 3 * PI_8 < ad <= 5 * PI_8:
            return "Right"
        elif 5 * PI_8 < ad < 7 * PI_8:
            return "Back-Right"
    else: # Left side
        if -3 * PI_8 <= ad < -PI_8:
            return "Front-Left"
        elif -5 * PI_8 <= ad < -3 * PI_8:
            return "Left"
        elif -7 * PI_8 < ad < -5 * PI_8:
            return "Back-Left"

    return "Unknown"

# ===========================================================
# Direction primitives & mapping
# ===========================================================

PRIMITIVES = ["left", "right", "in front of", "behind", "up", "down"]


def get_allowed_direction_sets(use_vertical: bool) -> list[frozenset[str]]:
    """
    Return allowed sets of direction primitives (up to 2 primitives),
    controlled by whether vertical relations are allowed.
    """
    base = [
        frozenset(["left"]),
        frozenset(["right"]),
        frozenset(["in front of"]),
        frozenset(["behind"]),
        frozenset(["in front of", "left"]),
        frozenset(["in front of", "right"]),
        frozenset(["behind", "left"]),
        frozenset(["behind", "right"]),
    ]
    if not use_vertical:
        return base

    vertical_sets = [
        frozenset(["up"]),
        frozenset(["down"]),
        frozenset(["up", "left"]),
        frozenset(["up", "right"]),
        frozenset(["down", "left"]),
        frozenset(["down", "right"]),
        frozenset(["in front of", "up"]),
        frozenset(["in front of", "down"]),
        frozenset(["behind", "up"]),
        frozenset(["behind", "down"]),
    ]
    return base + vertical_sets


DIR_LABELS_MAP = {
    frozenset(["left"]): "to the left of <OBJ_B>",
    frozenset(["right"]): "to the right of <OBJ_B>",
    frozenset(["in front of"]): "in front of <OBJ_B>",
    frozenset(["behind"]): "behind <OBJ_B>",
    frozenset(["up"]): "above <OBJ_B>",
    frozenset(["down"]): "below <OBJ_B>",
    frozenset(["in front of", "left"]): "in front-left of <OBJ_B>",
    frozenset(["in front of", "right"]): "in front-right of <OBJ_B>",
    frozenset(["behind", "left"]): "back-left of <OBJ_B>",
    frozenset(["behind", "right"]): "back-right of <OBJ_B>",
    frozenset(["up", "left"]): "upper-left of <OBJ_B>",
    frozenset(["up", "right"]): "upper-right of <OBJ_B>",
    frozenset(["down", "left"]): "lower-left of <OBJ_B>",
    frozenset(["down", "right"]): "lower-right of <OBJ_B>",
    frozenset(["in front of", "up"]): "in front and above <OBJ_B>",
    frozenset(["in front of", "down"]): "in front and below <OBJ_B>",
    frozenset(["behind", "up"]): "behind and above <OBJ_B>",
    frozenset(["behind", "down"]): "behind and below <OBJ_B>",
}


def primitives_from_relation(rel: dict[str, int]) -> frozenset[str]:
    """Convert a relation bit dict into a set of primitive names that are active (bit=1)."""
    active = [p for p in PRIMITIVES if rel.get(p, 0) == 1]
    return frozenset(active)


# ===========================================================
# DIRECTION: single-question builders
# ===========================================================

def bool_answer_for_direction_template(
    template_q: str,
    rel_AB: dict[str, int]
) -> bool | None:
    """
    Given a direction template question and relation of B relative to A (rel_AB),
    return True/False for "Yes"/"No", or None if cannot interpret.
    """
    q = template_q.lower()

    if "in front of" in q:
        return rel_AB.get("behind", 0) == 1
    if "behind" in q:
        return rel_AB.get("in front of", 0) == 1
    if "to the left of" in q or "on the left of" in q:
        return rel_AB.get("right", 0) == 1
    if "to the right of" in q or "on the right of" in q:
        return rel_AB.get("left", 0) == 1
    if "above" in q:
        return rel_AB.get("down", 0) == 1
    if "below" in q:
        return rel_AB.get("up", 0) == 1
    return None


def generate_one_direction_yesno(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    config: QAConfig,
    max_attempts: int = 100,
) -> dict[str, Any] | None:
    """Generate a single Yes/No direction question from the given template."""
    obj_dict = build_object_dict(scene)
    rel_idx = build_relation_index(scene)
    node_ids = list(obj_dict.keys())
    if len(node_ids) < 2:
        return None

    q_template = template["question"]
    q_low = q_template.lower()
    if (not config.use_vertical_relations) and ("above" in q_low or "below" in q_low):
        return None

    for _ in range(max_attempts):
        a_id, b_id = rng.sample(node_ids, 2)
        rel_AB = get_rel_AB(a_id, b_id, rel_idx)
        if rel_AB is None:
            continue

        bool_ans = bool_answer_for_direction_template(q_template, rel_AB)
        if bool_ans is None:
            continue

        obj_a = obj_dict[a_id]
        obj_b = obj_dict[b_id]
        text_a = get_object_text_for_question(obj_a)
        text_b = get_object_text_for_question(obj_b)

        question = q_template.replace("<OBJ_A>", text_a).replace("<OBJ_B>", text_b)
        options = template["answer"]  # e.g., ["A: Yes", "B: No"]

        answer_label = "A" if bool_ans else "B"
        answer_text = options[0] if bool_ans else options[1]

        return {
            "image": scene["image"],
            "question": question,
            "options": options,
            "answer_label": answer_label,
            "answer_text": answer_text,
            "category": "direction",
        }

    return None


def generate_one_direction_multichoice(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    config: QAConfig,
    max_attempts: int = 100,
) -> dict[str, Any] | None:
    """Generate a single 4-option direction question from the given template."""
    obj_dict = build_object_dict(scene)
    rel_idx = build_relation_index(scene)
    node_ids = list(obj_dict.keys())
    if len(node_ids) < 2:
        return None

    allowed_sets = get_allowed_direction_sets(config.use_vertical_relations)

    for _ in range(max_attempts):
        a_id, b_id = rng.sample(node_ids, 2)
        rel_B_rel_A = get_rel_AB(a_id, b_id, rel_idx)  # B relative to A
        if rel_B_rel_A is None:
            continue

        rel_A_rel_B = invert_relation(rel_B_rel_A)      # A relative to B
        true_set = primitives_from_relation(rel_A_rel_B)
        if len(true_set) == 0 or len(true_set) > 2:
            continue
        if true_set not in allowed_sets or true_set not in DIR_LABELS_MAP:
            continue

        # Distractors: cannot be equal to true_set, nor subset of true_set
        distractor_sets = [
            s for s in allowed_sets
            if s != true_set and not s.issubset(true_set) and s in DIR_LABELS_MAP
        ]
        if len(distractor_sets) < 3:
            continue

        distractor_sets = rng.sample(distractor_sets, 3)
        all_sets = [true_set] + distractor_sets
        rng.shuffle(all_sets)

        obj_a = obj_dict[a_id]
        obj_b = obj_dict[b_id]
        text_a = get_object_text_for_question(obj_a)
        text_b = get_object_text_for_question(obj_b)

        q_template = template["question"]
        question = q_template.replace("<OBJ_A>", text_a).replace("<OBJ_B>", text_b)

        options: list[str] = []
        correct_label = None
        correct_option_text = None

        for idx, dset in enumerate(all_sets):
            label_tpl = DIR_LABELS_MAP[dset]
            direction_text = label_tpl.replace("<OBJ_B>", text_b)
            opt_letter = chr(ord("A") + idx)
            opt_text = f"{opt_letter}: {direction_text}"
            options.append(opt_text)
            if dset == true_set:
                correct_label = opt_letter
                correct_option_text = opt_text

        if correct_label is None or correct_option_text is None:
            continue

        return {
            "image": scene["image"],
            "question": question,
            "options": options,
            "answer_label": correct_label,
            "answer_text": correct_option_text,
            "category": "direction",
        }

    return None


# ===========================================================
# DISTANCE: single-question builders
# ===========================================================

def generate_one_distance_mcq(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    max_attempts: int = 100,
) -> dict[str, Any] | None:
    """
    Generate one distance MCQ question from template.
    Handles both Object-to-Object relative distance and Camera-to-Object distance.
    """
    obj_dict = build_object_dict(scene)
    node_ids = list(obj_dict.keys())

    if len(node_ids) < 2:
        return None

    q_template = template["question"]
    options_template = template["answer"]
    q_low = q_template.lower()
    edges = scene.get("edges", [])

    # CASE 1: Object-to-Object Relative Distance Comparison
    if "physically closer to <obj_a>" in q_low or "physically farther from <obj_a>" in q_low:
        if len(edges) < 2:
            return None

        for _ in range(max_attempts):
            e1 = rng.choice(edges)
            u, v = e1["from"], e1["to"]

            if u not in obj_dict or v not in obj_dict:
                continue

            a_id, b_id = u, v
            dist_ab = float(e1.get("distance", 0.0))

            candidates_c = []
            for e2 in edges:
                u2, v2 = e2["from"], e2["to"]
                target_id = v2

                if (u2 != 'ego' and
                    target_id and
                    target_id != a_id and
                    target_id != b_id and
                    target_id in obj_dict):
                    candidates_c.append((target_id, float(e2.get("distance", 0.0))))

            if not candidates_c:
                continue

            c_id, dist_ac = rng.choice(candidates_c)

            if abs(dist_ab - dist_ac) < 0.5:
                continue

            obj_a = obj_dict[a_id]
            obj_b = obj_dict[b_id]
            obj_c = obj_dict[c_id]

            text_a = get_object_text_for_question(obj_a)
            text_b = get_object_text_for_question(obj_b)
            text_c = get_object_text_for_question(obj_c)

            question = q_template.replace("<OBJ_A>", text_a).replace("<OBJ_B>", text_b).replace("<OBJ_C>", text_c)
            options = [
                opt.replace("<OBJ_A>", text_a).replace("<OBJ_B>", text_b).replace("<OBJ_C>", text_c)
                for opt in options_template
            ]

            if "closer to" in q_low:
                is_b_answer = dist_ab < dist_ac
            else:
                is_b_answer = dist_ab > dist_ac

            answer_label = "A" if is_b_answer else "B"
            answer_text = options[0] if answer_label == "A" else options[1]

            return {
                "image": scene["image"],
                "question": question,
                "options": options,
                "answer_label": answer_label,
                "answer_text": answer_text,
                "category": "distance",
            }

    # CASE 2: Camera-to-Object Distance Comparison
    else:
        for _ in range(max_attempts):
            a_id, b_id = rng.sample(node_ids, 2)
            obj_a = obj_dict[a_id]
            obj_b = obj_dict[b_id]

            d_a = distance_to_camera(obj_a)
            d_b = distance_to_camera(obj_b)

            if abs(d_a - d_b) < 1.0:
                continue

            text_a = get_object_text_for_question(obj_a)
            text_b = get_object_text_for_question(obj_b)

            question = q_template.replace("<OBJ_A>", text_a).replace("<OBJ_B>", text_b)
            options = [
                opt.replace("<OBJ_A>", text_a).replace("<OBJ_B>", text_b)
                for opt in options_template
            ]

            if "closer to the camera" in q_low or "nearest to the camera" in q_low:
                answer_label = "A" if d_a < d_b else "B"
            elif "farther from the camera" in q_low or "farthest from the camera" in q_low:
                answer_label = "A" if d_a > d_b else "B"
            else:
                return None

            answer_text = options[0] if answer_label == "A" else options[1]
            return {
                "image": scene["image"],
                "question": question,
                "options": options,
                "answer_label": answer_label,
                "answer_text": answer_text,
                "category": "distance",
            }

    return None


def generate_one_distance_na(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    max_attempts: int = 100,
) -> dict[str, Any] | None:
    """Generate one distance NAQ question from template."""
    obj_dict = build_object_dict(scene)
    node_ids = list(obj_dict.keys())
    edges = scene.get("edges", [])

    q_template = template["question"]
    q_low = q_template.lower()

    # Object-to-camera distance
    if "from the camera" in q_low:
        if not node_ids:
            return None
        for _ in range(max_attempts):
            node_id = rng.choice(node_ids)
            obj = obj_dict[node_id]
            text_a = get_object_text_for_question(obj)
            question = q_template.replace("<OBJ_A>", text_a)
            d = distance_to_camera(obj)
            answer_text = str(round(d, 2))
            return {
                "image": scene["image"],
                "question": question,
                "options": None,
                "answer_label": None,
                "answer_text": answer_text,
                "category": "distance",
            }

    # Distance between two objects
    elif "distance between" in q_low:
        if not edges:
            return None
        for _ in range(max_attempts):
            e = rng.choice(edges)
            a_id = e["from"]
            b_id = e["to"]
            if a_id not in obj_dict or b_id not in obj_dict:
                continue
            obj_a = obj_dict[a_id]
            obj_b = obj_dict[b_id]
            text_a = get_object_text_for_question(obj_a)
            text_b = get_object_text_for_question(obj_b)
            question = q_template.replace("<OBJ_A>", text_a).replace("<OBJ_B>", text_b)
            d = float(e.get("distance", 0.0))
            answer_text = str(round(d, 2))
            return {
                "image": scene["image"],
                "question": question,
                "options": None,
                "answer_label": None,
                "answer_text": answer_text,
                "category": "distance",
            }

    return None


# ===========================================================
# SIZE: single-question builders
# ===========================================================

def generate_one_size_mcq(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    max_attempts: int = 100,
) -> dict[str, Any] | None:
    """Generate one size MCQ question (larger/taller/wider)."""
    obj_dict = build_object_dict(scene)
    node_ids = list(obj_dict.keys())
    if len(node_ids) < 2:
        return None

    q_template = template["question"]
    options_template = template["answer"]
    q_low = q_template.lower()

    for _ in range(max_attempts):
        a_id, b_id = rng.sample(node_ids, 2)
        obj_a = obj_dict[a_id]
        obj_b = obj_dict[b_id]

        vol_a = volume(obj_a)
        vol_b = volume(obj_b)
        h_a = height(obj_a)
        h_b = height(obj_b)
        w_a = width(obj_a)
        w_b = width(obj_b)

        text_a = get_object_text_for_question(obj_a)
        text_b = get_object_text_for_question(obj_b)

        question = q_template.replace("<OBJ_A>", text_a).replace("<OBJ_B>", text_b)
        options = [
            opt.replace("<OBJ_A>", text_a).replace("<OBJ_B>", text_b)
            for opt in options_template
        ]

        if "larger" in q_low or "bigger" in q_low:
            answer_label = "A" if vol_a > vol_b else "B"
        elif "taller" in q_low:
            answer_label = "A" if h_a > h_b else "B"
        elif "wider" in q_low:
            answer_label = "A" if w_a > w_b else "B"
        else:
            return None

        answer_text = options[0] if answer_label == "A" else options[1]
        return {
            "image": scene["image"],
            "question": question,
            "options": options,
            "answer_label": answer_label,
            "answer_text": answer_text,
            "category": "size",
        }

    return None


def generate_one_size_na(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    max_attempts: int = 100,
) -> dict[str, Any] | None:
    """Generate one size NAQ question (length/height/width) from template."""
    obj_dict = build_object_dict(scene)
    node_ids = list(obj_dict.keys())
    if not node_ids:
        return None

    q_template = template["question"]
    q_low = q_template.lower()

    for _ in range(max_attempts):
        node_id = rng.choice(node_ids)
        obj = obj_dict[node_id]
        s = obj["attributes"]["size"]

        if "long" in q_low or "length" in q_low:
            val = s["l"]
        elif "height" in q_low:
            val = s["h"]
        elif "width" in q_low:
            val = s["w"]
        else:
            return None

        text_a = get_object_text_for_question(obj)
        question = q_template.replace("<OBJ_A>", text_a)
        answer_text = str(round(val, 2))
        return {
            "image": scene["image"],
            "question": question,
            "options": None,
            "answer_label": None,
            "answer_text": answer_text,
            "category": "size",
        }

    return None


# ===========================================================
# RANKING: single-question builder (size/height/distance)
# ===========================================================

def build_perm_to_label_mapping(options_template: list[str]) -> dict[tuple[str, str, str], str]:
    """
    Build a mapping from permutation of ('A','B','C') to option label.

    e.g. "A: <OBJ_A>-<OBJ_B>-<OBJ_C>" -> ("A","B","C") -> "A"
    """
    perm_to_label: dict[tuple[str, str, str], str] = {}
    for opt in options_template:
        if ":" not in opt:
            continue
        label, pattern = opt.split(":", 1)
        label = label.strip()
        pattern = pattern.strip()

        idx_a = pattern.find("<OBJ_A>")
        idx_b = pattern.find("<OBJ_B>")
        idx_c = pattern.find("<OBJ_C>")
        items = []
        if idx_a >= 0:
            items.append((idx_a, "A"))
        if idx_b >= 0:
            items.append((idx_b, "B"))
        if idx_c >= 0:
            items.append((idx_c, "C"))
        if len(items) != 3:
            continue
        items.sort(key=lambda x: x[0])
        perm = tuple([p for _, p in items])  # e.g., ("A","B","C")
        perm_to_label[perm] = label
    return perm_to_label


def generate_one_ranking_mcq(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    max_attempts: int = 100,
) -> dict[str, Any] | None:
    """Generate one ranking MCQ (size/height/distance) from template."""
    obj_dict = build_object_dict(scene)
    node_ids = list(obj_dict.keys())
    if len(node_ids) < 3:
        return None

    q_template = template["question"]
    q_low = q_template.lower()
    options_template = template["answer"]

    if "by size" in q_low:
        metric_type = "volume"
        descending = True
    elif "by height" in q_low or "tallest to shortest" in q_low:
        metric_type = "height"
        descending = True
    elif "by distance from the camera" in q_low or "nearest to farthest" in q_low:
        metric_type = "distance"
        descending = False
    else:
        return None

    perm_to_label = build_perm_to_label_mapping(options_template)
    if not perm_to_label:
        return None

    for _ in range(max_attempts):
        a_id, b_id, c_id = rng.sample(node_ids, 3)
        obj_a = obj_dict[a_id]
        obj_b = obj_dict[b_id]
        obj_c = obj_dict[c_id]

        if metric_type == "volume":
            val_a = volume(obj_a)
            val_b = volume(obj_b)
            val_c = volume(obj_c)
        elif metric_type == "height":
            val_a = height(obj_a)
            val_b = height(obj_b)
            val_c = height(obj_c)
        else:
            val_a = distance_to_camera(obj_a)
            val_b = distance_to_camera(obj_b)
            val_c = distance_to_camera(obj_c)

        metrics = {"A": val_a, "B": val_b, "C": val_c}
        sorted_items = sorted(metrics.items(), key=lambda kv: kv[1], reverse=descending)
        correct_perm = tuple([k for k, _ in sorted_items])

        if correct_perm not in perm_to_label:
            continue

        correct_label = perm_to_label[correct_perm]

        text_a = get_object_text_for_question(obj_a)
        text_b = get_object_text_for_question(obj_b)
        text_c = get_object_text_for_question(obj_c)

        question = (
            q_template
            .replace("<OBJ_A>", text_a)
            .replace("<OBJ_B>", text_b)
            .replace("<OBJ_C>", text_c)
        )

        options: list[str] = []
        correct_option_text: str | None = None
        for opt in options_template:
            if ":" not in opt:
                continue
            label, pat = opt.split(":", 1)
            label = label.strip()
            text = (
                pat
                .replace("<OBJ_A>", text_a)
                .replace("<OBJ_B>", text_b)
                .replace("<OBJ_C>", text_c)
                .strip()
            )
            full_opt = f"{label}: {text}"
            options.append(full_opt)
            if label == correct_label:
                correct_option_text = full_opt

        if correct_option_text is None:
            continue

        return {
            "image": scene["image"],
            "question": question,
            "options": options,
            "answer_label": correct_label,
            "answer_text": correct_option_text,
            "category": "ranking",
        }

    return None


# ===========================================================
# COUNTING: single-question builder
# ===========================================================

def generate_one_count_mcq(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    max_attempts: int = 100,
) -> dict[str, Any] | None:
    """
    Generate one count MCQ question from template.
    
    Constraint: <OBJ_A> is ALWAYS the center node.
    """
    obj_dict = build_object_dict(scene)
    rel_idx = build_relation_index(scene)
    node_ids = list(obj_dict.keys())
    edges = scene.get("edges", [])

    if len(node_ids) < 2:
        return None

    # Identify center node
    center_id = get_center_node_id(scene)
    if center_id is None or center_id not in obj_dict:
        return None

    # We fix a_id to be the center_id
    a_id = center_id
    obj_a = obj_dict[a_id]

    q_template = template["question"]
    q_low = q_template.lower()

    # -----------------------------
    # Type 1: Directional count
    #   ("how many objects are <REL_PHRASE> <OBJ_A>")
    # -----------------------------
    if "<rel_phrase>" in q_low:
        rel_phrases = [
            "to the left of",
            "to the right of",
            "in front of",
            "behind",
        ]

        for _ in range(max_attempts):
            # a_id is fixed, so we only randomize the relation phrase
            rel_phrase = rng.choice(rel_phrases)

            count = 0
            for b_id in node_ids:
                if b_id == a_id:
                    continue

                rel_AB = get_rel_AB(a_id, b_id, rel_idx)
                if rel_AB is None:
                    continue

                if rel_phrase == "to the left of" and rel_AB.get("left", 0) == 1:
                    count += 1
                elif rel_phrase == "to the right of" and rel_AB.get("right", 0) == 1:
                    count += 1
                elif rel_phrase == "in front of" and rel_AB.get("in front of", 0) == 1:
                    count += 1
                elif rel_phrase == "behind" and rel_AB.get("behind", 0) == 1:
                    count += 1

            correct_num = count
            # avoid too many 0-answer questions
            if correct_num == 0 and rng.random() > 0.05:
                continue

            options_nums: set[int] = {correct_num}
            while len(options_nums) < 4:
                distractor = correct_num + rng.randint(-3, 3)
                if distractor >= 0:
                    options_nums.add(distractor)

            options_nums = sorted(list(options_nums))
            rng.shuffle(options_nums)

            text_a = get_object_text_for_question(obj_a)
            question = (
                q_template
                .replace("<REL_PHRASE>", rel_phrase)
                .replace("<OBJ_A>", text_a)
            )

            options = [
                f"{chr(ord('A') + i)}: {num}"
                for i, num in enumerate(options_nums)
            ]
            correct_idx = options_nums.index(correct_num)
            correct_label = chr(ord("A") + correct_idx)
            correct_text = options[correct_idx]

            return {
                "image": scene["image"],
                "question": question,
                "options": options,
                "answer_label": correct_label,
                "answer_text": correct_text,
                "category": "counting",
            }

    # -----------------------------
    # Type 2: Distance-based count
    #   ("within/farther than <DIST> meters")
    # -----------------------------
    elif "<dist>" in q_low:
        if not edges:
            return None

        for _ in range(max_attempts):
            # a_id is fixed to center.
            # We vary the distance threshold.
            dist_threshold = rng.choice([10, 20]) # too large distances!

            count_within = 0
            count_farther = 0

            for e in edges:
                # With star topology, e['from'] is always center (a_id)
                # We just check the edge source to be safe
                if e.get("from") == a_id and e.get("to") in obj_dict:
                    dist = float(e.get("distance", 0.0))
                    if dist <= dist_threshold:
                        count_within += 1
                    else:
                        count_farther += 1

            is_within = "within" in q_low
            correct_num = count_within if is_within else count_farther

            if correct_num == 0 and rng.random() > 0.05:
                continue

            options_nums: set[int] = {correct_num}
            while len(options_nums) < 4:
                distractor = correct_num + rng.randint(-3, 3)
                if distractor >= 0:
                    options_nums.add(distractor)

            options_nums = sorted(list(options_nums))
            rng.shuffle(options_nums)

            text_a = get_object_text_for_question(obj_a)
            question = (
                q_template
                .replace("<DIST>", str(dist_threshold))
                .replace("<OBJ_A>", text_a)
            )

            options = [
                f"{chr(ord('A') + i)}: {num}"
                for i, num in enumerate(options_nums)
            ]
            correct_idx = options_nums.index(correct_num)
            correct_label = chr(ord("A") + correct_idx)
            correct_text = options[correct_idx]

            return {
                "image": scene["image"],
                "question": question,
                "options": options,
                "answer_label": correct_label,
                "answer_text": correct_text,
                "category": "counting",
            }

    return None

# ===========================================================
# DEDUCTION: single-question builders
# ===========================================================

def compute_new_position(
    obj_a: dict[str, Any],
    direction: str,
    distance: float
) -> tuple[float, float, float]:
    """
    Compute the position of a new object placed in the specified direction
    from obj_a at the given distance, in camera coordinates (x, y, z_cam).

    Directions:
      - left/right: x-axis (x decreases/increases)
      - in front of: move towards the camera (z_cam decreases)
      - behind: move away from the camera (z_cam increases)
    """
    t_a = obj_a["attributes"]["translation"]
    x_a, y_a, z_a = t_a["x"], t_a["y"], t_a["z_cam"]

    # We only place the new object in horizontal directions (no vertical)
    direction_vectors = {
        "left": (-1.0, 0.0, 0.0),
        "right": (1.0, 0.0, 0.0),
        # camera at z=0, objects have z_cam > 0; "in front of" = closer to camera
        "in front of": (0.0, 0.0, -1.0),
        "behind": (0.0, 0.0, 1.0),
    }

    if direction not in direction_vectors:
        direction = "right"

    dx, dy, dz = direction_vectors[direction]

    x_new = x_a + distance * dx
    y_new = y_a + distance * dy
    z_new = z_a + distance * dz

    return x_new, y_new, z_new

def euclidean_distance_2d(pos1: tuple[float, float, float], pos2: tuple[float, float, float]) -> float:
    """Calculate 2D Euclidean distance between two positions (ignoring y-axis)."""
    x1, _, z1 = pos1
    x2, _, z2 = pos2
    return math.sqrt((x2 - x1)**2 + (z2 - z1)**2)

def generate_one_deduction_na(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    config: QAConfig,
    max_attempts: int = 100
) -> dict[str, Any] | None:
    """
    Generate one deduction NAQ question from template.
    """
    obj_dict = build_object_dict(scene)
    node_ids = list(obj_dict.keys())

    if len(node_ids) < 2:
        return None

    # Identify center node and non-center nodes
    center_id = get_center_node_id(scene)
    if center_id is None or center_id not in obj_dict:
        return None

    other_ids = [s['to'] for s in scene.get('edges', []) if s['from'] == center_id]
    if not other_ids:
        return None

    # Only horizontal placement directions
    placement_directions = ["left", "right", "in front of", "behind"]

    for _ in range(max_attempts):
        # Sample one non-center object
        other_id = rng.choice(other_ids)

        # Randomly assign roles: who is A (reference for placement) and who is B (target)
        # But ensure the pair is (Center, Other) or (Other, Center)
        if rng.random() < 0.5:
            a_id, b_id = center_id, other_id
        else:
            a_id, b_id = other_id, center_id

        obj_a = obj_dict[a_id]
        obj_b = obj_dict[b_id]

        placement_dir = rng.choice(placement_directions)
        if config.dataset == "nuscenes":
            placement_dist = rng.choice([5, 10, 15, 20]) # larger distances for nuscenes
        elif config.dataset == "scannetpp":
            placement_dist = rng.choice([0.5, 1, 2]) # smaller distances for scannet++
        else:
            placement_dist = rng.choice([0.5, 1, 2]) # default distances for others

        # New hypothetical position of A
        x_new, y_new, z_new = compute_new_position(obj_a, placement_dir, placement_dist)
        pos_new = (x_new, y_new, z_new)

        # Position of B
        t_b = obj_b["attributes"]["translation"]
        pos_b = (t_b["x"], t_b["y"], t_b["z_cam"])

        # Horizontal distance in camera coordinates
        dist_new_to_b = euclidean_distance_2d(pos_new, pos_b)

        if abs(dist_new_to_b - placement_dist) < 1:
            continue  # avoid trivial cases

        text_a = get_object_text_for_question(obj_a)
        text_b = get_object_text_for_question(obj_b)

        q_template = template["question"]
        question = (
            q_template
            .replace("<DIR>", placement_dir)
            .replace("<OBJ_A>", text_a)
            .replace("<DIST>", str(placement_dist))
            .replace("<OBJ_B>", text_b)
        )

        answer_text = str(round(dist_new_to_b, 2))

        return {
            "image": scene["image"],
            "question": question,
            "options": None,
            "answer_label": None,
            "answer_text": answer_text,
            "category": "deduction",
        }

    return None


# ===========================================================
# TRANSFORMATION: single-question builder
# ===========================================================

def generate_one_transformation_mcq(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    config: QAConfig,
    max_attempts: int = 100,
) -> dict[str, Any] | None:
    """
    Generate a Perspective Transformation MCQ with 8-sector granularity.
    
    Constraints:
    1. A is center.
    2. B and C are sampled from edges' "to" nodes (connected to center).
    3. B is NOT directly behind A (to ensure visibility).
    4. C must have a distinct direction from B (e.g., not Front).
    """
    obj_dict = build_object_dict(scene)
    rel_idx = build_relation_index(scene)

    center_id = get_center_node_id(scene)
    if center_id is None or center_id not in obj_dict:
        return None
    a_id = center_id
    obj_a = obj_dict[a_id]

    edges = scene.get("edges", [])
    connected_node_ids = []
    for e in edges:
        if e.get("from") == center_id:
            to_id = e.get("to")
            if to_id and to_id != center_id and to_id in obj_dict:
                connected_node_ids.append(to_id)

    if len(connected_node_ids) < 2:
        return None

    sector_text_map = {
        "Front": "directly in front of you",
        "Behind": "directly behind you",
        "Left": "directly to your left",
        "Right": "directly to your right",
        "Front-Left": "to your front-left",
        "Front-Right": "to your front-right",
        "Back-Left": "to your back-left",
        "Back-Right": "to your back-right"
    }

    all_sectors = list(sector_text_map.keys())

    # --- Ambiguity Control Map ---
    exclusion_map = {
        "Front":       {"Front-Left", "Front-Right"},
        "Front-Right": {"Front", "Right"},
        "Right":       {"Front-Right", "Back-Right"},
        "Back-Right":  {"Right", "Behind"},
        "Behind":      {"Back-Right", "Back-Left"},
        "Back-Left":   {"Behind", "Left"},
        "Left":        {"Back-Left", "Front-Left"},
        "Front-Left":  {"Left", "Front"},
    }

    for _ in range(max_attempts):
        b_id, c_id = rng.sample(connected_node_ids, 2)
        obj_b = obj_dict[b_id]
        obj_c = obj_dict[c_id]

        # Constraint: B not directly behind A
        rel_A_to_B = get_rel_AB(a_id, b_id, rel_idx)
        if rel_A_to_B and rel_A_to_B.get("behind", 0) == 1:
            continue

        h_AB = get_heading(obj_a, obj_b)
        h_AC = get_heading(obj_a, obj_c)
        diff = normalize_angle(h_AC - h_AB)

        true_sector = get_8_sector_direction(diff)

        if true_sector == "Unknown" or true_sector == "Front":
            continue

        # --- Smart Distractor Selection ---
        forbidden_sectors = exclusion_map.get(true_sector, set())

        distractor_pool = [
            s for s in all_sectors
            if s != true_sector and s not in forbidden_sectors
        ]

        if len(distractor_pool) < 3:
            continue

        rng.shuffle(distractor_pool)
        selected_distractors = distractor_pool[:3]

        final_options_keys = [true_sector] + selected_distractors
        rng.shuffle(final_options_keys)

        # Build Output
        text_a = get_object_text_for_question(obj_a)
        text_b = get_object_text_for_question(obj_b)
        text_c = get_object_text_for_question(obj_c)

        q_template = template["question"]
        question = (
            q_template
            .replace("<OBJ_A>", text_a)
            .replace("<OBJ_B>", text_b)
            .replace("<OBJ_C>", text_c)
        )

        options_list = []
        answer_label = None
        answer_text = None

        for idx, key in enumerate(final_options_keys):
            letter = chr(ord('A') + idx)
            opt_str = f"{letter}: {sector_text_map[key]}"
            options_list.append(opt_str)

            if key == true_sector:
                answer_label = letter
                answer_text = opt_str

        return {
            "image": scene["image"],
            "question": question,
            "options": options_list,
            "answer_label": answer_label,
            "answer_text": answer_text,
            "category": "transformation",
        }

    return None

# ===========================================================
# Single-question dispatcher by template
# ===========================================================

def generate_single_question_by_template(
    scene: dict[str, Any],
    template: dict[str, Any],
    rng: random.Random,
    config: QAConfig,
) -> dict[str, Any] | None:
    """
    Dispatch to the appropriate single-question builder based on template's
    category and type.
    """
    category = template.get("category")
    ttype = template.get("type")

    if category == "direction" and ttype == "MCQ":
        answers = template.get("answer", [])
        if isinstance(answers, list) and len(answers) == 2 and not any("<DIR" in a for a in answers):
            return generate_one_direction_yesno(scene, template, rng, config)
        if isinstance(answers, list) and any("<DIR" in a for a in answers):
            return generate_one_direction_multichoice(scene, template, rng, config)
        return None

    if category == "distance":
        if ttype == "MCQ":
            return generate_one_distance_mcq(scene, template, rng)
        if ttype in {"NA", "NAQ"}:
            return generate_one_distance_na(scene, template, rng)
        return None

    if category == "size":
        if ttype == "MCQ":
            return generate_one_size_mcq(scene, template, rng)
        if ttype in {"NA", "NAQ"}:
            return generate_one_size_na(scene, template, rng)
        return None

    if category == "ranking" and ttype == "MCQ":
        return generate_one_ranking_mcq(scene, template, rng)

    if category == "counting" and ttype == "MCQ":
        return generate_one_count_mcq(scene, template, rng)

    if category == "deduction" and ttype in {"NA", "NAQ"}:
        return generate_one_deduction_na(scene, template, rng, config)

    if category == "transformation" and ttype == "MCQ":
        return generate_one_transformation_mcq(scene, template, rng, config)

    return None


# ===========================================================
# Difficulty & per-scene selection
# ===========================================================

def get_difficulty_requirements(num_edges: int) -> tuple[str, dict[str, int]]:
    """
    Determine difficulty and required number of questions per category.

    Easy (1-2 edges):
        size:1, distance:1, direction:1, deduction:1
    Medium (3-7 edges):
        size:2, distance:1, direction:2, ranking:1, counting:1, deduction:1, transformation:1
    Hard (8-11 edges):
        size:3, distance:2, direction:2, ranking:2, counting:1, deduction:1, transformation:2

    <1 edges -> no questions.
    >11 edges -> treat as Hard.
    """
    if num_edges < 1:
        return "none", {}

    if 1 <= num_edges <= 2:
        return "easy", {
            "size": 1,
            "distance": 1,
            "direction": 1,
            "deduction": 1,
        }
    elif 3 <= num_edges <= 7:
        return "medium", {
            "size": 2,
            "distance": 1,
            "direction": 2,
            "ranking": 1,
            "counting": 1,
            "deduction": 1,
            "transformation": 1,
        }
    else:
        return "hard", {
            "size": 3,
            "distance": 2,
            "direction": 2,
            "ranking": 2,
            "counting": 1,
            "deduction": 1,
            "transformation": 2,
        }


def generate_questions_for_scene(
    scene: dict[str, Any],
    templates: list[dict[str, Any]],
    rng: random.Random,
    config: QAConfig,
    max_attempts_per_question: int = 50,
) -> list[dict[str, Any]]:
    """
    For a given scene, directly sample the required number of questions per category:
      1) Determine difficulty & required counts.
      2) For each required question:
         - Randomly pick a template among all templates of that category.
         - Call a single-question builder.
         - Reject candidates whose question text duplicates an existing one in this scene.
    """
    num_objects = len(scene.get("objects", []))
    num_edges = len(scene.get("edges", [])) - num_objects
    difficulty, requirements = get_difficulty_requirements(num_edges)
    if difficulty == "none" or not requirements:
        return []

    # Group templates by category
    templates_by_cat: dict[str, list[dict[str, Any]]] = {}
    for t in templates:
        cat = t.get("category")
        if cat is None:
            continue
        templates_by_cat.setdefault(cat, []).append(t)

    selected: list[dict[str, Any]] = []
    used_questions: set[str] = set()  # dedup within this scene by question text

    for cat, k in requirements.items():
        if k <= 0:
            continue
        cat_templates = templates_by_cat.get(cat, [])
        if not cat_templates:
            continue

        for _ in range(k):
            qa = None
            attempts = 0
            while attempts < max_attempts_per_question:
                attempts += 1
                tmpl = rng.choice(cat_templates)
                candidate = generate_single_question_by_template(scene, tmpl, rng, config)
                if candidate is None:
                    continue
                q_text = candidate.get("question", "").strip()
                if not q_text:
                    continue
                if q_text in used_questions:
                    # duplicated question in this scene, reject and try again
                    continue

                # accept this candidate
                qa = candidate
                used_questions.add(q_text)
                break

            if qa is not None:
                qa["difficulty"] = difficulty
                qa["image_with_2dbox"] = scene.get("image_with_2dbox", "")
                selected.append(qa)
            # if qa is None after attempts, we simply give up this slot,
            # so this category may end up with fewer questions than requested,
            # but we guarantee no duplicate questions within the scene.

    rng.shuffle(selected)
    return selected


# ===========================================================
# ShareGPT conversion
# ===========================================================

def qa_to_sharegpt_item(qa: dict[str, Any], idx: int, sample_token: str) -> dict[str, Any]:
    """
    Convert a QA dict into a ShareGPT-style item:

    {
      "id": "...",
      "image": "...",
      "conversations": [
        {"from": "human", "value": "<question + options>"},
        {"from": "gpt",   "value": "<answer>"}
      ]
    }

    - For MCQ, answer is only the label ("A", "B", ...).
    - For NAQ, answer is the free-form text (e.g., numeric).
    - No meta field.
    """
    image = qa["image_with_2dbox"]
    base_question = qa["question"]
    options = qa.get("options")

    if options:
        question_with_options = base_question + "\nOptions:\n" + "\n".join(options)
    else:
        question_with_options = base_question

    if qa.get("answer_label") is not None and options:
        answer = str(qa["answer_label"])
    else:
        answer = str(qa.get("answer_text", ""))

    qa_id = f"{sample_token}_{idx}"

    return {
        "id": qa_id,
        "image": image,
        "conversations": [
            {"from": "human", "value": question_with_options},
            {"from": "gpt", "value": answer},
        ],
        "meta": {"category": qa.get("category", "unknown"),
                 "difficulty": qa.get("difficulty", "unknown"),
                 "type": "MCQ" if options else "NAQ"},
    }


# ===========================================================
# Main
# ===========================================================

def build_vlm_benchmark(
    scene_graph_path: str,
    template_path: str,
    output_path: str,
    seed: int = 42,
    use_vertical_relations: bool = False,
    dataset: str = "nuscenes",
):
    """Generate a VLM spatial reasoning QA benchmark from scene graphs."""
    rng = random.Random(seed)
    config = QAConfig(use_vertical_relations=use_vertical_relations, dataset=dataset)

    scenes = load_scene_graphs(scene_graph_path)
    templates = load_templates(template_path)

    total_items = 0

    with open(output_path, "w", encoding="utf-8") as out_f:
        for scene in scenes:
            sample_token = scene.get("sample_data_token", "unknown")
            qa_list = generate_questions_for_scene(scene, templates, rng, config)
            for idx, qa in enumerate(qa_list):
                item = qa_to_sharegpt_item(qa, idx, sample_token)
                out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                total_items += 1

    print(f"Total ShareGPT items written: {total_items}")
    print(f"Output written to: {output_path}")


def parse_args():
    """Parse command-line arguments for QA generation."""
    ap = argparse.ArgumentParser(
        description="Generate VLM spatial reasoning QA in ShareGPT format from scene graphs (balanced per difficulty)."
    )
    ap.add_argument(
        "--scene-graphs",
        type=str,
        required=True,
        help="Path to scene graph JSON (list) or JSONL (see comments in load_scene_graphs).",
    )
    ap.add_argument(
        "--templates",
        type=str,
        required=True,
        help="Path to QA template JSON file.",
    )
    ap.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL path in ShareGPT format.",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    ap.add_argument(
        "--use-vertical",
        type=int,
        default=0,
        help="Whether to include vertical (up/down) relations in direction questions. 0/1.",
    )
    ap.add_argument(
        "--dataset",
        default="nuscenes",
        type=str,
        choices=["nuscenes", "scannetpp"],
        help="Dataset type to adjust distance scales accordingly.",
    )

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_vlm_benchmark(
        scene_graph_path=args.scene_graphs,
        template_path=args.templates,
        output_path=args.output,
        seed=args.seed,
        use_vertical_relations=bool(args.use_vertical),
        dataset=args.dataset,
    )
