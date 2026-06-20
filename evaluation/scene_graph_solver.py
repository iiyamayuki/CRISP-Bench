import argparse
import json
import math
import os
import re
from typing import Any

# ==============================================================================
# 1. Template Matcher (Regex Compilation)
# ==============================================================================

class TemplateMatcher:
    """
    Compiles QA templates into regex patterns to extract entities (e.g., <OBJ_A>).
    """
    def __init__(self, templates_list: list[dict[str, Any]]):
        self.templates_map: dict[str, list[re.Pattern]] = {}
        self.patterns = {
            "<OBJ_A>": r"(?P<OBJ_A>.+?)",
            "<OBJ_B>": r"(?P<OBJ_B>.+?)",
            "<OBJ_C>": r"(?P<OBJ_C>.+?)",
            "<DIR>": r"(?P<DIR>.+?)",
            "<REL_PHRASE>": r"(?P<REL_PHRASE>.+?)",
            "<DIST>": r"(?P<DIST>\d+(?:\.\d+)?)",
        }
        self._compile_templates(templates_list)

    def _compile_templates(self, templates_list: list[dict[str, Any]]):
        for tmpl in templates_list:
            category = tmpl.get("category", "unknown")
            q_str = tmpl["question"]
            regex_pattern = re.escape(q_str)
            for placeholder, capture_group in self.patterns.items():
                regex_pattern = regex_pattern.replace(re.escape(placeholder), capture_group)
            regex_pattern = f"^{regex_pattern}\\s*$"

            self.templates_map.setdefault(category, [])
            self.templates_map[category].append(re.compile(regex_pattern, re.IGNORECASE))

    def extract(self, question_text: str, category: str) -> dict[str, str] | None:
        """Match a question against category templates and return captured groups."""
        clean_question = question_text.split("\nOptions:")[0].strip()
        clean_question = clean_question.split("\noptions:")[0].strip()

        if category not in self.templates_map:
            return None

        for pattern in self.templates_map[category]:
            match = pattern.match(clean_question)
            if match:
                return match.groupdict()
        return None


# ==============================================================================
# 2. Scene Graph Solver Logic (ID-based, no translation)
# ==============================================================================

class SceneGraphSolver:
    """
    Solves QA questions using a predicted scene graph in the id-only setting.

    New SG schema:
      {
        "objects": [{"id": int, "dist_to_cam": float, "size": {"w":float,"l":float,"h":float}}],
        "edges":   [{"from": int, "to": int, "distance": float, "relation": ["string"]}]
      }
    """
    MISSING_LABEL = "FAILED: Missing Data"

    def __init__(
        self,
        scene_graph: dict[str, Any],
        matcher: "TemplateMatcher",
    ):
        """Initialize solver from a predicted scene graph and a template matcher."""
        self.sg = scene_graph
        self.matcher = matcher

        # Index objects by id
        self.objects_by_id: dict[int, dict[str, Any]] = {}
        for obj in self.sg.get("objects", []):
            try:
                oid = int(obj.get("id"))
                self.objects_by_id[oid] = obj
            except Exception:
                continue

        # Index edges by (from, to)
        self.edges_map: dict[tuple[int, int], dict[str, Any]] = {}
        for e in self.sg.get("edges", []):
            try:
                u = int(e.get("from"))
                v = int(e.get("to"))
                self.edges_map[(u, v)] = e
            except Exception:
                continue

        # Canonical primitives in your current definition
        # (SG should output ONLY these: left/right/in front of/behind/up/down)
        self.dir_keywords: dict[str, str] = {
            # ===== Horizontal (Left/Right) =====
            "to the left of": "left",
            "on the left of": "left",
            "left of": "left",
            "left": "left",

            "to the right of": "right",
            "on the right of": "right",
            "right of": "right",
            "right": "right",

            # ===== Depth (Front/Behind) =====
            "in front of": "in front of",
            "in front": "in front of",
            "front of": "in front of",
            "front": "in front of",

            "behind": "behind",
            "back of": "behind",
            "back": "behind",

            # ===== Vertical (Up/Down) =====
            "above": "up",
            "over": "up",
            "on top of": "up",
            "upper": "up",
            "up": "up",

            "below": "down",
            "under": "down",
            "underneath": "down",
            "lower": "down",
            "down": "down",
        }

        self.rel_inversion: dict[str, str] = {
            "left": "right",
            "right": "left",
            "in front of": "behind",
            "behind": "in front of",
            "up": "down",
            "down": "up",
        }

    # --------------------------------------------------------------------------
    # Helpers
    # --------------------------------------------------------------------------

    def _normalize_obj_to_id(self, text: str) -> int | None:
        """
        Convert mentions like "Object 3" / "object3" / "3" to int id.
        """
        if not text:
            return None
        t = text.strip()
        m = re.search(r"\bobject\s*([0-9]+)\b", t, flags=re.IGNORECASE)
        if m:
            return int(m.group(1))
        if t.isdigit():
            return int(t)
        return None

    def _get_edge_data(self, u: int, v: int) -> tuple[dict[str, Any] | None, bool]:
        """
        Return edge and inversion flag for (u, v).
        If (u, v) not found but (v, u) exists, return that edge with inverted=True.
        """
        if (u, v) in self.edges_map:
            return self.edges_map[(u, v)], False
        if (v, u) in self.edges_map:
            return self.edges_map[(v, u)], True
        return None, False

    def _extract_dir_set_from_text(self, text: str) -> set[str]:
        """
        Parse a free-form text into a set of canonical direction primitives.
        Handles composite directions like "back-right", "front-left", "upper-right".
        """
        if not text:
            return set()

        t = text.lower().strip()
        t = t.replace("  ", " ")

        t = t.replace("-", " ")

        dir_set: set[str] = set()

        for k in sorted(self.dir_keywords.keys(), key=len, reverse=True):
            if k in t:
                canonical = self.dir_keywords[k]
                dir_set.add(canonical)
                t = t.replace(k, " ", 1)

        return dir_set

    def _edge_relation_set(self, edge: dict[str, Any], inverted: bool) -> set[str]:
        """
        Extract canonical primitive set from edge["relation"] and apply inversion if needed.
        edge["relation"] is expected to be a list of strings, but we handle robustness.
        """
        raw = edge.get("relation", [])
        rels: set[str] = set()

        if isinstance(raw, list):
            for item in raw:
                rels.update(self._extract_dir_set_from_text(str(item)))
        else:
            rels.update(self._extract_dir_set_from_text(str(raw)))

        if inverted and rels:
            rels = {self.rel_inversion.get(p, p) for p in rels}
        return rels

    def _parse_options(self, full_text: str) -> dict[str, str]:
        """
        Parses 'A: xxx\\nB: yyy' into {'A': 'xxx', 'B': 'yyy'}
        """
        options_map: dict[str, str] = {}
        if "Options:" in full_text:
            parts = full_text.split("Options:")[-1].strip().split("\n")
        elif "options:" in full_text:
            parts = full_text.split("options:")[-1].strip().split("\n")
        else:
            return {}

        for line in parts:
            line = line.strip()
            match = re.match(r"^([A-Z])[:\.]\s+(.*)", line)
            if match:
                label, text = match.groups()
                options_map[label] = text.strip()
        return options_map

    def _pick_label_by_number(self, options: dict[str, str], target_num: int) -> str | None:
        """
        Find which option label corresponds to the target integer.
        """
        for label, text in options.items():
            m = re.search(r"(-?\d+)", text.strip())
            if m and int(m.group(1)) == int(target_num):
                return label
        return None

    # --- Deduction: build pseudo-geometry from relations + distances (no translation) ---

    def _dirset_to_unit_vec_xz(self, dset: set[str]) -> tuple[float, float] | None:
        """
        Map a direction set to a unit vector in (x, z) plane.
        Convention:
          left  -> (-1, 0)
          right -> ( 1, 0)
          in front of -> (0, -1)   # closer to camera
          behind      -> (0,  1)   # farther
        If set is empty or cancels out, return None.
        """
        vx, vz = 0.0, 0.0
        for d in dset:
            if d == "left":
                vx += -1.0
            elif d == "right":
                vx += 1.0
            elif d == "in front of":
                vz += -1.0
            elif d == "behind":
                vz += 1.0
            else:
                # ignore up/down for deduction (you said no vertical placement)
                continue

        norm = math.sqrt(vx * vx + vz * vz)
        if norm < 1e-6:
            return None
        return (vx / norm, vz / norm)

    # --- Transformation helpers ---
    def _get_8_sector_label(self, angle_diff: float) -> str:
        """
        Map relative angle to one of 8 direction labels.
        Angle diff should be in (-pi, pi].
        """
        PI_8 = math.pi / 8
        ad = angle_diff
        abs_d = abs(ad)

        if abs_d <= PI_8:
            return "Front"
        elif abs_d >= 7 * PI_8:
            return "Behind"

        if ad > 0:  # Right side (positive in our calc frame)
            if PI_8 < ad <= 3 * PI_8:
                return "Front-Right"
            elif 3 * PI_8 < ad <= 5 * PI_8:
                return "Right"
            elif 5 * PI_8 < ad < 7 * PI_8:
                return "Back-Right"
        else:  # Left side
            if -3 * PI_8 <= ad < -PI_8:
                return "Front-Left"
            elif -5 * PI_8 <= ad < -3 * PI_8:
                return "Left"
            elif -7 * PI_8 < ad < -5 * PI_8:
                return "Back-Left"

        return "Unknown"

    # --------------------------------------------------------------------------
    # Main entry
    # --------------------------------------------------------------------------

    def solve(self, qa_item: dict[str, Any]) -> str:
        """Derive an answer for a QA item using the predicted scene graph."""
        try:
            user_turn = next(t for t in qa_item["conversations"] if t["from"] == "human")
            full_text = user_turn["value"]
        except StopIteration:
            return "Error: Bad Format"

        meta = qa_item.get("meta", {})
        category = meta.get("category", "unknown")

        extracted = self.matcher.extract(full_text, category)
        if not extracted:
            return self.MISSING_LABEL

        # Normalize OBJ_* to int ids and check existence
        vars_id: dict[str, Any] = dict(extracted)
        for key in ["OBJ_A", "OBJ_B", "OBJ_C"]:
            if key in vars_id:
                oid = self._normalize_obj_to_id(vars_id[key])
                if oid is None or oid not in self.objects_by_id:
                    return self.MISSING_LABEL
                vars_id[key] = oid

        # Parse DIST as float if present
        if "DIST" in vars_id:
            try:
                vars_id["DIST"] = float(vars_id["DIST"])
            except Exception:
                return self.MISSING_LABEL

        # Dispatch
        if category == "direction":
            return self._solve_direction(vars_id, full_text)
        if category == "distance":
            return self._solve_distance(vars_id, meta, full_text)
        if category == "size":
            return self._solve_size(vars_id, meta, full_text)
        if category == "ranking":
            return self._solve_ranking(vars_id, full_text)
        if category == "counting":
            return self._solve_count(vars_id, full_text)
        if category == "deduction":
            return self._solve_deduction(vars_id, full_text)
        if category == "transformation":
            return self._solve_transformation(vars_id, full_text)

        return "Error: Unknown Category"

    # --------------------------------------------------------------------------
    # Existing categories (adapted to id-based SG)
    # --------------------------------------------------------------------------

    def _solve_direction(self, vars: dict[str, Any], full_text: str) -> str:
        """
        Solve direction questions with robust matching for composite directions.
        """
        a_id, b_id = int(vars["OBJ_A"]), int(vars["OBJ_B"])

        edge, edge_inverted = self._get_edge_data(a_id, b_id)
        if not edge:
            return self.MISSING_LABEL

        # Get "A relative to B"
        sg_rels_set = self._edge_relation_set(edge, not edge_inverted)

        if not sg_rels_set:
            return self.MISSING_LABEL

        options = self._parse_options(full_text)
        q_text_lower = full_text.split("Options:")[0].lower()

        # Check if Yes/No question
        opt_values_str = " ".join(str(v).lower() for v in options.values())
        has_yes_no = any(x in opt_values_str for x in ["yes", "no", "true", "false"])

        if has_yes_no:
            q_dir_set = self._extract_dir_set_from_text(q_text_lower)
            if q_dir_set:
                yes_label = next((k for k, v in options.items()
                                  if "yes" in v.lower() or "true" in v.lower()), "A")
                no_label = next((k for k, v in options.items()
                                 if "no" in v.lower() or "false" in v.lower()), "B")

                is_true = q_dir_set.issubset(sg_rels_set)
                return yes_label if is_true else no_label

        # MCQ: Extract direction from each option
        option_dir_sets = {
            label: self._extract_dir_set_from_text(text)
            for label, text in options.items()
        }

        exact_matches = [
            label for label, dset in option_dir_sets.items()
            if dset and dset == sg_rels_set
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return sorted(exact_matches)[0]

        subset_matches = []
        superset_matches = []

        for label, opt_set in option_dir_sets.items():
            if not opt_set:
                continue
            if sg_rels_set.issubset(opt_set):
                subset_matches.append(label)
            elif opt_set.issubset(sg_rels_set):
                superset_matches.append(label)

        if len(subset_matches) == 1:
            return subset_matches[0]
        if len(superset_matches) == 1:
            return superset_matches[0]

        best_label = None
        max_overlap = 0

        for label, opt_set in option_dir_sets.items():
            if not opt_set:
                continue
            overlap = len(sg_rels_set.intersection(opt_set))

            union_size = len(sg_rels_set.union(opt_set))
            if union_size > 0:
                jaccard = overlap / union_size
            else:
                jaccard = 0.0

            if overlap > max_overlap or (overlap == max_overlap and jaccard > 0.5):
                max_overlap = overlap
                best_label = label

        if best_label and max_overlap > 0:
            return best_label

        return "FAILED: Option Mismatch"

    def _solve_distance(self, vars: dict[str, Any], meta: dict[str, Any], full_text: str) -> str:
        """
        Handles all distance-related questions:
        1. Camera-to-Object (NAQ or MCQ)
        2. Object-to-Object Edge Distance (NAQ or MCQ)
        """
        q_lower = full_text.lower()
        a_id = int(vars["OBJ_A"])
        obj_a = self.objects_by_id.get(a_id)

        if not obj_a:
            return self.MISSING_LABEL

        is_na = meta.get("type") in {"NA", "NAQ"}

        # ==========================================
        # CASE 1: Camera Distance
        # ==========================================
        if "camera" in q_lower:
            try:
                dist_a = float(obj_a.get("dist_to_cam", 0.0))
            except (ValueError, TypeError):
                return self.MISSING_LABEL

            if is_na:
                return str(round(dist_a, 2))

            if "OBJ_B" in vars:
                b_id = int(vars["OBJ_B"])
                obj_b = self.objects_by_id.get(b_id)
                if not obj_b:
                    return self.MISSING_LABEL

                try:
                    dist_b = float(obj_b.get("dist_to_cam", 0.0))
                except (ValueError, TypeError):
                    return self.MISSING_LABEL

                closer = ("closer" in q_lower) or ("nearest" in q_lower)
                if closer:
                    winner_id = a_id if (dist_a < dist_b) else b_id
                else:
                    winner_id = a_id if (dist_a > dist_b) else b_id

                options = self._parse_options(full_text)
                target_str = f"object {winner_id}"
                pattern = r"\b" + re.escape(target_str) + r"\b"
                for label, text in options.items():
                    if re.search(pattern, text, re.IGNORECASE):
                        return label
            return self.MISSING_LABEL

        # ==========================================
        # CASE 2: Object-to-Object Edge Distance (Unified)
        # ==========================================
        if "OBJ_B" not in vars:
            return self.MISSING_LABEL

        b_id = int(vars["OBJ_B"])
        edge_ab, _ = self._get_edge_data(a_id, b_id)

        if not edge_ab:
            return self.MISSING_LABEL

        try:
            dist_ab = float(edge_ab.get("distance", 0.0))
        except (ValueError, TypeError):
            return self.MISSING_LABEL

        # NAQ: Return numeric value
        if is_na:
            return str(round(dist_ab, 2))

        # MCQ: Compare three objects
        if "OBJ_C" not in vars:
            return self.MISSING_LABEL

        c_id = int(vars["OBJ_C"])
        edge_ac, _ = self._get_edge_data(a_id, c_id)

        if not edge_ac:
            return self.MISSING_LABEL

        try:
            dist_ac = float(edge_ac.get("distance", 0.0))
        except (ValueError, TypeError):
            return self.MISSING_LABEL

        closer = ("closer" in q_lower) or ("nearest" in q_lower)
        if closer:
            winner_id = b_id if (dist_ab < dist_ac) else c_id
        else:
            winner_id = b_id if (dist_ab > dist_ac) else c_id

        options = self._parse_options(full_text)
        target_str = f"object {winner_id}"
        pattern = r"\b" + re.escape(target_str) + r"\b"
        for label, text in options.items():
            if re.search(pattern, text, re.IGNORECASE):
                return label

        return self.MISSING_LABEL

    def _solve_size(self, vars: dict[str, Any], meta: dict[str, Any], full_text: str) -> str:
        """
        Solve size comparison questions.
        Handles: height, width, length, and volume comparisons.
        """
        a_id = int(vars["OBJ_A"])
        obj_a = self.objects_by_id.get(a_id)

        if not obj_a:
            return self.MISSING_LABEL

        def get_metric(obj: dict[str, Any], text_context: str) -> float:
            """
            Extract the relevant size metric based on question keywords.
            """
            s = obj.get("size", {})
            t = text_context.lower()

            # Volume (product of dimensions)
            if any(keyword in t for keyword in ["volume", "larger", "bigger", "smaller"]):
                w = float(s.get("w", 0.0))
                l = float(s.get("l", 0.0))
                h = float(s.get("h", 0.0))
                return w * l * h

            # Height
            if any(keyword in t for keyword in ["height", "taller", "shorter", "tall"]):
                return float(s.get("h", 0.0))

            # Width
            if any(keyword in t for keyword in ["width", "wider", "narrower", "wide"]):
                return float(s.get("w", 0.0))

            # Length (default for linear dimensions)
            if any(keyword in t for keyword in ["length", "longer", "shorter", "long"]):
                return float(s.get("l", 0.0))

            # Fallback: If no specific keyword, assume volume
            w = float(s.get("w", 0.0))
            l = float(s.get("l", 0.0))
            h = float(s.get("h", 0.0))
            return w * l * h

        try:
            val_a = get_metric(obj_a, full_text)
        except (ValueError, TypeError):
            return self.MISSING_LABEL

        # NAQ questions: return numeric value
        if meta.get("type") in {"NA", "NAQ"}:
            return str(round(val_a, 2))

        # MCQ: Compare two objects
        if "OBJ_B" in vars:
            b_id = int(vars["OBJ_B"])
            obj_b = self.objects_by_id.get(b_id)

            if not obj_b:
                return self.MISSING_LABEL

            try:
                val_b = get_metric(obj_b, full_text)
            except (ValueError, TypeError):
                return self.MISSING_LABEL

            # Determine winner based on question type
            q_lower = full_text.lower()

            if any(kw in q_lower for kw in ["larger", "bigger", "taller", "wider", "longer"]):
                # Comparative: who is larger/bigger/etc.?
                winner_id = a_id if val_a > val_b else b_id
            elif any(kw in q_lower for kw in ["smaller", "shorter", "narrower"]):
                # Comparative: who is smaller/shorter/etc.?
                winner_id = a_id if val_a < val_b else b_id
            else:
                # Default: assume asking for larger
                winner_id = a_id if val_a > val_b else b_id

            # Match winner to options
            options = self._parse_options(full_text)
            target_str = f"object {winner_id}"
            pattern = r"\b" + re.escape(target_str) + r"\b"

            for label, text in options.items():
                if re.search(pattern, text, re.IGNORECASE):
                    return label

            return self.MISSING_LABEL

        return self.MISSING_LABEL

    def _solve_ranking(self, vars: dict[str, Any], full_text: str) -> str:
        a_id = int(vars.get("OBJ_A", -1))
        b_id = int(vars.get("OBJ_B", -1))
        c_id = int(vars.get("OBJ_C", -1))
        if a_id not in self.objects_by_id or b_id not in self.objects_by_id or c_id not in self.objects_by_id:
            return self.MISSING_LABEL

        ids = [a_id, b_id, c_id]
        data: list[tuple[int, float]] = []

        for oid in ids:
            obj = self.objects_by_id[oid]
            if "distance" in full_text.lower():
                val = float(obj.get("dist_to_cam", 0.0))
            elif "height" in full_text.lower():
                val = float(obj.get("size", {}).get("h", 0.0))
            else:
                s = obj.get("size", {})
                val = float(s.get("w", 0.0)) * float(s.get("l", 0.0)) * float(s.get("h", 0.0))
            data.append((oid, val))

        descending = True
        if "distance" in full_text.lower():
            descending = False

        data.sort(key=lambda x: x[1], reverse=descending)
        sorted_ids = [x[0] for x in data]

        options = self._parse_options(full_text)
        for label, text in options.items():
            opt_ids = [int(x) for x in re.findall(r"\bobject\s*([0-9]+)\b", text, flags=re.IGNORECASE)]
            if len(opt_ids) >= 3 and opt_ids[:3] == sorted_ids:
                return label

        return self.MISSING_LABEL

    # --------------------------------------------------------------------------
    # COUNT solver (MCQ)
    # --------------------------------------------------------------------------

    def _solve_count(self, vars: dict[str, Any], full_text: str) -> str:
        options = self._parse_options(full_text)
        if not options:
            return self.MISSING_LABEL

        q_lower = full_text.split("Options:")[0].lower()
        a_id = int(vars["OBJ_A"])

        MAX_COUNT_RANGE = 25.0  # meters

        # Case 1: directional count uses REL_PHRASE
        if "REL_PHRASE" in vars:
            rel_phrase = str(vars["REL_PHRASE"])
            rel_set = self._extract_dir_set_from_text(rel_phrase)
            if not rel_set:
                rel_set = self._extract_dir_set_from_text(q_lower)
            if not rel_set:
                return self.MISSING_LABEL

            target_rel = sorted(list(rel_set))[0]

            cnt = 0
            for b_id in self.objects_by_id.keys():
                if b_id == a_id:
                    continue

                edge, inverted = self._get_edge_data(a_id, b_id)
                if not edge:
                    continue

                # only count objects within 25m of OBJ_A
                try:
                    d_ab = float(edge.get("distance", 0.0))
                except Exception:
                    continue
                if d_ab >= MAX_COUNT_RANGE:
                    continue

                rels = self._edge_relation_set(edge, inverted)
                if target_rel in rels:
                    cnt += 1

            label = self._pick_label_by_number(options, cnt)
            return label if label is not None else self.MISSING_LABEL

        # Case 2: distance threshold count
        dist_threshold = vars.get("DIST", None)
        if dist_threshold is None:
            m = re.search(r"within\s+(\d+(?:\.\d+)?)\s+meters", q_lower)
            if not m:
                m = re.search(r"farther\s+than\s+(\d+(?:\.\d+)?)\s+meters", q_lower)
            if m:
                dist_threshold = float(m.group(1))

        if dist_threshold is None:
            return self.MISSING_LABEL

        is_within = "within" in q_lower
        is_farther = "farther than" in q_lower
        if not (is_within or is_farther):
            return self.MISSING_LABEL

        cnt = 0
        for b_id in self.objects_by_id.keys():
            if b_id == a_id:
                continue

            edge, _ = self._get_edge_data(a_id, b_id)
            if not edge:
                continue

            try:
                d = float(edge.get("distance", 0.0))
            except Exception:
                continue

            # only count objects within 25m of OBJ_A
            if d >= MAX_COUNT_RANGE:
                continue

            if is_within and d <= float(dist_threshold):
                cnt += 1
            elif is_farther and d > float(dist_threshold):
                cnt += 1

        label = self._pick_label_by_number(options, cnt)
        return label if label is not None else self.MISSING_LABEL


    # --------------------------------------------------------------------------
    # DEDUCTION solver (NAQ) without translation
    # --------------------------------------------------------------------------

    def _solve_deduction(self, vars: dict[str, Any], full_text: str) -> str:

        a_id = int(vars["OBJ_A"])
        b_id = int(vars["OBJ_B"])

        place_dir_text = str(vars.get("DIR", "")).strip()
        d_place = vars.get("DIST", None)
        if d_place is None:
            return self.MISSING_LABEL
        d_place = float(d_place)

        edge = None
        rel_ab_set: set[str] | None = None
        d_ab: float | None = None

        if (a_id, b_id) in self.edges_map:
            # a is center, edge encodes B relative to A
            edge = self.edges_map[(a_id, b_id)]
            try:
                d_ab = float(edge.get("distance", 0.0))
            except Exception:
                return self.MISSING_LABEL
            rel_ab_set = self._edge_relation_set(edge, inverted=False)

        elif (b_id, a_id) in self.edges_map:
            # b is center, edge encodes A relative to B
            edge = self.edges_map[(b_id, a_id)]
            try:
                d_ab = float(edge.get("distance", 0.0))
            except Exception:
                return self.MISSING_LABEL
            rel_a_relative_b = self._edge_relation_set(edge, inverted=False)  # A | B
            # We need B relative to A => invert(A|B)
            rel_ab_set = {self.rel_inversion.get(p, p) for p in rel_a_relative_b} if rel_a_relative_b else set()

        else:
            print(f"Missing edge for deduction between {a_id} and {b_id}")
            return self.MISSING_LABEL

        if d_ab is None or rel_ab_set is None:
            return self.MISSING_LABEL

        v_ab = self._dirset_to_unit_vec_xz(rel_ab_set)
        if v_ab is None:
            return self.MISSING_LABEL

        place_dir_set = self._extract_dir_set_from_text(place_dir_text)
        if not place_dir_set:
            place_dir_set = self._extract_dir_set_from_text(full_text.split("Options:")[0])
        if not place_dir_set:
            return self.MISSING_LABEL

        # generator uses one of: left/right/in front of/behind
        v_place = self._dirset_to_unit_vec_xz(set([sorted(list(place_dir_set))[0]]))
        if v_place is None:
            return self.MISSING_LABEL

        # Pseudo coordinates in xz plane: A at origin
        bx, bz = d_ab * v_ab[0], d_ab * v_ab[1]
        nx, nz = d_place * v_place[0], d_place * v_place[1]

        dx = nx - bx
        dz = nz - bz

        d_nb = math.sqrt(dx * dx + dz * dz)
        return str(round(d_nb, 2))

    # =============================================================================
    # Transformation solver
    # =============================================================================
    def _solve_transformation(self, vars: dict[str, Any], full_text: str) -> str:
        """
        Solves: "Standing at A, facing B, where is C?"
        Using the same coordinate system as the generator.
        """
        # 1. Get vectors
        def get_vec_relative_to_center(center_id: int, target_id: int) -> tuple[float, float] | None:
            edge, inverted = self._get_edge_data(center_id, target_id)
            if not edge:
                return None
            rels = self._edge_relation_set(edge, inverted)
            if not rels:
                return None
            return self._dirset_to_unit_vec_xz(rels)

        a_id = int(vars["OBJ_A"])
        b_id = int(vars["OBJ_B"])
        c_id = int(vars["OBJ_C"])

        vec_b = get_vec_relative_to_center(a_id, b_id)
        vec_c = get_vec_relative_to_center(a_id, c_id)

        if vec_b is None or vec_c is None:
            return self.MISSING_LABEL

        # 2. Calculate angles using atan2(X, Z)
        dx_b, dz_b = vec_b
        dx_c, dz_c = vec_c

        theta_b = math.atan2(dx_b, dz_b)
        theta_c = math.atan2(dx_c, dz_c)

        # 3. Calculate relative angle
        diff = theta_c - theta_b

        # Normalize to (-π, π]
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff <= -math.pi:
            diff += 2 * math.pi

        # 4. Get sector label
        calculated_label = self._get_8_sector_label(diff)
        if calculated_label == "Unknown":
            return self.MISSING_LABEL

        # 5. Build expected direction set from calculated label
        # Map sector labels to canonical direction primitives
        sector_to_dirs = {
            "Front": {"in front of"},
            "Behind": {"behind"},
            "Left": {"left"},
            "Right": {"right"},
            "Front-Left": {"in front of", "left"},
            "Front-Right": {"in front of", "right"},
            "Back-Left": {"behind", "left"},
            "Back-Right": {"behind", "right"},
        }

        expected_dir_set = sector_to_dirs.get(calculated_label, set())
        if not expected_dir_set:
            return self.MISSING_LABEL

        # 6. Parse options and extract direction sets
        options = self._parse_options(full_text)
        if not options:
            return self.MISSING_LABEL

        option_dir_sets = {
            label: self._extract_dir_set_from_text(text)
            for label, text in options.items()
        }

        exact_matches = [
            label for label, dset in option_dir_sets.items()
            if dset and dset == expected_dir_set
        ]
        if len(exact_matches) == 1:
            return exact_matches[0]
        if len(exact_matches) > 1:
            return sorted(exact_matches)[0]

        subset_matches = []
        superset_matches = []

        for label, opt_set in option_dir_sets.items():
            if not opt_set:
                continue
            if expected_dir_set.issubset(opt_set):
                subset_matches.append(label)
            elif opt_set.issubset(expected_dir_set):
                superset_matches.append(label)

        if len(subset_matches) == 1:
            return subset_matches[0]
        if len(superset_matches) == 1:
            return superset_matches[0]

        best_label = None
        max_jaccard = 0.0

        for label, opt_set in option_dir_sets.items():
            if not opt_set:
                continue

            intersection = len(expected_dir_set.intersection(opt_set))
            union = len(expected_dir_set.union(opt_set))

            if union > 0:
                jaccard = intersection / union

                # Require at least 50% overlap
                if jaccard > max_jaccard and jaccard >= 0.5:
                    max_jaccard = jaccard
                    best_label = label

        if best_label:
            return best_label

        return "FAILED: Option Mismatch"

# ==============================================================================
# 3. Main Execution
# ==============================================================================

def main():
    """CLI entry point for solving QA items against predicted scene graphs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-list", required=True, help="Path to QA JSONL file")
    parser.add_argument("--templates", required=True, help="Path to templates JSON")
    parser.add_argument("--output", required=True, help="Path to save output JSONL")
    args = parser.parse_args()

    print("Loading data...")
    with open(args.scene_graphs) as f:
        pred_sgs = json.load(f)

    with open(args.templates) as f:
        templates = json.load(f)

    # Use basename as lookup key to handle absolute-vs-relative path
    # differences robustly.  Image filenames are unique hashes.
    pred_lookup = {}
    for sg in pred_sgs:
        if "image" in sg:
            pred_lookup[os.path.basename(sg["image"])] = sg

    matcher = TemplateMatcher(templates)

    results_buffer = []
    missing_count = 0
    total_count = 0

    print("Processing QA items...")
    with open(args.qa_list) as f:
        for line in f:
            if not line.strip():
                continue
            qa_item = json.loads(line)
            total_count += 1

            image_key = os.path.basename(qa_item.get("image", ""))
            pred_sg = pred_lookup.get(image_key)

            if not pred_sg:
                prediction = "FAILED: Scene Graph Missing"
                missing_count += 1
            else:
                solver = SceneGraphSolver(pred_sg, matcher)
                prediction = solver.solve(qa_item)
                if prediction == SceneGraphSolver.MISSING_LABEL:
                    missing_count += 1

            qa_item["predict"] = prediction
            results_buffer.append(qa_item)

    print(f"Writing results to {args.output}...")
    with open(args.output, "w") as f:
        for item in results_buffer:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Total: {total_count}, Failed: {missing_count}")


if __name__ == "__main__":
    main()
