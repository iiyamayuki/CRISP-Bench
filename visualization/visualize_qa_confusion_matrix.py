#!/usr/bin/env python3

import argparse
import json
import os

import numpy as np

# Import reusable classes from analyze_qa_sg_results.py
from analyze_qa_sg_results import DataExtractor, DataLoader, Visualizer


class QAConfusionAnalyzer:
    """Analyzer for QA confusion matrices comparing different QA settings"""

    def __init__(self, results_dir: str, derived_qa_dir: str | None = None,
                 include_na: bool = False):
        self.results_dir = results_dir
        self.aggregated_file = os.path.join(results_dir, 'aggregated.json')
        self.derived_qa_dir = derived_qa_dir
        self.include_na = include_na

        # Reuse existing components
        self.loader = DataLoader()
        self.extractor = DataExtractor(self.loader)
        self.visualizer = Visualizer()

        self.aggregated_data = None
        self.results = {}
        self.model_names = {}
        self.model_name_mapping = {}

    def load_data(self):
        """Load aggregated data"""
        if not os.path.exists(self.aggregated_file):
            raise FileNotFoundError(f"Aggregated file not found: {self.aggregated_file}")

        self.aggregated_data = self.loader.load_json(self.aggregated_file)
        print(f"Loaded {len(self.aggregated_data)} models from aggregated.json")

    def extract_qa_results(self, model_key: str, model_data: dict,
                          task_name: str, mode: str) -> dict | None:
        """Extract QA results for a specific task and mode"""
        if 'tasks' not in model_data or task_name not in model_data['tasks']:
            return None

        task_data = model_data['tasks'][task_name]
        if mode not in task_data:
            return None

        mode_data = task_data[mode]

        # Get source file
        source_file = None
        if 'source_file' in mode_data:
            source_file = mode_data['source_file']
        elif isinstance(mode_data, dict):
            for key in ['gt', 'pred']:
                if key in mode_data and isinstance(mode_data[key], dict):
                    source_file = mode_data[key].get('source_file')
                    if source_file:
                        break

        if not source_file:
            return None

        # Handle consistency_score differently (no samples file)
        if task_name == 'consistency_score':
            return self.load_consistency_results(source_file)

        # For other tasks, convert results.json to samples.jsonl
        samples_file = self.loader.get_samples_file(source_file, task_name)
        if not samples_file or not os.path.exists(samples_file):
            return None

        # Load and analyze predictions
        data = self.loader.load_jsonl(samples_file)
        return self.analyze_predictions(data)

    def load_consistency_results(self, results_file: str) -> dict | None:
        """Load consistency_score results from JSON file"""
        if not os.path.exists(results_file):
            return None

        with open(results_file) as f:
            data = json.load(f)

        # Extract detailed_accuracy data
        if 'detailed_accuracy' not in data:
            return None

        detailed_accuracy = data['detailed_accuracy']

        mcq_results = {}
        na_results = {}

        for item in detailed_accuracy:
            doc_id = item['id']
            q_type = item['type']
            score = item['score']
            category = item['category']

            if q_type == 'MCQ':
                # MCQ: score is 1.0 or 0.0, interpret as boolean
                mcq_results[doc_id] = {
                    'correct': bool(score > 0.5),
                    'type': q_type,
                    'category': category,
                }
            elif q_type in {'NA', 'NAQ'}:
                # NAQ: score is MRA value 0-1
                na_results[doc_id] = {
                    'score': score,
                    'type': q_type,
                    'category': category,
                }

        mcq_accuracy = np.mean([r['correct'] for r in mcq_results.values()]) if mcq_results else 0
        na_mean_mra = np.mean([r['score'] for r in na_results.values()]) if na_results else 0

        return {
            'mcq': mcq_results,
            'na': na_results,
            'mcq_accuracy': mcq_accuracy,
            'na_mean_mra': na_mean_mra,
            'total_mcq': len(mcq_results),
            'total_na': len(na_results)
        }

    def _build_model_name_mapping(self):
        """Build mapping from model_key to derived QA filename"""
        # Manual mappings for known models
        manual_mappings = {
            'GPT5/gpt-5-mini': 'gpt5_mini',
            'GPT5/gpt-5.2': 'gpt5_2',
            'cambrians_eval_result/nyu-visionx__Cambrian-S-7B': 'cambrians',
            'gemini_eval_result/gemini-2.5-flash': 'gemini2_5_flash',
            'gemini_eval_result/gemini-2.5-pro': 'gemini2_5_pro',
            'gemini_eval_result/gemini-3-flash-preview': 'gemini3_flash',
            'internvl3_5/OpenGVLab__InternVL3_5-38B': 'internvl3_5_38B',
            'internvl3_5/OpenGVLab__InternVL3_5-8B': 'internvl3_5_8B',
            'llava_onevision1_5/lmms-lab__LLaVA-OneVision-1.5-8B-Instruct': 'llava_onevision1_5',
            'qwen2_5vl_vllm/Qwen__Qwen2.5-VL-7B-Instruct': 'qwen2_5vl_7B',
            'qwen3vl_vllm/Qwen__Qwen3-VL-32B-Instruct': 'qwen3vl_32B',
            'qwen3vl_vllm/Qwen__Qwen3-VL-8B-Instruct': 'qwen3vl_8B',
            'vgllm_eval_result/VG-LLM': 'vgllm',
        }

        for model_key in self.aggregated_data.keys():
            if model_key in manual_mappings:
                self.model_name_mapping[model_key] = manual_mappings[model_key]
            else:
                # Fallback: try to extract from key
                if '/' in model_key:
                    name_part = model_key.split('/', 1)[1]
                else:
                    name_part = model_key

                filename_base = name_part.lower().replace('-', '_').replace('.', '_')
                self.model_name_mapping[model_key] = filename_base

    def load_derived_qa_from_file(self, model_key: str) -> dict | None:
        """Load derived QA results from generated_sg directory"""
        if not self.derived_qa_dir:
            return None

        # Get the filename base for this model
        if model_key not in self.model_name_mapping:
            return None

        filename_base = self.model_name_mapping[model_key]
        qa_file = os.path.join(self.derived_qa_dir, f'{filename_base}_qa.jsonl')

        if not os.path.exists(qa_file):
            return None

        return self.analyze_derived_qa_predictions(qa_file)

    def analyze_derived_qa_predictions(self, qa_file: str) -> dict:
        """Analyze derived QA predictions from jsonl file"""
        data = self.loader.load_jsonl(qa_file)

        mcq_results = {}
        na_results = {}

        for item in data:
            doc_id = item['id']
            meta = item['meta']
            q_type = meta['type']
            category = meta['category']

            # Get ground truth from conversations
            gt_answer = item['conversations'][1]['value']  # gpt response
            pred_answer = item.get('predict', '')
            question = item['conversations'][0]['value']  # question

            if q_type == 'MCQ':
                # Compare answers (case-insensitive)
                correct = (str(pred_answer).strip().upper() == str(gt_answer).strip().upper())
                mcq_results[doc_id] = {
                    'correct': correct,
                    'type': q_type,
                    'category': category,
                    'question': question,
                    'predicted_answer': pred_answer,
                    'ground_truth': gt_answer,
                    'full_item': item
                }
            elif q_type in {'NA', 'NAQ'}:
                # For NAQ questions, we'd need to calculate MRA
                # For now, just skip or use a placeholder
                # Since we only need MCQ for confusion matrix, this is fine
                try:
                    gt_val = float(gt_answer)
                    pred_val = float(pred_answer)
                    # Simple normalized error as placeholder
                    error = abs(gt_val - pred_val) / max(abs(gt_val), 0.1)
                    score = max(0, 1 - error)  # Simple score
                except (ValueError, TypeError):
                    score = 0.0

                na_results[doc_id] = {
                    'score': score,
                    'type': q_type,
                    'category': category,
                    'question': question,
                    'predicted_answer': pred_answer,
                    'ground_truth': gt_answer,
                    'full_item': item
                }

        mcq_accuracy = np.mean([r['correct'] for r in mcq_results.values()]) if mcq_results else 0
        na_mean_score = np.mean([r['score'] for r in na_results.values()]) if na_results else 0

        return {
            'mcq': mcq_results,
            'na': na_results,
            'mcq_accuracy': mcq_accuracy,
            'na_mean_mra': na_mean_score,
            'total_mcq': len(mcq_results),
            'total_na': len(na_results)
        }

    def analyze_predictions(self, data: list[dict]) -> dict:
        """Analyze predictions and separate MCQ and NAQ questions
        
        Uses ID extraction logic from calculate_consistency.py:
        PRIORITY: vsibench_score['id'] -> root 'id' -> root 'question_id' -> doc_id
        """
        mcq_results = {}
        na_results = {}

        for item in data:
            vsibench_score = item.get('vsibench_score', {})
            meta = vsibench_score.get('meta', {})

            # Extract ID using same logic as load_direct_qa_map
            if "vsibench_score" in item and "id" in item["vsibench_score"]:
                doc_id = item["vsibench_score"]["id"]
            else:
                # Fallback keys
                doc_id = item.get("id") or item.get("question_id") or str(item.get("doc_id"))

            # Extract question, ground truth, and prediction from vsibench_score structure
            conversations = vsibench_score.get('conversations', [])
            question = conversations[0]['value'] if len(conversations) > 0 else ''
            ground_truth = conversations[1]['value'] if len(conversations) > 1 else item.get('target', '')
            predicted_answer = vsibench_score.get('prediction', '')
            if not predicted_answer and 'filtered_resps' in item:
                predicted_answer = item['filtered_resps'][0] if item['filtered_resps'] else ''

            if 'accuracy' in vsibench_score:
                # MCQ question - use boolean accuracy
                mcq_results[doc_id] = {
                    'correct': vsibench_score['accuracy'],
                    'type': meta.get('type'),
                    'category': meta.get('category'),
                    'question': question,
                    'predicted_answer': predicted_answer,
                    'ground_truth': ground_truth,
                    'full_item': item  # Store full item for reference
                }
            elif 'MRA:.5:.95:.05' in vsibench_score:
                # NAQ question - use MRA score
                na_results[doc_id] = {
                    'score': vsibench_score['MRA:.5:.95:.05'],
                    'type': meta.get('type'),
                    'category': meta.get('category'),
                    'question': question,
                    'predicted_answer': predicted_answer,
                    'ground_truth': ground_truth,
                    'full_item': item
                }

        # Calculate overall metrics
        mcq_accuracy = np.mean([r['correct'] for r in mcq_results.values()]) if mcq_results else 0
        na_mean_mra = np.mean([r['score'] for r in na_results.values()]) if na_results else 0

        return {
            'mcq': mcq_results,
            'na': na_results,
            'mcq_accuracy': mcq_accuracy,
            'na_mean_mra': na_mean_mra,
            'total_mcq': len(mcq_results),
            'total_na': len(na_results)
        }

    def process_models(self):
        """Process all models and extract QA results"""
        print("\nProcessing models...")

        # Build model name mapping
        self._build_model_name_mapping()

        for model_key, model_data in self.aggregated_data.items():
            self.model_names[model_key] = model_data.get('model', model_key)

            # Extract results for different modes
            # 1. Base QA (crisp_qa, multimodal)
            base_qa = self.extract_qa_results(model_key, model_data,
                                             'crisp_qa', 'multimodal')

            # 2. Text-only QA (crisp_qa, text_only)
            text_only_qa = self.extract_qa_results(model_key, model_data,
                                                   'crisp_qa', 'text_only')

            # 3. Derived QA (from generated_sg directory)
            derived_qa = self.load_derived_qa_from_file(model_key)

            # Store results
            if base_qa or text_only_qa or derived_qa:
                self.results[model_key] = {
                    'base_qa': base_qa,
                    'text_only_qa': text_only_qa,
                    'derived_qa': derived_qa
                }

            # Print debug info
            if base_qa or text_only_qa or derived_qa:
                formatted_name = self.visualizer.format_model_name(
                    self.model_names.get(model_key, model_key)
                )
                print(f"\n  {formatted_name}:")
                if base_qa:
                    print(f"    - Base QA: {base_qa['total_mcq']} MCQ, {base_qa['total_na']} NAQ")
                if text_only_qa:
                    print(f"    - Text-only QA: {text_only_qa['total_mcq']} MCQ, {text_only_qa['total_na']} NAQ")
                if derived_qa:
                    print(f"    - Derived QA: {derived_qa['total_mcq']} MCQ, {derived_qa['total_na']} NAQ")

        print(f"\nTotal models with valid data: {len(self.results)}")

    def compare_mcq_results(self, base_results: dict, comp_results: dict) -> dict | None:
        """Compare MCQ results between two settings (legacy wrapper)"""
        return self.compare_results(base_results, comp_results, include_na=False)

    def compare_results(self, base_results: dict, comp_results: dict,
                        include_na: bool = False) -> dict | None:
        """Compare results between two settings, optionally including NAQ questions.

        For NAQ questions correctness is strict: MRA == 1.0 only.
        """
        if not base_results or not comp_results:
            return None

        comparison: dict = {
            'doc_ids': [],
            'base': [],
            'comp': [],
            'base_data': {},
            'comp_data': {}
        }

        # ── MCQ questions ──────────────────────────────────────────────────────
        base_mcq = base_results.get('mcq', {})
        comp_mcq = comp_results.get('mcq', {})
        common_mcq_ids = sorted(set(base_mcq.keys()).intersection(comp_mcq.keys()))

        for doc_id in common_mcq_ids:
            comparison['doc_ids'].append(doc_id)
            comparison['base'].append(base_mcq[doc_id]['correct'])
            comparison['comp'].append(comp_mcq[doc_id]['correct'])
            comparison['base_data'][doc_id] = {**base_mcq[doc_id], '_qa_type': 'mcq'}
            comparison['comp_data'][doc_id] = {**comp_mcq[doc_id], '_qa_type': 'mcq'}

        # ── NAQ questions (optional) ───────────────────────────────────────────
        if include_na:
            base_na = base_results.get('na', {})
            comp_na = comp_results.get('na', {})
            common_na_ids = sorted(set(base_na.keys()).intersection(comp_na.keys()))

            for doc_id in common_na_ids:
                # Strict criterion: only MRA score == 1.0 counts as correct
                base_correct = (base_na[doc_id]['score'] == 1.0)
                comp_correct = (comp_na[doc_id]['score'] == 1.0)
                comparison['doc_ids'].append(doc_id)
                comparison['base'].append(base_correct)
                comparison['comp'].append(comp_correct)
                comparison['base_data'][doc_id] = {
                    **base_na[doc_id],
                    'correct': base_correct,
                    '_qa_type': 'na'
                }
                comparison['comp_data'][doc_id] = {
                    **comp_na[doc_id],
                    'correct': comp_correct,
                    '_qa_type': 'na'
                }

        if not comparison['doc_ids']:
            return None

        return comparison

    def generate_confusion_matrix(self, model_key: str, comparison_type: str, output_dir: str,
                                   include_na: bool = False):
        """Generate confusion matrix for a specific comparison"""
        model_results = self.results.get(model_key)
        if not model_results:
            return False

        base_qa = model_results.get('base_qa')
        if not base_qa:
            return False

        # Get comparison data based on type
        if comparison_type == 'text_only':
            comp_qa = model_results.get('text_only_qa')
            comp_label = 'Text-only QA'
            colormap = 'Purples'
            output_suffix = 'text_only'
        elif comparison_type == 'derived':
            comp_qa = model_results.get('derived_qa')
            comp_label = 'Derived QA'
            colormap = 'Greens'
            output_suffix = 'derived'
        else:
            return False

        if not comp_qa:
            return False

        # Compare results (MCQ + optionally NAQ)
        comparison = self.compare_results(base_qa, comp_qa, include_na=include_na)
        if not comparison or not comparison['doc_ids']:
            return False

        # Generate visualization
        model_name = self.visualizer.format_model_name(
            self.model_names.get(model_key, model_key)
        )
        safe_model_name = model_key.replace('/', '_')

        # Create model-specific subdirectory
        model_output_dir = os.path.join(output_dir, safe_model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        na_suffix = '_with_na' if include_na else ''
        output_path = os.path.join(model_output_dir,
                                   f'{output_suffix}_confusion_matrix{na_suffix}.png')

        # Compute consistent hallucination rate.
        # MCQ: both wrong AND predicted answers are identical strings.
        # NAQ: both wrong AND relative error of comp_pred vs base_pred ≤ 5%.
        total = len(comparison['doc_ids'])
        consistent_hall_count = 0
        for doc_id in comparison['doc_ids']:
            base_item = comparison['base_data'][doc_id]
            comp_item = comparison['comp_data'][doc_id]
            if not base_item['correct'] and not comp_item['correct']:
                qa_type = base_item.get('_qa_type', 'mcq')
                if qa_type == 'mcq':
                    base_pred = str(base_item.get('predicted_answer', '')).strip().upper()
                    comp_pred = str(comp_item.get('predicted_answer', '')).strip().upper()
                    if base_pred == comp_pred and base_pred != '':
                        consistent_hall_count += 1
                elif qa_type == 'na':
                    try:
                        base_val = float(base_item.get('predicted_answer', ''))
                        comp_val = float(comp_item.get('predicted_answer', ''))
                        if abs(base_val) > 1e-9:
                            rel_err = abs(base_val - comp_val) / abs(base_val)
                            if rel_err <= 0.05:
                                consistent_hall_count += 1
                    except (ValueError, TypeError):
                        pass
        consistent_hallucination = consistent_hall_count / total if total > 0 else None

        self.visualizer.plot_confusion_matrix(
            comparison['base'], comparison['comp'],
            model_name, comp_label, output_path, colormap,
            consistent_hallucination=consistent_hallucination
        )

        n_mcq = sum(1 for d in comparison['doc_ids']
                    if comparison['base_data'][d].get('_qa_type', 'mcq') == 'mcq')
        n_na  = len(comparison['doc_ids']) - n_mcq
        type_info = f'{n_mcq} MCQ' + (f', {n_na} NAQ' if n_na else '')
        print(f"  Created: {output_suffix}{na_suffix} confusion matrix ({type_info})")
        print(f"    Saved to: {os.path.relpath(output_path, output_dir.split(os.sep)[0] if os.sep in output_dir else '.')}")
        return True, comparison

    def extract_misaligned_samples(self, comparison: dict, base_label: str, comp_label: str) -> dict:
        """Extract misaligned samples (FP and FN) from comparison
        
        Args:
            comparison: Comparison dict with base/comp predictions and data
            base_label: Label for base QA (e.g., 'Base QA')
            comp_label: Label for comparison QA (e.g., 'Text-only QA')
            
        Returns:
            Dict with 'false_positive' and 'false_negative' lists
        """
        misaligned = {
            'false_positive': [],  # Base wrong, Comp correct
            'false_negative': []   # Base correct, Comp wrong
        }

        base_data = comparison['base_data']
        comp_data = comparison['comp_data']

        for doc_id in comparison['doc_ids']:
            base_correct = base_data[doc_id]['correct']
            comp_correct = comp_data[doc_id]['correct']

            # False Positive: Base wrong (0), Comp correct (1)
            if not base_correct and comp_correct:
                misaligned['false_positive'].append({
                    'doc_id': doc_id,
                    'misalignment_type': 'False Positive',
                    'description': f'{base_label} wrong, {comp_label} correct',
                    'question': base_data[doc_id].get('question', ''),
                    'ground_truth': base_data[doc_id].get('ground_truth', ''),
                    'category': base_data[doc_id].get('category', ''),
                    'base_qa': {
                        'correct': False,
                        'predicted_answer': base_data[doc_id].get('predicted_answer', '')
                    },
                    'comparison_qa': {
                        'correct': True,
                        'predicted_answer': comp_data[doc_id].get('predicted_answer', '')
                    }
                })

            # False Negative: Base correct (1), Comp wrong (0)
            elif base_correct and not comp_correct:
                misaligned['false_negative'].append({
                    'doc_id': doc_id,
                    'misalignment_type': 'False Negative',
                    'description': f'{base_label} correct, {comp_label} wrong',
                    'question': base_data[doc_id].get('question', ''),
                    'ground_truth': base_data[doc_id].get('ground_truth', ''),
                    'category': base_data[doc_id].get('category', ''),
                    'base_qa': {
                        'correct': True,
                        'predicted_answer': base_data[doc_id].get('predicted_answer', '')
                    },
                    'comparison_qa': {
                        'correct': False,
                        'predicted_answer': comp_data[doc_id].get('predicted_answer', '')
                    }
                })

        return misaligned

    def extract_consistent_hallucination_samples(self, comparison: dict,
                                                   base_label: str,
                                                   comp_label: str) -> list[dict]:
        """Extract consistent hallucination samples from comparison.

        A consistent hallucination is a question where:
        - Both base and comp are wrong, AND
        - MCQ: their predicted answers are the same string.
        - NAQ: relative error between base_pred and comp_pred ≤ 5%.

        Returns:
            List of dicts, one per consistent hallucination sample.
        """
        samples = []

        base_data = comparison['base_data']
        comp_data = comparison['comp_data']

        for doc_id in comparison['doc_ids']:
            base_item = base_data[doc_id]
            comp_item = comp_data[doc_id]

            if base_item['correct'] or comp_item['correct']:
                continue  # At least one correct – not a consistent hallucination

            qa_type = base_item.get('_qa_type', 'mcq')

            if qa_type == 'mcq':
                base_pred = str(base_item.get('predicted_answer', '')).strip().upper()
                comp_pred = str(comp_item.get('predicted_answer', '')).strip().upper()
                if base_pred != comp_pred or base_pred == '':
                    continue
                samples.append({
                    'doc_id': doc_id,
                    'qa_type': 'mcq',
                    'hallucination_type': 'Consistent Hallucination',
                    'description': f'Both {base_label} and {comp_label} wrong with identical answer',
                    'question': base_item.get('question', ''),
                    'ground_truth': base_item.get('ground_truth', ''),
                    'category': base_item.get('category', ''),
                    'base_qa': {
                        'correct': False,
                        'predicted_answer': base_item.get('predicted_answer', ''),
                        'label': base_label,
                    },
                    'comparison_qa': {
                        'correct': False,
                        'predicted_answer': comp_item.get('predicted_answer', ''),
                        'label': comp_label,
                    },
                    'consistent': True,
                })

            elif qa_type == 'na':
                try:
                    base_val = float(base_item.get('predicted_answer', ''))
                    comp_val = float(comp_item.get('predicted_answer', ''))
                    if abs(base_val) <= 1e-9:
                        continue
                    rel_err = abs(base_val - comp_val) / abs(base_val)
                    if rel_err > 0.05:
                        continue
                    samples.append({
                        'doc_id': doc_id,
                        'qa_type': 'na',
                        'hallucination_type': 'Consistent Hallucination',
                        'description': (
                            f'Both {base_label} and {comp_label} wrong with similar '
                            f'answers (rel_err={rel_err:.4f})'
                        ),
                        'question': base_item.get('question', ''),
                        'ground_truth': base_item.get('ground_truth', ''),
                        'category': base_item.get('category', ''),
                        'base_qa': {
                            'correct': False,
                            'predicted_answer': base_item.get('predicted_answer', ''),
                            'score': base_item.get('score'),
                            'label': base_label,
                        },
                        'comparison_qa': {
                            'correct': False,
                            'predicted_answer': comp_item.get('predicted_answer', ''),
                            'score': comp_item.get('score'),
                            'label': comp_label,
                        },
                        'relative_error': rel_err,
                        'consistent': True,
                    })
                except (ValueError, TypeError):
                    pass

        return samples

    def save_consistent_hallucination_samples(self, model_key: str, comparison_type: str,
                                              comparison: dict, output_dir: str):
        """Save consistent hallucination samples to JSON file."""
        if comparison_type == 'text_only':
            base_label = 'Base QA (Multimodal)'
            comp_label = 'Text-only QA'
        elif comparison_type == 'derived':
            base_label = 'Base QA (Multimodal)'
            comp_label = 'Derived QA (from Scene Graph)'
        else:
            return

        samples = self.extract_consistent_hallucination_samples(
            comparison, base_label, comp_label
        )

        # Categorise by qa_type for statistics
        mcq_samples = [s for s in samples if s['qa_type'] == 'mcq']
        na_samples  = [s for s in samples if s['qa_type'] == 'na']

        total_questions = len(comparison['doc_ids'])
        total_wrong_both = sum(
            1 for d in comparison['doc_ids']
            if not comparison['base_data'][d]['correct']
            and not comparison['comp_data'][d]['correct']
        )

        output_data = {
            'model': self.model_names.get(model_key, model_key),
            'model_key': model_key,
            'comparison_type': comparison_type,
            'base_label': base_label,
            'comparison_label': comp_label,
            'statistics': {
                'total_questions': total_questions,
                'both_wrong_count': total_wrong_both,
                'consistent_hallucination_count': len(samples),
                'consistent_hallucination_mcq': len(mcq_samples),
                'consistent_hallucination_na': len(na_samples),
                'consistent_hallucination_rate': (
                    len(samples) / total_questions if total_questions > 0 else None
                ),
                'consistent_hallucination_rate_among_wrong': (
                    len(samples) / total_wrong_both if total_wrong_both > 0 else None
                ),
            },
            'consistent_hallucination_samples': samples,
        }

        safe_model_name = model_key.replace('/', '_')
        model_output_dir = os.path.join(output_dir, safe_model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        output_path = os.path.join(
            model_output_dir,
            f'{comparison_type}_consistent_hallucination_samples.json'
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"    \u2713 Saved consistent hallucination samples: "
              f"{len(samples)} total "
              f"({len(mcq_samples)} MCQ, {len(na_samples)} NAQ)")
        print(f"      File: {os.path.relpath(output_path, output_dir.split(os.sep)[0] if os.sep in output_dir else '.')}")

    def save_misaligned_samples(self, model_key: str, comparison_type: str,
                               comparison: dict, output_dir: str):
        """Save misaligned samples to JSON file"""
        if comparison_type == 'text_only':
            base_label = 'Base QA (Multimodal)'
            comp_label = 'Text-only QA'
        elif comparison_type == 'derived':
            base_label = 'Base QA (Multimodal)'
            comp_label = 'Derived QA (from Scene Graph)'
        else:
            return

        misaligned = self.extract_misaligned_samples(comparison, base_label, comp_label)

        # Prepare output data
        output_data = {
            'model': self.model_names.get(model_key, model_key),
            'model_key': model_key,
            'comparison_type': comparison_type,
            'base_label': base_label,
            'comparison_label': comp_label,
            'statistics': {
                'total_questions': len(comparison['doc_ids']),
                'false_positive_count': len(misaligned['false_positive']),
                'false_negative_count': len(misaligned['false_negative']),
                'agreement_count': len(comparison['doc_ids']) - len(misaligned['false_positive']) - len(misaligned['false_negative'])
            },
            'misaligned_samples': misaligned
        }

        # Save to JSON file in model-specific subdirectory
        safe_model_name = model_key.replace('/', '_')
        model_output_dir = os.path.join(output_dir, safe_model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        output_path = os.path.join(model_output_dir,
                                   f'{comparison_type}_misaligned_samples.json')

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"    \u2713 Saved misaligned samples: {len(misaligned['false_positive'])} FP, "
              f"{len(misaligned['false_negative'])} FN")
        print(f"      File: {os.path.relpath(output_path, output_dir.split(os.sep)[0] if os.sep in output_dir else '.')}")

    def generate_all_visualizations(self, output_dir: str, target_model: str | None = None):
        """Generate all confusion matrices
        
        Args:
            output_dir: Output directory for visualizations
            target_model: If specified, also save misaligned samples for this model
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*80)
        print("Generating Confusion Matrices")
        print("="*80)

        if target_model:
            print(f"\n{'='*80}")
            print(f"Target model for misaligned samples: {target_model}")
            print("\nAvailable models:")
            for mk in self.results.keys():
                mn = self.model_names.get(mk, '')
                fmt_name = self.visualizer.format_model_name(mn) if mn else ''
                mapping_name = self.model_name_mapping.get(mk, '')
                print(f"  - key: '{mk}'")
                print(f"    name: '{mn}', formatted: '{fmt_name}', mapping: '{mapping_name}'")
            print(f"{'='*80}\n")

        stats = {
            'text_only': {'success': 0, 'failed': 0},
            'derived': {'success': 0, 'failed': 0}
        }

        for model_key in self.results.keys():
            model_name = self.visualizer.format_model_name(
                self.model_names.get(model_key, model_key)
            )
            print(f"\nProcessing: {model_name}")

            # Check if this is the target model - support multiple matching formats
            is_target = False
            if target_model:
                # Normalize both strings for comparison
                target_norm = target_model.lower().replace('-', '_').replace('/', '_')

                # Try multiple matching strategies
                key_norm = model_key.lower().replace('-', '_').replace('/', '_')
                name_norm = self.model_names.get(model_key, '').lower().replace('-', '_').replace('/', '_')
                fmt_name_norm = model_name.lower().replace('-', '_').replace('/', '_')
                mapping_norm = self.model_name_mapping.get(model_key, '').lower().replace('-', '_')

                is_target = (
                    target_model == model_key or  # Exact match
                    target_model == self.model_names.get(model_key, '') or  # Name match
                    target_model == model_name or  # Formatted name match
                    target_model == self.model_name_mapping.get(model_key, '') or  # Mapping match
                    target_norm == key_norm or  # Normalized key
                    target_norm == name_norm or  # Normalized name
                    target_norm == fmt_name_norm or  # Normalized formatted
                    target_norm == mapping_norm or  # Normalized mapping
                    target_norm in key_norm or  # Substring in key
                    target_norm in mapping_norm  # Substring in mapping
                )

                if is_target:
                    print("  ✓ Matched target model! Will extract misaligned samples.")

            # Generate text-only comparison
            result = self.generate_confusion_matrix(model_key, 'text_only', output_dir,
                                                    include_na=self.include_na)
            if result and result[0]:  # Check if tuple and successful
                stats['text_only']['success'] += 1
                if is_target and len(result) > 1:
                    self.save_misaligned_samples(model_key, 'text_only', result[1], output_dir)
                    self.save_consistent_hallucination_samples(
                        model_key, 'text_only', result[1], output_dir)
            else:
                stats['text_only']['failed'] += 1

            # Generate derived comparison
            result = self.generate_confusion_matrix(model_key, 'derived', output_dir,
                                                    include_na=self.include_na)
            if result and result[0]:  # Check if tuple and successful
                stats['derived']['success'] += 1
                if is_target and len(result) > 1:
                    self.save_misaligned_samples(model_key, 'derived', result[1], output_dir)
                    self.save_consistent_hallucination_samples(
                        model_key, 'derived', result[1], output_dir)
            else:
                stats['derived']['failed'] += 1

        # Print summary
        print("\n" + "="*80)
        print("Summary")
        print("="*80)
        print(f"Text-only QA comparisons: {stats['text_only']['success']} success, "
              f"{stats['text_only']['failed']} failed")
        print(f"Derived QA comparisons: {stats['derived']['success']} success, "
              f"{stats['derived']['failed']} failed")
        print(f"\nAll visualizations saved to: {output_dir}")
        print("  - Each model has its own subdirectory")
        print(f"  - Structure: {output_dir}/<model_name>/")

        if target_model:
            # Count how many models were matched
            matched_count = sum(1 for mk in self.results.keys()
                              if self._is_target_model(mk, target_model))
            if matched_count > 0:
                print(f"\n\u2713 Extracted misaligned samples for {matched_count} model(s)")
            else:
                print(f"\n\u26a0 Warning: No models matched '{target_model}'")
                print(f"  Available model keys: {list(self.results.keys())[:3]}...")

    def _is_target_model(self, model_key: str, target_model: str) -> bool:
        """Check if model_key matches target_model"""
        model_name = self.model_names.get(model_key, '')
        fmt_name = self.visualizer.format_model_name(model_name) if model_name else ''
        mapping_name = self.model_name_mapping.get(model_key, '')

        target_norm = target_model.lower().replace('-', '_').replace('/', '_')
        key_norm = model_key.lower().replace('-', '_').replace('/', '_')
        name_norm = model_name.lower().replace('-', '_').replace('/', '_')
        fmt_name_norm = fmt_name.lower().replace('-', '_').replace('/', '_')
        mapping_norm = mapping_name.lower().replace('-', '_')

        return (
            target_model == model_key or
            target_model == model_name or
            target_model == fmt_name or
            target_model == mapping_name or
            target_norm == key_norm or
            target_norm == name_norm or
            target_norm == fmt_name_norm or
            target_norm == mapping_norm or
            target_norm in key_norm or
            target_norm in mapping_norm
        )


def main():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrices for QA results comparison'
    )
    parser.add_argument('--input_dir', required=True,
                       help='Path to results directory containing aggregated.json')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for confusion matrices (default: input_dir/qa_confusion_matrices)')
    parser.add_argument('--derived_qa_dir', default=None,
                       help='Path to directory containing derived QA files (*_qa.jsonl)')
    parser.add_argument('--model', default=None,
                       help='Model name to extract misaligned samples (optional). '
                            'Can be model key, model name, or formatted name.')
    parser.add_argument('--include_na', action='store_true', default=False,
                       help='Include NAQ (numerical-answer) questions in the confusion matrix. '
                            'Correctness criterion: MRA == 1.0. '
                            'Consistent hallucination criterion: relative error of '
                            'comp_pred vs base_pred ≤ 5%%.')

    args = parser.parse_args()

    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(args.input_dir, 'qa_confusion_matrices')

    # Create analyzer and run analysis
    analyzer = QAConfusionAnalyzer(args.input_dir, args.derived_qa_dir,
                                   include_na=args.include_na)
    analyzer.load_data()
    analyzer.process_models()
    analyzer.generate_all_visualizations(output_dir, target_model=args.model)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

    return analyzer


if __name__ == '__main__':
    analyzer = main()
