import argparse
import json
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.metrics import confusion_matrix


@dataclass
class ModelResult:
    """Store analysis results for a single model"""
    qa_multimodal: dict = None
    qa_sg_gt: dict = None
    qa_sg_pred: dict = None


class DataLoader:
    """Handle all data loading operations"""

    @staticmethod
    def load_json(filepath: str) -> dict:
        """Load JSON file"""
        with open(filepath, encoding='utf-8') as f:
            return json.load(f)

    @staticmethod
    def load_jsonl(filepath: str) -> list[dict]:
        """Load JSONL file"""
        if not filepath or not os.path.exists(filepath):
            return []

        data = []
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return data

    @staticmethod
    def get_samples_file(results_file: str, task_name: str = 'crisp_qa_sg') -> str | None:
        """Convert results.json path to samples.jsonl path"""
        if not results_file:
            return None

        samples_file = results_file.replace('_results.json', f'_samples_{task_name}.jsonl')
        if os.path.exists(samples_file):
            return samples_file

        # Try alternative naming
        samples_file = results_file.replace('_results.json', '_samples_crisp_qa.jsonl')
        return samples_file if os.path.exists(samples_file) else None


class DataExtractor:
    """Extract and process data from aggregated files"""

    def __init__(self, data_loader: DataLoader):
        self.loader = data_loader

    def extract_source_files(self, aggregated_data: dict, task_name: str,
                            mode: str, common_models: set | None = None) -> dict:
        """Extract source files from aggregated data"""
        source_files = {}

        for model_key, model_data in aggregated_data.items():
            if common_models and model_key not in common_models:
                continue

            if 'tasks' not in model_data or task_name not in model_data['tasks']:
                continue

            task_data = model_data['tasks'][task_name]
            if mode not in task_data:
                continue

            mode_data = task_data[mode]
            files = {}

            # Extract GT and Pred files if available
            for key in ['gt', 'pred', 'source_file']:
                if key in mode_data:
                    if isinstance(mode_data[key], dict):
                        results_file = mode_data[key].get('source_file')
                    else:
                        results_file = mode_data[key]

                    samples_file = self.loader.get_samples_file(results_file, task_name)
                    if samples_file:
                        files[key] = samples_file

            if files:
                source_files[model_key] = files

        return source_files

    @staticmethod
    def analyze_predictions(data: list[dict]) -> dict:
        """Analyze predictions and separate MCQ and NAQ questions"""
        mcq_results = {}
        na_results = {}

        for item in data:
            vsibench_score = item.get('vsibench_score', {})
            doc_id = item.get('doc_id')
            meta = vsibench_score.get('meta', {})

            # Extract predicted answer (same logic as visualize_qa_confusion_matrix.py)
            predicted_answer = vsibench_score.get('prediction', '')
            if not predicted_answer and 'filtered_resps' in item:
                predicted_answer = item['filtered_resps'][0] if item['filtered_resps'] else ''

            if 'accuracy' in vsibench_score:
                # MCQ question - use score (1.0 for correct, 0.0 for incorrect)
                mcq_results[doc_id] = {
                    'score': 1.0 if vsibench_score['accuracy'] else 0.0,
                    'type': meta.get('type'),
                    'category': meta.get('category'),
                    'predicted_answer': str(predicted_answer).strip(),
                }
            elif 'MRA:.5:.95:.05' in vsibench_score:
                # NAQ question - use score (MRA value 0-1)
                na_results[doc_id] = {
                    'score': vsibench_score['MRA:.5:.95:.05'],
                    'type': meta.get('type'),
                    'category': meta.get('category'),
                    'predicted_answer': str(predicted_answer).strip(),
                }

        mcq_accuracy = np.mean([r['score'] for r in mcq_results.values()]) if mcq_results else 0
        na_mean_mra = np.mean([r['score'] for r in na_results.values()]) if na_results else 0

        return {
            'mcq': mcq_results,
            'na': na_results,
            'mcq_accuracy': mcq_accuracy,
            'na_mean_mra': na_mean_mra
        }

class StatisticsCalculator:
    """Calculate statistics and metrics"""

    LOOKUP_CATEGORIES = ['direction', 'distance', 'size']
    REASONING_CATEGORIES = ['ranking', 'counting', 'transformation', 'deduction']

    @staticmethod
    def calculate_category_accuracy(all_results: dict) -> dict:
        """Calculate macro-average accuracy using (type, category) tuples"""
        type_category_stats = {}

        # Accumulate scores by (type, category) tuple
        for doc_id, result in all_results.items():
            q_type = result.get('type')
            category = result.get('category')
            score = result.get('score', 0.0)
            key = (q_type, category)

            if key not in type_category_stats:
                type_category_stats[key] = {'score_sum': 0.0, 'total': 0, 'accuracy': 0}

            type_category_stats[key]['total'] += 1
            type_category_stats[key]['score_sum'] += score

        # Calculate per-(type,category) accuracies
        for key in type_category_stats:
            if type_category_stats[key]['total'] > 0:
                type_category_stats[key]['accuracy'] = (
                    type_category_stats[key]['score_sum'] / type_category_stats[key]['total']
                )

        # Group by lookup/reasoning categories
        lookup_accuracies = []
        reasoning_accuracies = []
        all_accuracies = []

        for (q_type, category), stats in type_category_stats.items():  # noqa: F402
            acc = stats['accuracy']
            all_accuracies.append(acc)

            if category in StatisticsCalculator.LOOKUP_CATEGORIES:
                lookup_accuracies.append(acc)
            elif category in StatisticsCalculator.REASONING_CATEGORIES:
                reasoning_accuracies.append(acc)

        # Calculate macro-averages
        total_categories = len(type_category_stats)
        lookup_acc = sum(lookup_accuracies) / total_categories if total_categories > 0 else 0
        reasoning_acc = sum(reasoning_accuracies) / total_categories if total_categories > 0 else 0
        overall_acc = sum(all_accuracies) / total_categories if total_categories > 0 else 0

        # Calculate stats for reference
        lookup_score_sum = sum(
            stats['score_sum'] for (q_type, cat), stats in type_category_stats.items()
            if cat in StatisticsCalculator.LOOKUP_CATEGORIES
        )
        lookup_total = sum(
            stats['total'] for (q_type, cat), stats in type_category_stats.items()
            if cat in StatisticsCalculator.LOOKUP_CATEGORIES
        )
        reasoning_score_sum = sum(
            stats['score_sum'] for (q_type, cat), stats in type_category_stats.items()
            if cat in StatisticsCalculator.REASONING_CATEGORIES
        )
        reasoning_total = sum(
            stats['total'] for (q_type, cat), stats in type_category_stats.items()
            if cat in StatisticsCalculator.REASONING_CATEGORIES
        )

        return {
            'lookup': lookup_acc,
            'reasoning': reasoning_acc,
            'overall': overall_acc,
            'lookup_stats': {'score_sum': lookup_score_sum, 'total': lookup_total},
            'reasoning_stats': {'score_sum': reasoning_score_sum, 'total': reasoning_total},
            'type_category_details': type_category_stats,
            'num_categories': total_categories
        }

    @staticmethod
    def compare_results(base_results: dict, comp_results: dict, metric: str = 'score') -> dict:
        """Compare results across different settings"""
        common_doc_ids = set(base_results.keys()).intersection(set(comp_results.keys()))

        comparison = {
            'doc_ids': sorted(common_doc_ids),
            'base': [],
            'comp': [],
            'base_data': {},
            'comp_data': {},
        }

        for doc_id in sorted(common_doc_ids):
            comparison['base'].append(base_results[doc_id][metric])
            comparison['comp'].append(comp_results[doc_id][metric])
            comparison['base_data'][doc_id] = base_results[doc_id]
            comparison['comp_data'][doc_id] = comp_results[doc_id]

        return comparison


class Visualizer:
    """Handle all visualization tasks"""

    @staticmethod
    def format_model_name(model_name: str) -> str:
        """Format model name: keep only text after '__' if present"""
        return model_name.split('__', 1)[1] if '__' in model_name else model_name

    def plot_confusion_matrix(self, base_qa: list[bool], comp_qa: list[bool],
                             model_name: str, comp_label: str, output_path: str,
                             colormap: str = 'Blues',
                             consistent_hallucination: float | None = None):
        """Plot confusion matrix comparing two sets of predictions"""
        if not comp_qa:
            return

        cm = confusion_matrix(base_qa, comp_qa)
        cm = np.flipud(np.fliplr(cm))  # Flip to put True-True in top-left

        fig, ax = plt.subplots(figsize=(8, 7))

        sns.heatmap(cm, annot=True, fmt='d', cmap=colormap,
                   xticklabels=['True', 'False'],
                   yticklabels=['True', 'False'],
                   ax=ax, cbar_kws={'label': 'Count'})

        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')
        ax.set_xlabel(comp_label, fontsize=14, fontweight='bold', labelpad=10)
        ax.set_ylabel('Base QA', fontsize=14, fontweight='bold')
        ax.set_title(f'Base QA vs {comp_label}\n{model_name}',
                    fontsize=14, fontweight='bold', pad=20)

        # Add statistics
        total = cm.sum()
        both_correct = cm[0, 0]
        both_wrong = cm[1, 1]
        agreement = (both_correct + both_wrong) / total * 100

        ch_line = ''
        if consistent_hallucination is not None:
            ch_line = f'\nConsistent Hallucination: {consistent_hallucination*100:.1f}%'
        ax.text(0.5, -0.12,
               f'Agreement: {agreement:.1f}%\n'
               f'Both Correct: {both_correct/total*100:.1f}%    Both Wrong: {both_wrong/total*100:.1f}%'
               + ch_line,
               ha='center', transform=ax.transAxes, fontsize=11)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_mra_distribution(self, base_mra: np.ndarray, comp_mra: np.ndarray,
                             model_name: str, comp_label: str, output_path: str,
                             color: str = '#ff7f0e'):
        """Plot MRA distribution comparison with KS test"""
        if comp_mra is None or len(comp_mra) == 0:
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        # 11 bins centered at 0, 0.1, 0.2, ..., 1.0
        bins = np.linspace(-0.05, 1.05, 12)

        base_weights = np.ones_like(base_mra) / len(base_mra) * 100
        comp_weights = np.ones_like(comp_mra) / len(comp_mra) * 100

        ax.hist(base_mra, bins=bins, alpha=0.6, label='Base QA',
               color='#1f77b4', edgecolor='black', linewidth=0.8,
               weights=base_weights, width=0.09)
        ax.hist(comp_mra, bins=bins, alpha=0.6, label=comp_label,
               color=color, edgecolor='black', linewidth=0.8,
               weights=comp_weights, width=0.09)

        ax.set_xlabel('MRA Score', fontsize=14, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=14, fontweight='bold')
        ax.set_title(f'Base QA vs {comp_label}\n{model_name}',
                    fontsize=14, fontweight='bold', pad=15)
        ax.legend(fontsize=11, loc='upper left')
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_xlim(-0.05, 1.05)

        # Add median lines
        base_median = np.median(base_mra)
        comp_median = np.median(comp_mra)
        ax.axvline(base_median, color='#1f77b4', linestyle='--', linewidth=2,
                  alpha=0.8, label=f'Base QA Median: {base_median:.3f}')
        ax.axvline(comp_median, color=color, linestyle='--', linewidth=2,
                  alpha=0.8, label=f'{comp_label} Median: {comp_median:.3f}')
        ax.legend(fontsize=10, loc='upper left')

        # KS test and statistics
        ks_statistic, p_value = stats.ks_2samp(base_mra, comp_mra)
        base_mean = np.mean(base_mra)
        comp_mean = np.mean(comp_mra)
        base_std = np.std(base_mra)
        comp_std = np.std(comp_mra)

        stats_text = (
            f'Base QA:\n  Mean={base_mean:.3f}, Median={base_median:.3f}\n  Std={base_std:.3f}\n'
            f'{comp_label}:\n  Mean={comp_mean:.3f}, Median={comp_median:.3f}\n  Std={comp_std:.3f}\n'
            f'\nKS Test:\n  Statistic={ks_statistic:.4f}\n  p-value={p_value:.4e}'
        )

        ax.text(0.98, 0.97, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               family='monospace')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    def plot_stacked_bar_chart(self, model_data: dict, output_path: str):
        """Plot stacked bar chart showing lookup and reasoning accuracy, with text-only baseline"""
        if not model_data:
            print("No data available for stacked bar chart")
            return

        fig, ax = plt.subplots(figsize=(max(16, len(model_data) * 4), 8))

        models = list(model_data.keys())
        x = np.arange(len(models))
        width = 0.22  # Slightly reduced width to accommodate 4 bars

        # Colors
        lookup_color = '#1f77b4'
        reasoning_color = '#ff7f0e'

        settings = ['base_qa', 'pred_sg', 'gt_sg', 'text_only_gt_sg']
        setting_labels = ['Base QA', 'Pred SG Input', 'GT SG Input', 'Text-only GT SG']
        offsets = [-1.5*width, -0.5*width, 0.5*width, 1.5*width]

        for i, (setting, label, offset) in enumerate(zip(settings, setting_labels, offsets)):
            lookup_values = []
            reasoning_values = []

            for model_name in models:
                if setting in model_data[model_name]:
                    lookup_values.append(model_data[model_name][setting]['lookup'] * 100)
                    reasoning_values.append(model_data[model_name][setting]['reasoning'] * 100)
                else:
                    lookup_values.append(0)
                    reasoning_values.append(0)

            # Plot stacked bars
            bars1 = ax.bar(x + offset, lookup_values, width,
                      color=lookup_color, alpha=0.5 + i*0.15,
                      edgecolor='black', linewidth=0.8)
            bars2 = ax.bar(x + offset, reasoning_values, width, bottom=lookup_values,
                      color=reasoning_color, alpha=0.5 + i*0.15,
                      edgecolor='black', linewidth=0.8)

            # Add value labels (only for bars with sufficient height)
            for j, (bar1, bar2, lookup_val, reasoning_val) in enumerate(
                zip(bars1, bars2, lookup_values, reasoning_values)):

                if lookup_val > 5:
                    height1 = bar1.get_height()
                    ax.text(bar1.get_x() + bar1.get_width()/2., height1/2,
                       f'{lookup_val:.1f}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')

                if reasoning_val > 5:
                    height2 = bar2.get_height()
                    ax.text(bar2.get_x() + bar2.get_width()/2., lookup_val + height2/2,
                       f'{reasoning_val:.1f}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')

                if lookup_val > 0 or reasoning_val > 0:
                    total = lookup_val + reasoning_val
                    ax.text(bar2.get_x() + bar2.get_width()/2., total + 1,
                       f'{total:.1f}%', ha='center', va='bottom',
                       fontsize=12, fontweight='bold')

        ax.set_xlabel('Models', fontsize=16, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=0, ha='center', fontsize=11)
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Legend with clearer labels
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=lookup_color, alpha=0.6, edgecolor='black',
             label='Atomic Access (direction, distance, size)'),
            Patch(facecolor=reasoning_color, alpha=0.6, edgecolor='black',
             label='Compositional Logic (ranking, counting, transformation, deduction)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12,
             title='Category Type', title_fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()


class ResultAnalyzer:
    """Main analyzer orchestrating all operations"""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.aggregated_qa_sg_file = os.path.join(results_dir, 'qa_sg', 'aggregated_qa_sg.json')
        self.aggregated_file = os.path.join(results_dir, 'aggregated.json')

        self.loader = DataLoader()
        self.extractor = DataExtractor(self.loader)
        self.calculator = StatisticsCalculator()
        self.visualizer = Visualizer()

        self.aggregated_qa_sg_data = None
        self.aggregated_data = None
        self.results = {}
        self.model_names = {}

    def load_data(self):
        """Load all aggregated data"""
        self.aggregated_qa_sg_data = self.loader.load_json(self.aggregated_qa_sg_file)
        self.aggregated_data = self.loader.load_json(self.aggregated_file)

        print(f"Loaded {len(self.aggregated_qa_sg_data)} models from aggregated_qa_sg")
        print(f"Loaded {len(self.aggregated_data)} models from aggregated")

    def get_common_models(self) -> set:
        """Get models present in both aggregated files"""
        qa_sg_models = set(self.aggregated_qa_sg_data.keys())
        qa_models = set(self.aggregated_data.keys())
        common_models = qa_sg_models.intersection(qa_models)
        print(f"\nProcessing {len(common_models)} common models")
        return common_models

    def process_models(self, task_name: str = 'crisp_qa_sg', mode: str = 'multimodal'):
        """Process all models from both aggregated files"""
        common_models = self.get_common_models()

        # Extract model names
        for model_key, model_data in self.aggregated_qa_sg_data.items():
            if model_key in common_models:
                self.model_names[model_key] = model_data.get('model', model_key)

        # Process aggregated_qa_sg (with scene graph) - multimodal
        source_files_qa_sg = self.extractor.extract_source_files(
            self.aggregated_qa_sg_data, task_name, mode, common_models
        )

        for model_key, files in source_files_qa_sg.items():
            if model_key not in self.results:
                self.results[model_key] = {}

            if 'gt' in files:
                data = self.loader.load_jsonl(files['gt'])
                self.results[model_key]['qa_sg_gt'] = self.extractor.analyze_predictions(data)

            if 'pred' in files:
                data = self.loader.load_jsonl(files['pred'])
                self.results[model_key]['qa_sg_pred'] = self.extractor.analyze_predictions(data)

        # Process aggregated_qa_sg (with scene graph) - text_only GT SG
        source_files_qa_sg_text_only = self.extractor.extract_source_files(
            self.aggregated_qa_sg_data, task_name, 'text_only', common_models
        )

        for model_key, files in source_files_qa_sg_text_only.items():
            if model_key not in self.results:
                self.results[model_key] = {}

            if 'gt' in files:
                data = self.loader.load_jsonl(files['gt'])
                self.results[model_key]['qa_sg_gt_text_only'] = self.extractor.analyze_predictions(data)

        # Process aggregated (without scene graph)
        source_files_qa = self.extractor.extract_source_files(
            self.aggregated_data, 'crisp_qa', mode, common_models
        )

        for model_key, files in source_files_qa.items():
            if model_key not in self.results:
                self.results[model_key] = {}

            if 'source_file' in files:
                data = self.loader.load_jsonl(files['source_file'])
                self.results[model_key]['qa_multimodal'] = self.extractor.analyze_predictions(data)

    def generate_visualizations(self, output_dirs: dict[str, str]):
        """Generate all visualizations"""
        confusion_dir = output_dirs['confusion']
        mra_dir = output_dirs['mra']
        stacked_dir = output_dirs['stacked']

        os.makedirs(confusion_dir, exist_ok=True)
        os.makedirs(mra_dir, exist_ok=True)
        os.makedirs(stacked_dir, exist_ok=True)

        # Prepare data for stacked bar chart
        stacked_bar_data = {}

        print("\nGenerating visualizations...")

        for model_key, model_results in self.results.items():
            has_base_qa = 'qa_multimodal' in model_results
            has_gt_sg = 'qa_sg_gt' in model_results
            has_pred_sg = 'qa_sg_pred' in model_results

            if not (has_base_qa and (has_gt_sg or has_pred_sg)):
                continue

            model_name = self.visualizer.format_model_name(
                self.model_names.get(model_key, model_key)
            )
            safe_model_name = model_key.replace('/', '_')

            print(f"\n{'='*80}")
            print(f"Processing {model_name}...")
            print('='*80)

            # MCQ Analysis - Confusion Matrices
            base_mcq = model_results['qa_multimodal'].get('mcq', {})

            if base_mcq:
                if has_gt_sg:
                    gt_mcq = model_results['qa_sg_gt'].get('mcq', {})
                    comparison = self.calculator.compare_results(base_mcq, gt_mcq, 'score')

                    if comparison['doc_ids']:
                        output_path = os.path.join(confusion_dir,
                                                  f'{safe_model_name}_gt_confusion_matrix.png')
                        # Convert score to bool for confusion matrix (1.0 -> True, 0.0 -> False)
                        base_correct = [s > 0.5 for s in comparison['base']]
                        comp_correct = [s > 0.5 for s in comparison['comp']]
                        # Compute consistent hallucination: both wrong AND same predicted answer
                        total_cmp = len(comparison['doc_ids'])
                        ch_count = sum(
                            1 for doc_id in comparison['doc_ids']
                            if not (comparison['base_data'][doc_id]['score'] > 0.5)
                            and not (comparison['comp_data'][doc_id]['score'] > 0.5)
                            and (comparison['base_data'][doc_id].get('predicted_answer', '').upper() ==
                                 comparison['comp_data'][doc_id].get('predicted_answer', '').upper())
                        )
                        consistent_hallucination = ch_count / total_cmp if total_cmp > 0 else None
                        self.visualizer.plot_confusion_matrix(
                            base_correct, comp_correct,
                            model_name, 'GT SG Input QA', output_path, 'Blues',
                            consistent_hallucination=consistent_hallucination
                        )

                if has_pred_sg:
                    pred_mcq = model_results['qa_sg_pred'].get('mcq', {})
                    comparison = self.calculator.compare_results(base_mcq, pred_mcq, 'score')

                    if comparison['doc_ids']:
                        output_path = os.path.join(confusion_dir,
                                                  f'{safe_model_name}_pred_confusion_matrix.png')
                        # Convert score to bool for confusion matrix (1.0 -> True, 0.0 -> False)
                        base_correct = [s > 0.5 for s in comparison['base']]
                        comp_correct = [s > 0.5 for s in comparison['comp']]
                        # Compute consistent hallucination: both wrong AND same predicted answer
                        total_cmp = len(comparison['doc_ids'])
                        ch_count = sum(
                            1 for doc_id in comparison['doc_ids']
                            if not (comparison['base_data'][doc_id]['score'] > 0.5)
                            and not (comparison['comp_data'][doc_id]['score'] > 0.5)
                            and (comparison['base_data'][doc_id].get('predicted_answer', '').upper() ==
                                 comparison['comp_data'][doc_id].get('predicted_answer', '').upper())
                        )
                        consistent_hallucination = ch_count / total_cmp if total_cmp > 0 else None
                        self.visualizer.plot_confusion_matrix(
                            base_correct, comp_correct,
                            model_name, 'Pred SG Input QA', output_path, 'Oranges',
                            consistent_hallucination=consistent_hallucination
                        )

            # NAQ Analysis - MRA Distributions
            base_na = model_results['qa_multimodal'].get('na', {})

            if base_na:
                base_mra = np.array([r['score'] for doc_id in sorted(base_na.keys())
                                    for r in [base_na[doc_id]]])

                if has_gt_sg:
                    gt_na = model_results['qa_sg_gt'].get('na', {})
                    comparison = self.calculator.compare_results(base_na, gt_na, 'score')

                    if comparison['doc_ids']:
                        gt_mra = np.array(comparison['comp'])
                        output_path = os.path.join(mra_dir,
                                                  f'{safe_model_name}_gt_mra_distribution.png')
                        self.visualizer.plot_mra_distribution(
                            base_mra[:len(gt_mra)], gt_mra,
                            model_name, 'GT SG Input QA', output_path, '#ff7f0e'
                        )

                if has_pred_sg:
                    pred_na = model_results['qa_sg_pred'].get('na', {})
                    comparison = self.calculator.compare_results(base_na, pred_na, 'score')

                    if comparison['doc_ids']:
                        pred_mra = np.array(comparison['comp'])
                        output_path = os.path.join(mra_dir,
                                                  f'{safe_model_name}_pred_mra_distribution.png')
                        self.visualizer.plot_mra_distribution(
                            base_mra[:len(pred_mra)], pred_mra,
                            model_name, 'Pred SG Input QA', output_path, '#d62728'
                        )

            # Prepare stacked bar chart data
            data = {}
            for setting, key in [('base_qa', 'qa_multimodal'),
                                ('gt_sg', 'qa_sg_gt'),
                                ('pred_sg', 'qa_sg_pred'),
                                ('text_only_gt_sg', 'qa_sg_gt_text_only')]:
                if key in model_results:
                    # Merge MCQ and NAQ results (both use 'score' field now)
                    merged = {**model_results[key].get('mcq', {}),
                             **model_results[key].get('na', {})}
                    if merged:
                        data[setting] = self.calculator.calculate_category_accuracy(merged)

            if data:
                stacked_bar_data[model_name] = data

        # Generate stacked bar chart
        print("\n" + "="*80)
        print("Generating stacked bar chart for all models...")
        print("="*80)
        output_path = os.path.join(stacked_dir, 'stacked_bar_accuracy_by_category.png')
        self.visualizer.plot_stacked_bar_chart(stacked_bar_data, output_path)
        print(f"\nSaved stacked bar chart to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze spatial QA results and generate visualizations'
    )
    parser.add_argument('--input_dir', required=True,
                       help='Path to results directory containing aggregated files')
    parser.add_argument('--output_dir', default=None,
                       help='Output directory for visualizations (default: input_dir)')

    args = parser.parse_args()

    output_dir = args.output_dir if args.output_dir else args.input_dir
    output_dirs = {
        'confusion': os.path.join(output_dir, 'confusion_matrices'),
        'mra': os.path.join(output_dir, 'mra_distributions'),
        'stacked': os.path.join(output_dir, 'stacked_bar_charts'),
    }

    analyzer = ResultAnalyzer(args.input_dir)
    analyzer.load_data()
    analyzer.process_models()
    analyzer.generate_visualizations(output_dirs)

    print("\n" + "="*80)
    print("DONE!")
    print("="*80)

    return analyzer


if __name__ == '__main__':
    analyzer = main()
