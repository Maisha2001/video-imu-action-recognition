import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from activity_data import file_hash
from attention_training import atomic_json


def analyze(results):
    report = {'method': 'Descriptive post-hoc analysis of published predictions; no model or threshold changes. Counts pool three seeds on the same trials and are not independent sample sizes.', 'models': {}}
    for family, stem, suffix in [('robustness', 'augmented', ''), ('transfer', 'transfer', '-conditional')]:
        records, hashes = [], {}
        for seed in (42, 43, 44):
            source = Path(results) / family / f'{stem}-{seed}.jsonl'
            hashes[source.relative_to(results).as_posix()] = file_hash(source)
            records.extend(json.loads(line) for line in source.read_text().splitlines())
        conditions = {}
        for prefix in (25, 100):
            for modality in ('both', 'video', 'imu'):
                key = f'f{prefix}-{modality}-n0{suffix}'
                rows = [r for r in records if r['condition'] == key]
                known, unknown = [r for r in rows if r['known']], [r for r in rows if not r['known']]
                if len(known) != 1008 or len(unknown) != 282:
                    raise ValueError('Expected three seeds over336 known and94 held-out trials')
                by_class = defaultdict(list)
                for row in known:
                    by_class[row['actual']].append(row)
                confused = Counter((r['actual'], r['predicted']) for r in known if r['actual'] != r['predicted'])
                conditions[key] = {'known_accuracy': sum(r['actual'] == r['predicted'] for r in known) / len(known),
                    'known_accepted_wrong': sum(not r['rejected'] and r['actual'] != r['predicted'] for r in known),
                    'known_predictions': len(known), 'unknown_accepted': sum(not r['rejected'] for r in unknown),
                    'unknown_predictions': len(unknown),
                    'lowest_recall_classes': sorted([{'action': k, 'recall': sum(r['actual'] == r['predicted'] for r in v)/len(v), 'predictions': len(v)} for k,v in by_class.items()], key=lambda d: (d['recall'], d['action']))[:5],
                    'largest_confusions': [{'actual': a, 'predicted': p, 'count': n} for (a,p),n in sorted(confused.items(), key=lambda x:(-x[1],x[0]))[:5]]}
        report['models'][family] = {'prediction_sha256': hashes, 'conditions': conditions}
    report['source_sha256'] = {'analyze_failures.py': file_hash(__file__)}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize errors without retuning evaluated models')
    parser.add_argument('--results', type=Path, default=Path('results'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error('Choose a new output')
    atomic_json(args.output, analyze(args.results))
