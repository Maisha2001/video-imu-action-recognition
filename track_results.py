import argparse
import json
import math
import os
from pathlib import Path
import re
import time

from activity_data import file_hash


def numeric_metrics(value, prefix=''):
    result = {}
    if isinstance(value, dict):
        for key, item in value.items():
            name = re.sub(r'[^\w./ -]', '_', str(key))
            result.update(numeric_metrics(item, f'{prefix}.{name}' if prefix else name))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        result[prefix] = float(value)
    return result


def import_results(source, store, experiment='Activity recognition'):
    os.environ['MLFLOW_DISABLE_TELEMETRY'] = 'true'
    os.environ['DO_NOT_TRACK'] = 'true'
    os.environ['MLFLOW_DISABLE_AGENT_HINT'] = 'true'
    from mlflow import MlflowClient
    from mlflow.entities import Metric
    source, store = Path(source).resolve(), Path(store).resolve()
    summary = json.loads(source.read_text())
    if not isinstance(summary.get('runs'), dict) or not summary['runs']:
        raise ValueError('Expected a completed study with named runs')
    digest = file_hash(source)
    store.mkdir(parents=True, exist_ok=True)
    uri = 'sqlite:///' + (store / 'tracking.db').as_posix()
    client = MlflowClient(tracking_uri=uri, registry_uri=uri)
    existing = client.get_experiment_by_name(experiment)
    if existing and existing.artifact_location != (store / 'artifacts').as_uri():
        raise ValueError('Existing experiment must use this local artifact directory')
    eid = existing.experiment_id if existing else client.create_experiment(experiment, artifact_location=(store / 'artifacts').as_uri())
    imported = []
    for name, run in summary['runs'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Unsupported run name')
        matches = client.search_runs([eid], filter_string=f"tags.source_sha256 = '{digest}' and tags.source_run = '{name}' and attributes.status = 'FINISHED'")
        if matches:
            imported.append({'source_run': name, 'run_id': matches[0].info.run_id, 'reused': True})
            continue
        record = client.create_run(eid, tags={'mlflow.runName': name, 'source_run': name,
                 'source_sha256': digest, 'kind': 'result_import',
                 'original_completed_at': str(summary.get('completed_at_utc', 'not recorded')),
                 'interpretation': 'Import of measured results, not a new training run. Import times are actual.'})
        rid = record.info.run_id
        try:
            for key, value in summary.get('config', {}).items():
                client.log_param(rid, key, value)
            for key in ('seed', 'variant', 'selected_epoch', 'model_sha256'):
                if key in run:
                    client.log_param(rid, key, run[key])
            metrics = [Metric(key, value, int(time.time() * 1000), 0) for key, value in numeric_metrics(run).items()]
            for offset in range(0, len(metrics), 500):
                client.log_batch(rid, metrics=metrics[offset:offset + 500])
            client.log_artifact(rid, str(source), artifact_path='measured-results')
            client.set_terminated(rid)
        except Exception:
            client.set_terminated(rid, status='FAILED')
            raise
        imported.append({'source_run': name, 'run_id': rid, 'reused': False})
    return {'source_sha256': digest, 'experiment_id': eid, 'runs': imported,
            'store': 'Local SQLite and local artifacts only; telemetry disabled.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Import measured study results into local MLflow')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--store', type=Path, default=Path('runs/tracking'))
    parser.add_argument('--experiment', default='Activity recognition')
    args = parser.parse_args()
    print(json.dumps(import_results(args.results, args.store, args.experiment), indent=2))
