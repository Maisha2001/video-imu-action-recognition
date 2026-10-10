import argparse
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from scipy.io.matlab import MatReadError
from starlette.datastructures import UploadFile

from serving_runtime import ModelRuntime


class BoundedUpload:
    def __init__(self, app, limit=75 * 1024 ** 2):
        self.app, self.limit = app, limit
        self.slots = BoundedSemaphore(1)

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope['method'] != 'POST':
            return await self.app(scope, receive, send)
        if not self.slots.acquire(blocking=False):
            return await JSONResponse({'detail': 'Inference busy; retry later'}, 503)(scope, receive, send)
        try:
            body = bytearray()
            while True:
                try:
                    message = await asyncio.wait_for(receive(), timeout=30)
                except TimeoutError:
                    return await JSONResponse({'detail': 'Upload timed out'}, 408)(scope, receive, send)
                if message['type'] == 'http.disconnect':
                    return
                chunk = message.get('body', b'')
                if len(body) + len(chunk) > self.limit:
                    return await JSONResponse({'detail': 'Request exceeds 75 MiB'}, 413)(scope, receive, send)
                body.extend(chunk)
                if not message.get('more_body', False):
                    break
            sent = False
            async def bounded_receive():
                nonlocal sent
                if sent:
                    return await receive()
                sent = True
                return {'type': 'http.request', 'body': bytes(body), 'more_body': False}
            await self.app(scope, bounded_receive, send)
        finally:
            self.slots.release()


def create_app(packages, backend='onnx', body_limit=75 * 1024 ** 2):
    if not packages or any(not name.replace('-', '').isalnum() for name in packages):
        raise ValueError('Provide named serving packages')

    @asynccontextmanager
    async def lifespan(app):
        app.state.models = {name: ModelRuntime(path, backend) for name, path in packages.items()}
        yield
        app.state.models.clear()

    app = FastAPI(title='Activity recognition', version='0.6.0', lifespan=lifespan,
                  telemetry={'auto_configure': False, 'tracing': False, 'metrics': False,
                             'logs': False, 'operation_spans': False})
    app.add_middleware(BoundedUpload, limit=body_limit)

    @app.get('/health')
    async def health():
        return {'status': 'ready', 'models': sorted(app.state.models)}

    @app.get('/models')
    async def models():
        return {name: {'family': m.family, 'sha256': m.manifest['source_model_sha256'],
                       'labels': m.manifest['labels'], 'fractions': [0.25, 0.5, 0.75, 1.0],
                       'limitations': m.manifest['limitations']} for name, m in app.state.models.items()}

    @app.post('/predict/{model_id}')
    async def predict(model_id: str, request: Request, fraction: float = 1.0, mode: str = 'conditional'):
        if model_id not in app.state.models:
            raise HTTPException(404, 'Unknown model')
        if fraction not in (0.25, 0.5, 0.75, 1.0) or mode not in ('pooled', 'conditional'):
            raise HTTPException(422, 'Unsupported fraction or calibration mode')
        try:
            async with request.form(max_files=2, max_fields=0, max_part_size=64 * 1024 ** 2) as form:
                if not form or any(key not in ('video', 'inertial') for key in form) or len(form.multi_items()) != len(form):
                    raise ValueError('Supply video and/or inertial once each')
                with TemporaryDirectory(prefix='activity-inference-') as temporary:
                    paths = {}
                    for key, upload in form.items():
                        if not isinstance(upload, UploadFile):
                            raise ValueError('Expected file uploads')
                        limit = (64 if key == 'video' else 10) * 1024 ** 2
                        content = await upload.read(limit + 1)
                        if not content or len(content) > limit:
                            raise ValueError('Empty or oversized file')
                        path = Path(temporary) / ('clip.avi' if key == 'video' else 'sensor.mat')
                        path.write_bytes(content)
                        paths[key] = path
                    return await asyncio.to_thread(app.state.models[model_id].predict, **paths, fraction=fraction, mode=mode)
        except (ValueError, OSError, KeyError, IndexError, RuntimeError, TypeError, MatReadError):
            raise HTTPException(422, 'Invalid recording or inference input') from None

    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser(description='Serve local activity models on loopback')
    parser.add_argument('--package', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--backend', choices=['onnx', 'torch'], default='onnx')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    pairs = [p.split('=', 1) for p in args.package]
    if any(len(p) != 2 for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
        parser.error('Use unique NAME=PATH packages')
    uvicorn.run(create_app(dict(pairs), args.backend), host='127.0.0.1', port=args.port, access_log=False)
