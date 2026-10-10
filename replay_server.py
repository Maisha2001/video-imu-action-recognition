import argparse
import asyncio
from pathlib import Path

from fastapi import HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from inference_api import create_app
from replay_data import ReplayStore


def create_replay_app(packages, replay, ui, backend='onnx'):
    store = ReplayStore(replay)
    ui = Path(ui)
    if not (ui / 'index.html').is_file() or not (ui / 'assets').is_dir():
        raise ValueError('Build the React app first: npm ci and npm run build in web/')
    app = create_app(packages, backend)

    @app.get('/replay')
    async def recordings():
        return [{k: m[k] for k in ('id', 'action', 'subject', 'downstream_known_action')}
                for m in store.metadata.values()]

    @app.get('/replay/{key}')
    async def recording(key: str):
        try:
            return store.trial(key)
        except KeyError:
            raise HTTPException(404, 'Unknown recording') from None

    @app.get('/replay/{key}/frames/{number}')
    async def frame(key: str, number: int):
        try:
            return FileResponse(store.frame(key, number), media_type='image/jpeg')
        except KeyError:
            raise HTTPException(404, 'Unknown frame') from None

    @app.post('/replay/{key}/predict/{model_id}')
    async def predict(key: str, model_id: str, fraction: float = 1.0, modality: str = 'both', mode: str = 'conditional'):
        if model_id not in app.state.models:
            raise HTTPException(404, 'Unknown model')
        if fraction not in (0.25, 0.5, 0.75, 1.0) or mode not in ('pooled', 'conditional'):
            raise HTTPException(422, 'Unsupported inference condition')
        try:
            inputs = store.inputs(key, modality)
        except KeyError:
            raise HTTPException(404, 'Unknown recording') from None
        except ValueError:
            raise HTTPException(422, 'Unsupported sensor mode') from None
        result = await asyncio.to_thread(app.state.models[model_id].predict, **inputs, fraction=fraction, mode=mode)
        return {'trial': key, **result}

    @app.get('/')
    async def index():
        return FileResponse(ui / 'index.html')

    app.mount('/assets', StaticFiles(directory=ui / 'assets'), name='assets')
    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser(description='Run the local video/sensor replay app')
    parser.add_argument('--package', action='append', required=True, metavar='NAME=PATH')
    parser.add_argument('--replay', type=Path, required=True)
    parser.add_argument('--ui', type=Path, default=Path('web/dist'))
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    pairs = [p.split('=', 1) for p in args.package]
    if any(len(p) != 2 for p in pairs) or len({p[0] for p in pairs}) != len(pairs):
        parser.error('Use unique NAME=PATH packages')
    uvicorn.run(create_replay_app(dict(pairs), args.replay, args.ui), host='127.0.0.1', port=args.port, access_log=False)
