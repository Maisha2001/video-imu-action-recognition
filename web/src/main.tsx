import {useEffect, useRef, useState} from 'react';
import {createRoot} from 'react-dom/client';
import {actionName, fractions, frameAt, modalities, observedPrefix, predictionKey, readJson, sensorNames} from './domain';
import type {Model, Prediction, Recording, Trial} from './domain';
import './style.css';

function Signal({recording, offset, cursor}: {recording: Recording; offset: number; cursor: number}) {
  const rows = recording.imu;
  const values = rows.flatMap(r => r.slice(offset, offset + 3));
  const min = Math.min(...values), max = Math.max(...values), span = Math.max(max - min, 0.001);
  return <div className="signal"><div className="signal-heading"><strong>{offset === 0 ? 'Acceleration' : 'Angular velocity'}</strong><span>Source units · X / Y / Z</span></div>
    <svg viewBox="0 0 600 100" role="img" aria-label={`${offset === 0 ? 'Acceleration' : 'Angular velocity'} traces aligned by trial fraction`}>
      {[25,50,75].map(y => <line key={y} x1="0" x2="600" y1={y} y2={y} className="grid"/>)}
      {[0,1,2].map(channel => <polyline key={channel} className={`channel channel-${channel}`} points={rows.map((r,i) => `${i/(rows.length-1)*600},${90-(r[channel+offset]-min)/span*80}`).join(' ')}/>)}
      <rect x={cursor*600} y="0" width={(1-cursor)*600} height="100" className="future"/>
      <line x1={cursor*600} x2={cursor*600} y1="0" y2="100" className="cursor"/>
    </svg><div className="signal-axis"><span>{min.toFixed(2)}</span><span>Relative trial time →</span><span>{max.toFixed(2)}</span></div>
  </div>;
}

function App() {
  const [trials,setTrials] = useState<Trial[]>([]), [models,setModels] = useState<Record<string,Model>>({});
  const [selected,setSelected] = useState(''), [recording,setRecording] = useState<Recording>();
  const [cursor,setCursor] = useState(.25), [playing,setPlaying] = useState(false);
  const [predictions,setPredictions] = useState<Record<string,Prediction>>({});
  const [busy,setBusy] = useState(false), [done,setDone] = useState(0), [error,setError] = useState('');
  const request = useRef<AbortController | null>(null);
  useEffect(() => { const control = new AbortController();
    Promise.all([readJson<Trial[]>('/replay',{signal:control.signal}),readJson<Record<string,Model>>('/models',{signal:control.signal})])
      .then(([t,m]) => {setTrials(t);setModels(m);setSelected(t[0]?.id ?? '');})
      .catch(e => {if (!control.signal.aborted) setError(String(e.message));});
    return () => control.abort();
  },[]);
  useEffect(() => { if (!selected) return; request.current?.abort(); const control = new AbortController();
    setRecording(undefined);setPredictions({});setPlaying(false);setCursor(.25);setError('');setDone(0);
    readJson<Recording>(`/replay/${selected}`,{signal:control.signal}).then(setRecording)
      .catch(e => {if (!control.signal.aborted) setError(String(e.message));});
    return () => control.abort();
  },[selected]);
  useEffect(() => { if (!playing || !recording) return;
    const timer = setInterval(() => setCursor(value => Math.min(1,value + 1/recording.fps/recording.duration_seconds)),1000/recording.fps);
    return () => clearInterval(timer);
  },[playing,recording]);
  useEffect(() => {if (cursor >= 1) setPlaying(false);},[cursor]);
  useEffect(() => () => request.current?.abort(),[]);
  const prefix = observedPrefix(cursor);
  const total = Object.keys(models).length * fractions.length * modalities.length;
  async function analyze() {
    request.current?.abort(); const control = new AbortController(); request.current = control;
    setBusy(true);setDone(0);setError('');setPredictions({});
    try {
      for (const fraction of fractions) for (const [model] of Object.entries(models)) for (const modality of modalities) {
        const prediction = await readJson<Prediction>(`/replay/${selected}/predict/${model}?fraction=${fraction}&modality=${modality}`,{method:'POST',signal:control.signal});
        if (control.signal.aborted) return;
        setPredictions(p => ({...p,[predictionKey(model,modality,fraction)]:prediction}));setDone(n=>n+1);
      }
    } catch(e) {if (!control.signal.aborted) setError(e instanceof Error ? e.message : 'Analysis failed');}
    finally {setBusy(false);}
  }
  function download() {
    const blob = new Blob([JSON.stringify({trial:selected,alignment:recording?.alignment,predictions},null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob), link=document.createElement('a');link.href=url;link.download=`${selected}-predictions.json`;link.click();URL.revokeObjectURL(url);
  }
  return <main>
    <header><div className="brand-mark" aria-hidden="true">↗</div><div><p className="eyebrow">MULTIMODAL ACTIVITY RECOGNITION</p><h1>Watch the evidence unfold.</h1><p className="intro">Replay a recording. Compare video and motion. See when a prediction changes.</p></div><span className="local-badge">● Local research demo</span></header>
    <section className="toolbar" aria-label="Recording controls"><label>Recording<select aria-label="Recording" value={selected} disabled={busy} onChange={e=>setSelected(e.target.value)}>{trials.map(t=><option key={t.id} value={t.id}>{actionName(t.action)} · person {t.subject} · {t.id}</option>)}</select></label>
      <button className="primary" onClick={analyze} disabled={!recording || busy || !total}>{busy ? `Analyzing ${done}/${total}…` : 'Analyze recording'}</button>
      <button onClick={download} disabled={!done || busy}>Export predictions</button>
    </section>
    {error && <p className="error" role="alert">{error} <button onClick={()=>location.reload()}>Reload</button></p>}
    {!recording ? <p role="status">Loading the local recording…</p> : <>
      <div className="workspace"><section className="card video-panel"><div className="card-heading"><span>01 / Video</span><span className="pill">{recording.downstream_known_action ? 'Known action' : 'Held-out action'}</span></div>
        <div className="frame"><img src={`/replay/${selected}/frames/${frameAt(cursor,recording.frame_count)}`} alt={`${actionName(recording.action)} recording at ${Math.round(cursor*100)} percent`}/><div className="frame-caption"><strong>{actionName(recording.action)}</strong><span>Person {recording.subject} · {recording.sensor_placement} sensor</span></div></div>
        <div className="playback"><button aria-label={playing ? 'Pause replay' : 'Play replay'} onClick={()=>{if(cursor>=1)setCursor(0);setPlaying(!playing);}}>{playing ? 'Pause' : 'Play'}</button><label className="scrub">Observed recording<input aria-label="Observed recording" type="range" min="0" max="1" step="0.001" value={cursor} onChange={e=>{setPlaying(false);setCursor(Number(e.target.value));}}/></label><output>{Math.round(cursor*100)}%</output></div>
        <div className="prefix-buttons" aria-label="Prediction checkpoints">{fractions.map(f=><button aria-pressed={prefix===f} key={f} onClick={()=>{setCursor(f);setPlaying(false);}}>{f*100}%</button>)}</div>
      </section><section className="card"><div className="card-heading"><span>02 / Motion</span><span>{recording.imu.length} samples</span></div><Signal recording={recording} offset={0} cursor={cursor}/><Signal recording={recording} offset={3} cursor={cursor}/><p className="alignment">{recording.alignment}</p></section></div>
      <section className="results"><div className="section-heading"><div><p className="eyebrow">03 / MODEL COMPARISON</p><h2>{prefix ? `Predictions after ${prefix*100}% of the recording` : 'Observe at least 25% to see predictions'}</h2></div><span role="status">{done ? `${done}/${total} conditions analyzed` : 'Run analysis to compare models'}</span></div>
        {Object.entries(models).map(([id,model])=><div className="model-row" key={id}><div className="model-label"><h3>{model.family==='robust' ? 'Attention model' : 'Frozen video transfer'}</h3><span>Seed 42 · {model.family==='robust' ? 'pooled' : 'conditional'} calibration</span><code title={model.sha256}>Model {model.sha256.slice(0,12)}</code></div><div className="prediction-grid">{modalities.map(modality=>{
          const p=prefix ? predictions[predictionKey(id,modality,prefix)] : undefined;
          return <article className="prediction" key={modality}><p className="sensor-name">{sensorNames[modality]}</p>{p ? <><span className={`decision ${p.accepted?'accepted':'abstain'}`}>{p.accepted?'Accepted':'Abstained'}</span><h3>{actionName(p.top_action)}</h3><p>{p.accepted ? 'Top predicted action' : 'Highest score; no accepted action'}</p><div className="confidence"><strong>{(p.confidence*100).toFixed(2)}%</strong><span>confidence · threshold {(p.threshold*100).toFixed(2)}%</span></div><div className="meter"><i style={{width:`${p.confidence*100}%`}}/><b style={{left:`${p.threshold*100}%`}}/></div><p className="outcome">{p.top_action===recording.action?'Matches recorded action':'Differs from recorded action'}</p></> : <div className="empty">{prefix ? 'Awaiting analysis' : 'Not enough observed recording'}</div>}</article>;
        })}</div></div>)}
      </section>
      <aside className="limitations"><strong>Confidence is not a guarantee.</strong> These experimental models can accept an incorrect or unfamiliar action. Held-out actions also use a different sensor placement. Predictions use four offline prefixes of a known-length recording, not streaming onset detection. This demo does not establish real-time performance.</aside>
    </>}
    <footer><span>UTD-MHAD · Video + wearable motion</span><span>Recordings and inference stay on this computer.</span></footer>
  </main>;
}

createRoot(document.getElementById('root')!).render(<App/>);
