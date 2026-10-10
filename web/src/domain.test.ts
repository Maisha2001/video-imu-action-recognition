import {describe,it,expect,vi} from 'vitest';
import {frameAt, observedPrefix, predictionKey, readJson} from './domain';

describe('visible evidence',()=>{
  it('never displays a prediction from a later prefix',()=>{
    expect(observedPrefix(.249)).toBeUndefined();expect(observedPrefix(.499)).toBe(.25);
    expect(observedPrefix(.75)).toBe(.75);expect(observedPrefix(1)).toBe(1);
  });
  it('keeps video frames within the cropped prefix and recording bounds',()=>{
    expect(frameAt(0,47)).toBe(0);expect(frameAt(.25,47)).toBe(10);
    expect(frameAt(1,47)).toBe(46);
  });
  it('separates model, sensor and fraction results',()=>{
    expect(predictionKey('robust','both',.25)).not.toBe(predictionKey('transfer','both',.25));
    expect(predictionKey('robust','both',.25)).not.toBe(predictionKey('robust','imu',.25));
  });
  it('surfaces a busy service without fabricating a prediction',async()=>{
    vi.stubGlobal('fetch',vi.fn().mockResolvedValue({ok:false,status:503,json:async()=>({detail:'Inference busy; retry later'})}));
    await expect(readJson('/predict')).rejects.toThrow('Inference busy');vi.unstubAllGlobals();
  });
});
