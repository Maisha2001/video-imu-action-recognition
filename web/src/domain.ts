export const fractions = [0.25, 0.5, 0.75, 1];
export const modalities = ['both', 'video', 'imu'] as const;
export type Modality = typeof modalities[number];
export const sensorNames: Record<Modality, string> = {both: 'Video + motion', video: 'Video only', imu: 'Motion only'};
export const actions = ['Swipe left', 'Swipe right', 'Hand wave', 'Clap', 'Throw', 'Cross arms', 'Basketball shot', 'Draw X', 'Clockwise circle', 'Counterclockwise circle', 'Draw triangle', 'Bowling', 'Boxing', 'Baseball swing', 'Tennis forehand', 'Arm curl', 'Tennis serve', 'Push', 'Knock', 'Catch', 'Pick up and throw', 'Jog in place', 'Walk in place', 'Sit to stand', 'Stand to sit', 'Lunge', 'Squat'];
export const actionName = (id: number) => actions[id - 1] ?? `Action ${id}`;
export const observedPrefix = (cursor: number) => fractions.filter(f => f <= cursor + 1e-8).at(-1);
export const frameAt = (cursor: number, count: number) => Math.min(count - 1, Math.max(0, Math.floor(cursor * count) - 1));
export const predictionKey = (model: string, modality: Modality, fraction: number) => `${model}/${modality}/${fraction}`;
export type Trial = {id: string; action: number; subject: number; downstream_known_action: boolean};
export type Recording = Trial & {frame_count: number; fps: number; duration_seconds: number; imu: number[][]; alignment: string; sensor_placement: string};
export type Prediction = {trial: string; top_action: number; accepted: boolean; confidence: number; threshold: number; model_sha256: string; fraction: number; probabilities: number[]; labels: number[]; calibration_mode: string};
export type Model = {family: string; sha256: string; limitations: string[]};

export async function readJson<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, options);
  if (!response.ok) {
    let reason = `Request failed (${response.status})`;
    try { const data = await response.json(); if (typeof data.detail === 'string') reason = data.detail; } catch { /* Non-JSON errors retain their status. */ }
    throw new Error(reason);
  }
  return response.json() as Promise<T>;
}
