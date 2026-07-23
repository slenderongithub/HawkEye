/**
 * Runtime-validated mirrors of the backend pydantic models
 * (hawkeye/events/schema.py). Every field here matches a field the server
 * emits via model_dump_json(). zod gives us parse-time validation of REST +
 * WebSocket payloads so a schema drift surfaces loudly instead of as a silent
 * `undefined` deep in a screen.
 */
import { z } from 'zod';

export const AnomalyType = z.enum([
  'normal',
  'violence',
  'robbery',
  'theft',
  'weapon',
  'intrusion',
  'fall',
  'medical',
  'accident',
  'fire',
  'unknown',
]);
export type AnomalyType = z.infer<typeof AnomalyType>;

export const Severity = z.enum(['low', 'medium', 'high', 'critical']);
export type Severity = z.infer<typeof Severity>;

export const ActionType = z.enum(['none', 'police', 'ambulance', 'fire']);
export type ActionType = z.infer<typeof ActionType>;

export const IncidentStatus = z.enum(['detected', 'confirmed', 'dismissed']);
export type IncidentStatus = z.infer<typeof IncidentStatus>;

export const BBox = z.object({
  x1: z.number(),
  y1: z.number(),
  x2: z.number(),
  y2: z.number(),
});

export const StageAEvent = z.object({
  frame_id: z.number(),
  ts: z.number(),
  anomaly_type: AnomalyType.catch('unknown'),
  score: z.number().default(0),
  motion_score: z.number().default(0),
  person_count: z.number().default(0),
  boxes: z.array(BBox).default([]),
  triggered: z.boolean().default(false),
  reason: z.string().default(''),
});

export const Reasoning = z.object({
  confirmed: z.boolean().default(false),
  description: z.string().default(''),
  anomaly_type: AnomalyType.catch('unknown'),
  severity: Severity.catch('low'),
  recommended_action: ActionType.catch('none'),
  confidence: z.number().default(0),
  backend: z.string().default('mock'),
});

export const FaceMatch = z.object({
  attempted: z.boolean().default(false),
  matched: z.boolean().default(false),
  person_id: z.string().nullish(),
  name: z.string().nullish(),
  similarity: z.number().default(0),
  face_image: z.string().nullish(),
});

export const ActionRecord = z.object({
  action: ActionType.catch('none'),
  dispatched: z.boolean().default(false),
  severity: Severity.catch('low'),
  reason: z.string().default(''),
  dispatchers: z.array(z.string()).default([]),
  ts: z.number(),
});

export const Incident = z.object({
  id: z.string(),
  ts: z.number(),
  anomaly_type: AnomalyType.catch('unknown'),
  score: z.number().default(0),
  status: IncidentStatus.catch('detected'),
  stage_a: StageAEvent.nullish(),
  reasoning: Reasoning.nullish(),
  face_match: FaceMatch.nullish(),
  action: ActionRecord.nullish(),
  thumb: z.string().nullish(),
  clip_path: z.string().nullish(),
  // server adds this on the REST/ws payload (Incident.headline())
  headline: z.string().nullish(),
});
export type Incident = z.infer<typeof Incident>;

export const Person = z.object({
  id: z.string(),
  name: z.string(),
  meta: z.record(z.string()).default({}),
  enrolled_at: z.number(),
  num_faces: z.number().default(0),
});
export type Person = z.infer<typeof Person>;

export const AttendanceRecord = z.object({
  person_id: z.string(),
  name: z.string(),
  date: z.string(),
  first_seen: z.number(),
  last_seen: z.number(),
  sightings: z.number().default(1),
});
export type AttendanceRecord = z.infer<typeof AttendanceRecord>;

/** Runtime status pushed on the ws `status` event / GET /api/status. */
export const Status = z
  .object({
    running: z.boolean().default(false),
    fps: z.number().nullish(),
    reasoner: z.string().nullish(),
    identity: z.boolean().nullish(),
    source: z.string().nullish(),
    enrolled: z.number().nullish(),
    present_today: z.number().nullish(),
  })
  .passthrough();
export type Status = z.infer<typeof Status>;

/** GET /api/health body. */
export const Health = z.object({
  status: z.enum(['ok', 'degraded']),
  components: z.record(z.any()),
});
export type Health = z.infer<typeof Health>;

/** Discriminated ws text-event envelope. */
export const WsEvent = z.discriminatedUnion('type', [
  z.object({ type: z.literal('incident'), data: Incident }),
  z.object({ type: z.literal('attendance'), data: AttendanceRecord }),
  z.object({ type: z.literal('status'), data: Status }),
]);
export type WsEvent = z.infer<typeof WsEvent>;
