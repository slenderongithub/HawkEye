# HawkEye App — Context Snapshot

Written for continuity across context resets. This is Phase 1 of the HawkEye
companion app: a read-only ops console (React Native + Expo) for the indoor
surveillance product, built against the existing FastAPI backend with **zero
backend changes**. Plan file: `/Users/slender/.claude/plans/alright-man-we-gotta-delegated-cherny.md`.

## Status: Phase 1 complete

- 37 TypeScript files, ~4,700 lines, under `app/`.
- `tsc --noEmit`: 0 errors (strict mode).
- `npx expo export --platform web`: builds clean (1121 modules).
- Adversarial 3-lens review (imports/types, design consistency, RN correctness)
  ran across all screens: 9 findings, all minor. 7 fixed, 2 intentionally left
  (cosmetic Reanimated re-entrance-on-scroll quirks in FlatLists — not bugs).
- Nothing committed yet — all files are untracked in the repo. User has not
  asked for a commit.

## Design decisions (locked, from `/model` plan-mode session)

- Client: React Native + Expo (expo-router), single codebase for iOS/Android;
  Expo-web reused for the "laptop" experience (revisit vs. Electron in Phase 4
  only if desktop-class needs like tray/multi-window come up).
- v1 scope: full ops suite — live monitoring, incidents, drone control (UI/API
  scaffolding only), attendance, people/enrollment. Auth/roles deferred to
  Phase 2 (see below).
- Visual direction: dark "command center" — restrained, one blue accent
  (#3EA6FF), semantic colors only for meaning, real type hierarchy, soft depth.
  Motion: quick spring-based Reanimated micro-interactions, never bouncy/flashy.
  Explicit user brief: "top class UI to impress judges/panel, not too flashy,
  smooth animations."
- Remote access (Phase 2, not yet built): VPN mesh (Tailscale-style), not a
  public tunnel or open port.
- Push notifications (Phase 2, not yet built): Expo push service.
- Drone control (Phase 3, not yet built): contract-first — mock
  `DroneController` + API now, real hardware integration later. No real
  hardware exists yet per the product roadmap.

## What exists in the backend (unmodified, for reference)

- `server/app.py` — FastAPI, binds `127.0.0.1:8000` by default
  (`configs/default.yaml` → `server.host`). No auth, no CORS today.
  - REST: `GET /api/status`, `GET /api/health`, `GET /api/incidents?limit=`,
    `GET /api/attendance?date=`, `GET /api/people`, `POST /api/people`
    (multipart enroll), `DELETE /api/people/{id}`, `GET /api/evidence/{path}`.
  - WebSocket `/ws`: binary = JPEG frame (~15fps), text = JSON
    `{type: "incident"|"attendance"|"status", data}`. Replays last 20 incidents
    on connect.
- Data models live in `hawkeye/events/schema.py`: `Incident`, `Person`,
  `AttendanceRecord`, plus sub-objects `StageAEvent`, `Reasoning`, `FaceMatch`,
  `ActionRecord`. The app's `src/api/types.ts` zod schemas are a hand-mirrored
  copy of these — if the backend schema changes, update both.
- Storage: SQLite (`hawkeye/events/store.py`, `hawkeye/attendance/store.py`),
  evidence media as files under `data/evidence/<hash>/`.
- Existing (separate, still there) vanilla dashboard: `server/static/index.html`
  — untouched by this work, still served at `GET /`.

## App architecture

```
app/
  app.json, package.json, tsconfig.json, babel.config.js
  app/                              # expo-router routes
    _layout.tsx                     # root: providers, live socket, toast host, stack
    index.tsx                       # redirects to /(tabs)/live
    (tabs)/_layout.tsx               # bottom tab navigator (AnimatedTabBar)
    (tabs)/live.tsx                  # flagship screen — hand-built, sets the bar
    (tabs)/incidents.tsx             # severity filter chips + FlatList
    (tabs)/attendance.tsx            # date stepper + hero present-count
    (tabs)/people.tsx                # roster + photo enrollment + delete
    (tabs)/drone.tsx                 # contract-first PREVIEW (locked D-pad, no fake data)
    incident/[id].tsx                # detail — resolves from useIncidents() cache by id
    settings.tsx                     # server address, live health, about
  src/
    theme/tokens.ts, theme/index.ts  # design system — single source of truth
    api/types.ts                     # zod mirrors of backend pydantic models
    api/client.ts                    # fetch wrapper; generic infers zod OUTPUT type
    api/queries.ts                   # react-query hooks; INCIDENTS_LIMIT=50 shared const
    api/evidence.ts                  # evidenceUrl(path) → absolute URL
    ws/useLiveSocket.ts              # the one WebSocket connection, mounted in root layout
    stores/{live,settings,toast}.ts  # zustand stores
    components/                      # Screen, AppHeader, Card, Badge, Button, Chip,
                                      # LiveFeed, IncidentCard, AnimatedTabBar, ToastHost,
                                      # ConnectionBanner, EmptyState, Skeleton, StatusDot,
                                      # SectionHeader, PressableScale
    utils/format.ts                  # timeOfDay/relativeTime/percent/isoDate etc (epoch seconds in)
```

### Key conventions to preserve when extending

- Path alias `@/*` → `app/src/*`.
- All screens wrap in `<Screen>`; tab screens use `<AppHeader>` +
  `<ConnectionBanner>` + a scrollable list with `paddingBottom: 108` (clears the
  floating tab bar); root-stack screens (`incident/[id]`, `settings`) use
  `edges={{top:true,bottom:true}}` and a custom header row instead.
  the sole in-memory WebSocket connection.
- `useIncidents()` — call with no args (defaults to `INCIDENTS_LIMIT=50`) so
  Live/Incidents/detail all share one react-query cache entry.
- No auth yet — every screen assumes an open LAN connection. Server address is
  user-editable in Settings, defaults to `http://127.0.0.1:8000`, persisted via
  `expo-secure-store` (native) / `localStorage` (web fallback).
- Don't hardcode colors/spacing — always pull from `@/theme` tokens.

## How to run

```bash
cd app
npm install
npm start            # press i / a / w for iOS / Android / web
npm run typecheck    # tsc --noEmit
```

For a physical phone on the same Wi-Fi, the backend needs to bind wider than
its `127.0.0.1` default:
```bash
# from repo root
uvicorn server.app:app --host 0.0.0.0 --port 8000
```
Then set the LAN address (e.g. `http://192.168.1.42:8000`) in the app's
Settings → Server field.

## Not done yet (future phases, per the plan file)

- **Phase 2**: JWT auth + user/role model, CORS, Tailscale-based remote access,
  push notification dispatch (Expo push API) hooked into `hawkeye/hub.py`'s
  incident-publish path, `/api/v1/` versioning + generated TS client.
- **Phase 3**: real `DroneController` behind the Phase-1 mock UI/contract in
  `(tabs)/drone.tsx` — no hardware/firmware integration exists yet.
- **Phase 4**: offline incident cache, EAS build/store submission, revisit
  Expo-web vs. Electron for the laptop target if desktop-class needs
  (tray/multi-window) come up, accessibility pass.

## Known minor items intentionally left as-is

- `IncidentCard` (used in Live + Incidents) and `AttendanceRow` use Reanimated
  `entering` animations inside `FlatList`/scroll views — rows that scroll out
  of the render window and back in will replay their entrance animation. This
  is a known Reanimated + virtualized-list quirk, cosmetic only, not a bug.
- `people.tsx`'s enroll-form collapse uses an `exiting` animation inside a
  FlatList `ListHeaderComponent`, which Reanimated doesn't reliably animate;
  worst case it disappears instantly instead of animating out. Cosmetic only.
