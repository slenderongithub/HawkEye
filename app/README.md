# HawkEye — Ops App

A cross-platform (iOS / Android / laptop-web) companion app for the HawkEye
indoor surveillance product. Built with **Expo (SDK 52) + expo-router +
Reanimated 3 + TypeScript**.

This is **Phase 1**: a polished, read-only operations console that talks to the
existing FastAPI backend (`../server/app.py`). Auth, push notifications, remote
access, and live drone control land in later phases (see the root `README.md`
roadmap and `.claude/plans/`).

## Screens

| Tab | What it does |
|-----|--------------|
| **Live** | Real-time video feed + runtime status + a live-updating recent-incidents preview. |
| **Incidents** | Full incident feed with severity filters and tap-through detail (AI assessment, identity match, detection + response). |
| **Attendance** | Per-day presence of enrolled people, with a date stepper. |
| **People** | Identity roster + photo enrollment (multi-image) + removal. |
| **Drone** | Contract-first preview of fleet control / patrol / manual override (Phase 3). |
| **Settings** | Server address, live connection + health, about. |

## Architecture

- **Design system** — `src/theme/tokens.ts` is the single source of truth
  (palette, spacing, radii, typography, motion). It implements Apple's design
  language on its **dark-tile chassis**: one accent (Sky Link Blue `#2997ff`,
  the dark-surface variant), elevation from surface colour and hairlines rather
  than shadow, exactly one drop-shadow spent on the live feed, no decorative
  gradients, 17px body copy, and the 300/400/600/700 weight ladder. Nothing
  outside that file hardcodes a colour, radius or tracking value.
- **Data** — `src/api/` wraps the REST endpoints with `@tanstack/react-query`
  and validates every payload with `zod` schemas that mirror the backend
  pydantic models (`hawkeye/events/schema.py`).
- **Live** — `src/ws/useLiveSocket.ts` holds one WebSocket to `/ws`: binary
  messages are JPEG frames (rendered via the live store), text messages are
  `incident` / `attendance` / `status` events fanned out to the query cache +
  toasts. Mounted once in `app/_layout.tsx`.
- **UI kit** — `src/components/` (Card, Badge, Button, LiveFeed, IncidentCard,
  AnimatedTabBar, ToastHost, …). All motion is Reanimated on the UI thread.

## Running

### One address (the normal way)

The FastAPI backend serves this app's web build itself, so the dashboard, the
REST API and the WebSocket all live at **one address** — there is no second UI
and no second port:

```bash
cd app && npm install && npm run build:web   # -> app/dist, served by the backend
cd .. && uvicorn server.app:app              # open http://127.0.0.1:8000
```

On web the app defaults its server address to `location.origin`, so opening that
URL just works with no trip through Settings. Rebuild (`npm run build:web`) after
changing app code; the backend picks up the new `dist/` on the next request.

If you open the backend before building the app, it says so and tells you the
command to run rather than 404ing.

### Iterating on the UI

```bash
npm start           # Expo dev server — press i / a / w for iOS / Android / web
npm run typecheck   # tsc --noEmit
```

The dev server runs on its own port with hot reload and talks to the backend on
`http://127.0.0.1:8000`. Use it while working on the app; use the build above for
everything else.

### A physical phone on the same Wi-Fi

Native builds have no origin to default to, so they keep the LAN address and the
Settings override. Make the backend reachable on the network:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then open **Settings → Server**, enter your machine's LAN address (e.g.
`http://192.168.1.42:8000`), and tap *Save & reconnect*. (Off-LAN access via a
private VPN mesh is a Phase 2 item — the app is LAN-only today.)

> Note: the backend binds `127.0.0.1` by default (`configs/default.yaml` →
> `server.host`); `--host 0.0.0.0` overrides it for LAN demos without changing
> product code.
