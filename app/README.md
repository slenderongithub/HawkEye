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

- **Design system** — `src/theme/` is the single source of truth (palette,
  spacing, radii, typography, motion). It extends the operator dashboard's dark
  palette so the app reads as the same product.
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

```bash
cd app
npm install
npm start          # Expo dev server — press i / a / w for iOS / Android / web
npm run typecheck  # tsc --noEmit
```

### Point the app at your server

The app defaults to `http://127.0.0.1:8000` (works for the iOS simulator and
web). To use a **physical phone on the same Wi-Fi**, the backend must be
reachable on the LAN:

```bash
# from the repo root
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Then open **Settings → Server** in the app and enter your machine's LAN address,
e.g. `http://192.168.1.42:8000`, and tap *Save & reconnect*. (Off-LAN access via
a private VPN mesh is a Phase 2 item — the app is LAN-only today.)

> Note: the current backend binds `127.0.0.1` by default (`configs/default.yaml`
> → `server.host`); `--host 0.0.0.0` above overrides it for LAN demos without
> changing product code.
