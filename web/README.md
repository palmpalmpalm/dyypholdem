# DyypHoldem web table

A compact React 19 + TypeScript frontend for authenticated, same-origin
DyypHoldem play sessions. The ACPC dealer and Python bridge remain authoritative;
this directory contains presentation and API-adapter code only.

```bash
npm ci
npm test
npm run build
```

`npm run dev` starts Vite on port 5173 and proxies `/api/*` to the existing
server on port 8000. Production assets are written to `dist/`.

For deterministic visual QA without ACPC or a GPU, run `npm run dev:mock`.
The representative fixture is implemented as Vite development middleware and
is never included in the browser production bundle.

Authentication is expected to use an HttpOnly same-origin cookie. The frontend
never reads or persists a session secret. If a legacy launch URL contains a
`token` query parameter, the app removes that parameter from the visible URL
without reading its value.

The UI treats `/api/state` as the source of truth for legal actions. Arbitrary
raise sizing is enabled only when the server advertises a generic bet/raise
action and bounds. A legacy server that advertises only discrete targets is
also supported; in that case the sizer exposes only those exact legal sizes.
