# DyypHoldem Live Browser Session

Date: 2026-08-23 (Asia/Bangkok)

## Architecture

The browser does not simulate poker rules. The bundled Linux ACPC dealer is
the authority for shuffling, private/public cards, legal transitions, street
advancement, showdown, and match state. The authenticated web process occupies
the human dealer seat and the unmodified game loop in
`player/dyypholdem_acpc_player.py` occupies the other seat.

```text
browser --HTTPS--> token-protected web seat --ACPC--> dealer <--ACPC-- DyypHoldem
```

Human buttons mirror the repository action abstraction: fold when facing a
bet, check/call, pot-size raise, and all-in. HTTP action requests only enqueue
an ACPC action and return immediately. The UI polls state while a long
continual resolve runs, which stays below RunPod's HTTP proxy request timeout.

## Launch and lifecycle

```shell
make play-ui-dry-run
make play-ui
```

The controller:

1. checks all four compact checkpoint files locally;
2. creates exactly one Secure RTX 4090 exposing only `22/tcp` and `8000/http`;
3. fixes an absolute paid-resource deadline, then arms both an authenticated
   remote stop and an independent local API stop/delete watchdog;
4. syncs code and compact checkpoints;
5. downloads and checksums the full play asset profile;
6. validates all recovered networks on CUDA;
7. starts dealer, UI, and real continual resolver;
8. waits for initialization, proxy health, and the first authenticated playable
   ACPC `MATCHSTATE` before publishing the URL;
9. copies the run directory home every 12 seconds;
10. quiesces all match processes, retries final copyback, then stops/deletes
    only the exact-name pod and requires six successful absence observations.

Use `make play-ui-stop` immediately after testing. `make play-ui-status` is a
redacted provider-status check. `make play-ui-logs` refreshes local artifacts
and prints only safe timing/result information.

## Decision timing contract

CUDA is synchronized at timing boundaries. Each private decision record
contains:

- hand number, decision number, street, board, pot, public bets/actions;
- 1,000 CFR iterations and the 500-iteration averaging skip;
- invariant/chance-reconstruction time;
- terminal-equity time;
- public-tree creation time;
- lookahead tensor allocation and lookahead-build time;
- CFR time and result extraction time;
- strategy sampling/invariant-update time;
- total response time;
- available-action probabilities and the sampled action;
- allocated, reserved, and per-decision peak CUDA memory;
- source commit, runtime versions, GPU name, and SHA-256 for every compact
  checkpoint.

The continuously refreshed safe report aggregates count, total, mean, p50,
p95, and maximum for every timing phase on preflop, flop, turn, and river.
The live report exposes chosen public actions and timing, but not strategy
probabilities or either player's private cards.

## Artifacts

Each ignored `runs/play-ui/<run-id>/` directory contains:

```text
decisions.jsonl          full private structured decisions
timing_report.json       safe live/final machine-readable aggregation
timing_report.txt        safe live/final human-readable aggregation
safe-events.jsonl        human actions and hand results
bot.log                  private raw bot log
dealer.stdout.log        private raw dealer stream
dealer.stderr.log        dealer diagnostics
<run-id>.log             ACPC dealer hand log
model-validation.json    compact checkpoint CUDA validation
model-validation.log     validation process output
play-assets.json         pinned asset sizes and checksum verification
environment.json         non-secret session settings
watchdog-config.json     pod identity and absolute deadline (no API key)
watchdog-status.json     local deadline-guard outcome, when it fires
```

The session token, SSH configuration, API responses, raw logs, and artifacts
stay under ignored `runs/`; none are committed.
