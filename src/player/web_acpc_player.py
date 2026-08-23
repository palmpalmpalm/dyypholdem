#!/usr/bin/env python3
"""Token-protected browser seat for the bundled ACPC dealer.

This process is intentionally independent from Torch. The ACPC dealer remains
the source of truth for cards, legal transitions, showdown, and match state;
the web process only bridges authenticated browser actions to one dealer seat.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
from pathlib import Path
import re
import secrets
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse


STACK = 20_000
SMALL_BLIND = 50
BIG_BLIND = 100
STREET_NAMES = {1: "preflop", 2: "flop", 3: "turn", 4: "river"}
MATCH_RE = re.compile(r"^MATCHSTATE:(\d):(\d*):([^:]*):(.*)$")
ACTION_RE = re.compile(r"r(\d+)|c|f")
CARD_RANKS = {rank: value for value, rank in enumerate("23456789TJQKA", start=2)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def split_cards(cards: str) -> list[str]:
    return [cards[index:index + 2] for index in range(0, len(cards), 2) if len(cards[index:index + 2]) == 2]


def parse_action_string(raw: str) -> list[dict[str, object]]:
    actions = []
    index = 0
    while index < len(raw):
        match = ACTION_RE.match(raw, index)
        if not match:
            raise ValueError(f"invalid ACPC action sequence near {raw[index:]!r}")
        token = match.group(0)
        if token == "c":
            actions.append({"kind": "call", "raise_to": 0})
        elif token == "f":
            actions.append({"kind": "fold", "raise_to": 0})
        else:
            actions.append({"kind": "raise", "raise_to": int(match.group(1))})
        index = match.end()
    return actions


def convert_actions(street_actions: list[list[dict[str, object]]]) -> list[dict[str, object]]:
    all_actions = [
        {"kind": "raise", "raise_to": SMALL_BLIND, "player": 0, "street": 1},
        {"kind": "raise", "raise_to": BIG_BLIND, "player": 1, "street": 1},
    ]
    for street_index, actions in enumerate(street_actions, start=1):
        first_player = 0 if street_index == 1 else 1
        for index, action in enumerate(actions):
            item = dict(action)
            item["player"] = first_player if index % 2 == 0 else 1 - first_player
            item["street"] = street_index
            all_actions.append(item)
    return all_actions


def acting_player(current_street: int, street_actions: list[list[dict[str, object]]], all_actions: list[dict[str, object]]) -> int:
    if len(all_actions) == 2:
        return 0
    last = all_actions[-1]
    if last["street"] != current_street:
        return 1
    if last["kind"] == "fold":
        return -1
    if current_street == 4 and len(street_actions[3]) >= 2 and last["kind"] == "call":
        return -1
    return 1 - int(last["player"])


def compute_bets(all_actions: list[dict[str, object]]) -> tuple[int, int]:
    terminal_fold = all_actions[-1]["kind"] == "fold"
    valid = all_actions[:-1] if terminal_fold else all_actions
    commitments = [SMALL_BLIND, BIG_BLIND]
    for action in valid[2:]:
        player = int(action["player"])
        if action["kind"] == "raise":
            commitments[player] = int(action["raise_to"])
        elif action["kind"] == "call":
            commitments[player] = commitments[1 - player]
    return commitments[0], commitments[1]


@dataclass
class DisplayState:
    raw: str
    position: int
    hand_number: int
    actions_raw: list[str]
    actions: list[list[dict[str, object]]]
    all_actions: list[dict[str, object]]
    board: str
    hand_p1: str
    hand_p2: str
    current_street: int
    acting_player: int
    bet1: int
    bet2: int

    @property
    def hero_player(self) -> int:
        return 1 - self.position

    @property
    def hero_hand(self) -> str:
        return self.hand_p1 if self.position == 0 else self.hand_p2

    @property
    def opponent_hand(self) -> str:
        return self.hand_p2 if self.position == 0 else self.hand_p1

    @property
    def hero_bet(self) -> int:
        return self.bet2 if self.position == 0 else self.bet1

    @property
    def opponent_bet(self) -> int:
        return self.bet1 if self.position == 0 else self.bet2

    @property
    def terminal(self) -> bool:
        return (
            self.acting_player == -1
            or (self.bet1 == STACK and self.bet2 == STACK)
        )


def parse_matchstate(message: str) -> DisplayState:
    match = MATCH_RE.match(message.strip())
    if not match:
        raise ValueError("not an ACPC MATCHSTATE message")
    position_raw, hand_raw, action_blob, cards_blob = match.groups()
    action_parts = (action_blob.split("/") + ["", "", "", ""])[:4]
    street_actions = [parse_action_string(part) for part in action_parts]

    card_match = re.match(r"([^|]*)\|([^/]*)/?([^/]*)/?([^/]*)/?([^/]*)", cards_blob)
    if not card_match:
        raise ValueError("invalid ACPC cards field")
    hand_p1, hand_p2, flop, turn, river = card_match.groups()
    board = f"{flop.strip()}{turn.strip()}{river.strip()}"
    current_street = 1 if not board else int(len(board) / 2 - 1)
    all_actions = convert_actions(street_actions)
    actor = acting_player(current_street, street_actions, all_actions)
    bet1, bet2 = compute_bets(all_actions)
    return DisplayState(
        raw=message.strip(),
        position=int(position_raw),
        hand_number=int(hand_raw),
        actions_raw=action_parts,
        actions=street_actions,
        all_actions=all_actions,
        board=board,
        hand_p1=hand_p1.strip(),
        hand_p2=hand_p2.strip(),
        current_street=current_street,
        acting_player=actor,
        bet1=bet1,
        bet2=bet2,
    )


def evaluate_five(cards: tuple[str, ...]) -> tuple[int, ...]:
    ranks = sorted((CARD_RANKS[card[0]] for card in cards), reverse=True)
    suits = [card[1] for card in cards]
    counts = Counter(ranks)
    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    unique = sorted(set(ranks), reverse=True)
    if 14 in unique:
        unique.append(1)
    straight_high = 0
    for start in range(len(unique) - 4):
        window = unique[start:start + 5]
        if window[0] - window[4] == 4:
            straight_high = window[0]
            break
    flush = len(set(suits)) == 1
    if flush and straight_high:
        return (8, straight_high)
    if groups[0][0] == 4:
        kicker = max(rank for rank in ranks if rank != groups[0][1])
        return (7, groups[0][1], kicker)
    if groups[0][0] == 3 and groups[1][0] == 2:
        return (6, groups[0][1], groups[1][1])
    if flush:
        return (5, *ranks)
    if straight_high:
        return (4, straight_high)
    if groups[0][0] == 3:
        kickers = sorted((rank for rank in ranks if rank != groups[0][1]), reverse=True)
        return (3, groups[0][1], *kickers)
    pairs = sorted((rank for count, rank in groups if count == 2), reverse=True)
    if len(pairs) >= 2:
        kicker = max(rank for rank in ranks if rank not in pairs[:2])
        return (2, pairs[0], pairs[1], kicker)
    if len(pairs) == 1:
        kickers = sorted((rank for rank in ranks if rank != pairs[0]), reverse=True)
        return (1, pairs[0], *kickers)
    return (0, *ranks)


def evaluate_seven(cards: list[str]) -> tuple[int, ...]:
    if len(cards) != 7:
        raise ValueError("seven cards are required for showdown evaluation")
    return max(evaluate_five(combo) for combo in itertools.combinations(cards, 5))


def hero_winnings(state: DisplayState) -> int:
    last = state.all_actions[-1]
    if last["kind"] == "fold":
        hero_won = state.hero_bet >= state.opponent_bet
    else:
        hero_cards = split_cards(state.hero_hand + state.board)
        opponent_cards = split_cards(state.opponent_hand + state.board)
        hero_rank = evaluate_seven(hero_cards)
        opponent_rank = evaluate_seven(opponent_cards)
        if hero_rank == opponent_rank:
            return 0
        hero_won = hero_rank > opponent_rank
    return state.opponent_bet if hero_won else -state.hero_bet


class HumanBridge:
    def __init__(self, dealer_host: str, dealer_port: int, events_path: Path):
        self.dealer_host = dealer_host
        self.dealer_port = dealer_port
        self.events_path = events_path
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self.condition = threading.Condition()
        self.current: DisplayState | None = None
        self.pending_action: str | None = None
        self.state_nonce = 0
        self.status = "connecting"
        self.error: str | None = None
        self.cumulative_winnings = 0
        self.last_result: dict[str, object] | None = None
        self.hand_history: deque[dict[str, object]] = deque(maxlen=20)
        self.finished_hands: set[int] = set()
        self.started_at = utc_now()
        self.worker = threading.Thread(target=self._run, name="acpc-human-bridge", daemon=True)

    def start(self) -> None:
        self.worker.start()

    def _event(self, payload: dict[str, object]) -> None:
        safe = {"timestamp": utc_now(), **payload}
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(safe, sort_keys=True) + "\n")
            stream.flush()

    def _connect(self) -> socket.socket:
        while True:
            try:
                connection = socket.create_connection((self.dealer_host, self.dealer_port), timeout=10)
                connection.settimeout(None)
                connection.sendall(b"VERSION:2.0.0\r\n")
                return connection
            except OSError as error:
                with self.condition:
                    self.status = "waiting_for_dealer"
                    self.error = str(error)
                time.sleep(2)

    def _run(self) -> None:
        try:
            connection = self._connect()
            reader = connection.makefile("rb")
            with self.condition:
                self.status = "waiting_for_bot"
                self.error = None
            while True:
                line = reader.readline()
                if not line:
                    with self.condition:
                        self.status = "match_complete"
                        self.condition.notify_all()
                    return
                message = line.decode("utf-8", errors="replace").strip()
                if not message.startswith("MATCHSTATE:"):
                    continue
                state = parse_matchstate(message)
                with self.condition:
                    self.current = state
                    self.state_nonce += 1
                    self.pending_action = None
                    if state.terminal:
                        self._finish_hand(state)
                        self.status = "hand_complete"
                    elif state.acting_player == state.hero_player:
                        self.status = "your_turn"
                    else:
                        self.status = "bot_thinking"
                    self.condition.notify_all()

                    while self.status == "your_turn" and self.pending_action is None:
                        self.condition.wait()
                    action = self.pending_action

                if action is not None:
                    connection.sendall(f"{state.raw}:{action}\r\n".encode("utf-8"))
                    self._event(
                        {
                            "event": "hero_action",
                            "hand_number": state.hand_number,
                            "street": STREET_NAMES[state.current_street],
                            "board": state.board,
                            "pot": state.bet1 + state.bet2,
                            "action": action,
                        }
                    )
        except Exception as error:  # keep the HTTP server alive for diagnosis
            with self.condition:
                self.error = f"{type(error).__name__}: {error}"
                self.status = "error"
                self.condition.notify_all()
            self._event({"event": "bridge_error", "error": self.error})

    def _finish_hand(self, state: DisplayState) -> None:
        if state.hand_number in self.finished_hands:
            return
        self.finished_hands.add(state.hand_number)
        try:
            winnings = hero_winnings(state)
        except Exception:
            winnings = 0
        self.cumulative_winnings += winnings
        result = {
            "hand_number": state.hand_number,
            "winnings": winnings,
            "cumulative_winnings": self.cumulative_winnings,
            "board": split_cards(state.board),
            "hero_hand": split_cards(state.hero_hand),
            "opponent_hand": split_cards(state.opponent_hand),
        }
        self.last_result = result
        self.hand_history.appendleft(result)
        self._event({"event": "hand_result", **result})

    def available_actions(self, state: DisplayState) -> list[dict[str, object]]:
        if state.terminal or state.acting_player != state.hero_player:
            return []
        facing = state.opponent_bet > state.hero_bet
        actions = []
        if facing:
            actions.append({"id": "fold", "label": "Fold"})
            actions.append(
                {
                    "id": "call",
                    "label": f"Call {state.opponent_bet - state.hero_bet}",
                    "amount": state.opponent_bet - state.hero_bet,
                }
            )
        else:
            actions.append({"id": "call", "label": "Check", "amount": 0})
        if state.opponent_bet < STACK:
            pot_raise = min(STACK, state.opponent_bet * 3)
            if pot_raise > state.opponent_bet and pot_raise < STACK:
                actions.append({"id": "pot", "label": f"Pot · {pot_raise}", "raise_to": pot_raise})
            actions.append({"id": "all_in", "label": "All-in", "raise_to": STACK})
        return actions

    def queue_action(self, action_id: str, nonce: int) -> tuple[bool, str]:
        with self.condition:
            state = self.current
            if state is None or self.status != "your_turn":
                return False, "it is not your turn"
            if nonce != self.state_nonce:
                return False, "stale state; refresh before acting"
            available = {item["id"]: item for item in self.available_actions(state)}
            if action_id not in available:
                return False, "action is not legal in this state"
            if action_id == "fold":
                protocol_action = "f"
            elif action_id == "call":
                protocol_action = "c"
            else:
                protocol_action = f"r{int(available[action_id]['raise_to'])}"
            self.pending_action = protocol_action
            self.status = "bot_thinking"
            self.condition.notify_all()
            return True, protocol_action

    def snapshot(self) -> dict[str, object]:
        with self.condition:
            state = self.current
            base = {
                "status": self.status,
                "error": self.error,
                "started_at": self.started_at,
                "state_nonce": self.state_nonce,
                "stack": STACK,
                "small_blind": SMALL_BLIND,
                "big_blind": BIG_BLIND,
                "cumulative_winnings": self.cumulative_winnings,
                "hands_completed": len(self.finished_hands),
                "last_result": self.last_result,
                "hand_history": list(self.hand_history),
            }
            if state is None:
                return base
            base.update(
                {
                    "hand_number": state.hand_number,
                    "street": STREET_NAMES[state.current_street],
                    "board": split_cards(state.board),
                    "hero_hand": split_cards(state.hero_hand),
                    "opponent_hand": split_cards(state.opponent_hand) if state.terminal else [],
                    "hero_position": "SB" if state.hero_player == 0 else "BB",
                    "hero_bet": state.hero_bet,
                    "opponent_bet": state.opponent_bet,
                    "hero_stack": STACK - state.hero_bet,
                    "opponent_stack": STACK - state.opponent_bet,
                    "pot": state.bet1 + state.bet2,
                    "actions_raw": state.actions_raw,
                    "available_actions": self.available_actions(state),
                }
            )
            return base


HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DyypHoldem · Live RTX 4090</title>
<style>
:root{color-scheme:dark;--bg:#07110e;--panel:#0e1d18;--line:#214436;--felt:#0b5b3c;--gold:#f0c96a;--text:#f4f4ec;--muted:#91aa9f;--red:#ed6a67}*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 50% 0,#163328 0,#07110e 52%);color:var(--text);font:15px/1.45 Inter,ui-sans-serif,system-ui,sans-serif}.wrap{max-width:1180px;margin:auto;padding:24px}.top{display:flex;justify-content:space-between;gap:20px;align-items:center;margin-bottom:18px}.brand{font-size:22px;font-weight:750;letter-spacing:.02em}.badge{border:1px solid var(--line);border-radius:999px;padding:7px 12px;color:var(--muted);background:#0a1713}.layout{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(300px,.8fr);gap:18px}.panel{background:rgba(14,29,24,.92);border:1px solid var(--line);border-radius:18px;box-shadow:0 20px 70px #0007}.table-panel{padding:18px}.felt{min-height:480px;border-radius:46% / 28%;background:radial-gradient(circle,#107a51,#074c32 72%);border:10px solid #37291c;box-shadow:inset 0 0 80px #001d13,0 12px 35px #0008;display:grid;grid-template-rows:1fr auto 1fr;align-items:center;padding:32px}.seat{text-align:center}.seat .name{font-weight:700}.stack{color:var(--gold)}.center{text-align:center}.pot{font-size:22px;font-weight:800;color:var(--gold);margin:12px}.cards{display:flex;justify-content:center;gap:8px;min-height:70px}.card{width:48px;height:68px;border-radius:8px;background:#f8f5e9;color:#111;display:grid;place-items:center;font-weight:850;font-size:20px;box-shadow:0 4px 10px #0006}.card.red{color:#b21e2b}.card.back{background:repeating-linear-gradient(45deg,#172f74,#172f74 6px,#e8e5d9 6px,#e8e5d9 9px);border:4px solid #eee}.actions{display:flex;flex-wrap:wrap;gap:10px;justify-content:center;margin-top:18px}.actions button{border:1px solid #476d5e;background:#173c30;color:#fff;padding:12px 18px;border-radius:12px;font-weight:700;cursor:pointer}.actions button:hover{border-color:var(--gold);transform:translateY(-1px)}.actions button.danger{background:#522522;border-color:#8b413c}.actions button:disabled{opacity:.4;cursor:not-allowed;transform:none}.side{padding:18px;display:flex;flex-direction:column;gap:18px}.section h2{font-size:13px;text-transform:uppercase;letter-spacing:.12em;color:var(--muted);margin:0 0 10px}.status{font-size:20px;font-weight:760}.thinking{color:var(--gold)}.error{color:var(--red)}table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:right;padding:7px 4px;border-bottom:1px solid #1a352b}th:first-child,td:first-child{text-align:left;color:var(--muted)}.recent{font-size:13px;color:var(--muted);max-height:180px;overflow:auto}.recent div{padding:7px 0;border-bottom:1px solid #1a352b}.foot{color:var(--muted);font-size:12px;margin-top:14px;text-align:center}@media(max-width:850px){.layout{grid-template-columns:1fr}.felt{min-height:430px;padding:22px}.wrap{padding:12px}.card{width:42px;height:60px}}
</style></head><body><div class="wrap">
<div class="top"><div><div class="brand">DyypHoldem <span style="color:var(--gold)">Live</span></div><div style="color:var(--muted)">Continual resolving · 1,000 CFR iterations · recovered value networks</div></div><div class="badge" id="gpu">RTX 4090 · connecting</div></div>
<div class="layout"><section class="panel table-panel"><div class="felt">
<div class="seat"><div class="name">DyypHoldem</div><div class="stack" id="oppStack">20,000</div><div class="cards" id="opponentCards"><div class="card back"></div><div class="card back"></div></div></div>
<div class="center"><div class="cards" id="board"></div><div class="pot" id="pot">Pot 150</div><div id="street" style="text-transform:uppercase;letter-spacing:.15em;color:#b6d9ca">Preflop</div></div>
<div class="seat"><div class="cards" id="heroCards"></div><div class="name">You · <span id="position">—</span></div><div class="stack" id="heroStack">20,000</div></div></div>
<div class="actions" id="actions"></div><div class="foot">Raises use the repository’s abstraction: pot-size or all-in. Actions return immediately; long solves run asynchronously.</div></section>
<aside class="panel side"><div class="section"><h2>Game status</h2><div class="status" id="status">Connecting to dealer…</div><div id="result" style="color:var(--muted);margin-top:6px"></div></div>
<div class="section"><h2>Calculation time by street</h2><table><thead><tr><th>Street</th><th>n</th><th>p50</th><th>p95</th><th>max</th></tr></thead><tbody id="timings"></tbody></table></div>
<div class="section"><h2>Recent bot decisions</h2><div class="recent" id="recent">No decisions yet.</div></div>
<div class="section"><h2>Session</h2><div style="color:var(--muted)">Hands completed: <span id="hands">0</span><br>Net: <span id="net">0</span> chips<br>Root precompute: <span id="rootTime">—</span><br>CFR: <span id="cfr">1,000 / skip 500</span></div></div></aside></div></div>
<script>
const params=new URLSearchParams(location.search);const supplied=params.get('token');if(supplied){localStorage.setItem('dyy-token',supplied);history.replaceState({},'',location.pathname)}const token=localStorage.getItem('dyy-token')||'';let current=null;
const api=async(path,opts={})=>{opts.headers={...(opts.headers||{}),'X-Session-Token':token};const r=await fetch(path,opts);if(!r.ok)throw new Error(await r.text());return r.json()};
const cardHtml=c=>{const red=c&&('dh'.includes(c[1]));return `<div class="card ${red?'red':''}">${c||''}</div>`};const fmt=n=>Number(n||0).toLocaleString();
function renderState(s){current=s;document.querySelector('#pot').textContent=`Pot ${fmt(s.pot||0)}`;document.querySelector('#street').textContent=s.street||'Waiting';document.querySelector('#heroCards').innerHTML=(s.hero_hand||[]).map(cardHtml).join('');const opponent=s.opponent_hand||[];document.querySelector('#opponentCards').innerHTML=opponent.length?opponent.map(cardHtml).join(''):'<div class="card back"></div><div class="card back"></div>';document.querySelector('#heroStack').textContent=fmt(s.hero_stack??20000);document.querySelector('#oppStack').textContent=fmt(s.opponent_stack??20000);document.querySelector('#position').textContent=s.hero_position||'—';document.querySelector('#board').innerHTML=(s.board||[]).map(cardHtml).join('');const labels={connecting:'Connecting…',waiting_for_dealer:'Waiting for dealer…',waiting_for_bot:'AI loading models and solving root…',your_turn:'Your turn',bot_thinking:'DyypHoldem is calculating…',hand_complete:'Hand complete',match_complete:'Match complete',error:'Session error'};const st=document.querySelector('#status');st.textContent=labels[s.status]||s.status;st.className='status '+(s.status==='bot_thinking'||s.status==='waiting_for_bot'?'thinking':'')+(s.status==='error'?' error':'');document.querySelector('#result').textContent=s.last_result?`Last hand: ${s.last_result.winnings>=0?'+':''}${fmt(s.last_result.winnings)} chips`:'';document.querySelector('#hands').textContent=s.hands_completed||0;document.querySelector('#net').textContent=(s.cumulative_winnings>=0?'+':'')+fmt(s.cumulative_winnings);const box=document.querySelector('#actions');box.innerHTML='';(s.available_actions||[]).forEach(a=>{const b=document.createElement('button');b.textContent=a.label;b.className=a.id==='fold'?'danger':'';b.onclick=()=>act(a.id,b);box.appendChild(b)})}
async function act(id,button){document.querySelectorAll('#actions button').forEach(b=>b.disabled=true);try{await api('/api/action',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:id,state_nonce:current.state_nonce})})}catch(e){alert(e.message)}await poll()}
function renderReport(r){const rows=['preflop','flop','turn','river'].map(st=>{const d=r.by_street?.[st]||{decisions:0,timing_seconds:{total_response:{p50:0,p95:0,max:0}}};const t=d.timing_seconds.total_response;return `<tr><td>${st}</td><td>${d.decisions}</td><td>${t.p50.toFixed(2)}s</td><td>${t.p95.toFixed(2)}s</td><td>${t.max.toFixed(2)}s</td></tr>`});document.querySelector('#timings').innerHTML=rows.join('');const m=r.metadata||{};if(m.gpu_name)document.querySelector('#gpu').textContent=m.gpu_name;document.querySelector('#rootTime').textContent=r.initialization?`${Number(r.initialization.seconds).toFixed(2)}s`:'—';document.querySelector('#cfr').textContent=`${fmt(m.cfr_iterations||1000)} / skip ${fmt(m.cfr_skip_iterations||500)}`;const recent=r.recent_decisions||[];document.querySelector('#recent').innerHTML=recent.length?recent.slice().reverse().map(x=>`<div><b>${x.street}</b> · ${x.chosen_action} · ${Number(x.total_response_seconds).toFixed(3)}s total · ${Number(x.cfr_seconds).toFixed(3)}s CFR</div>`).join(''):'No decisions yet.'}
async function poll(){if(!token){document.querySelector('#status').textContent='Missing session token';document.querySelector('#status').className='status error';return}try{const [s,r]=await Promise.all([api('/api/state'),api('/api/report')]);renderState(s);renderReport(r)}catch(e){document.querySelector('#status').textContent='Reconnecting…'}}poll();setInterval(poll,1200);
</script></body></html>'''


class RateLimiter:
    def __init__(self):
        self.lock = threading.Lock()
        self.requests: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str, limit: int, window: float) -> bool:
        now = time.monotonic()
        with self.lock:
            values = self.requests[key]
            while values and values[0] < now - window:
                values.popleft()
            if len(values) >= limit:
                return False
            values.append(now)
            return True


class PokerHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, bridge: HumanBridge, token: str, report_path: Path):
        super().__init__(address, PokerRequestHandler)
        self.bridge = bridge
        self.session_token = token
        self.report_path = report_path
        self.rate_limiter = RateLimiter()


class PokerRequestHandler(BaseHTTPRequestHandler):
    server: PokerHTTPServer

    def log_message(self, _format, *_args):
        return

    def _authorized(self, query: dict[str, list[str]]) -> bool:
        supplied = self.headers.get("X-Session-Token") or (query.get("token") or [""])[0]
        return bool(supplied) and secrets.compare_digest(supplied, self.server.session_token)

    def _send_json(self, payload: dict[str, object], status: int = 200):
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _reject(self, status: int, message: str):
        self._send_json({"error": message}, status)

    def do_GET(self):
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        client = self.client_address[0]
        if not self.server.rate_limiter.allow(f"get:{client}", 180, 60):
            return self._reject(HTTPStatus.TOO_MANY_REQUESTS, "rate limit exceeded")
        if parsed.path == "/healthz":
            return self._send_json({"ok": True})
        if not self._authorized(query):
            return self._reject(HTTPStatus.UNAUTHORIZED, "invalid or missing session token")
        if parsed.path == "/":
            body = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'self'")
            self.send_header("X-Frame-Options", "DENY")
            self.send_header("Referrer-Policy", "no-referrer")
            self.end_headers()
            return self.wfile.write(body)
        if parsed.path == "/api/state":
            return self._send_json(self.server.bridge.snapshot())
        if parsed.path == "/api/report":
            try:
                report = json.loads(self.server.report_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                report = {"decision_count": 0, "metadata": {}, "initialization": None, "by_street": {}, "recent_decisions": []}
            return self._send_json(report)
        return self._reject(HTTPStatus.NOT_FOUND, "not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        client = self.client_address[0]
        if not self.server.rate_limiter.allow(f"post:{client}", 20, 30):
            return self._reject(HTTPStatus.TOO_MANY_REQUESTS, "action rate limit exceeded")
        if not self._authorized(parse_qs(parsed.query)):
            return self._reject(HTTPStatus.UNAUTHORIZED, "invalid or missing session token")
        if parsed.path != "/api/action":
            return self._reject(HTTPStatus.NOT_FOUND, "not found")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 4096:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            action = str(payload["action"])
            nonce = int(payload["state_nonce"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return self._reject(HTTPStatus.BAD_REQUEST, "invalid action request")
        accepted, detail = self.server.bridge.queue_action(action, nonce)
        if not accepted:
            return self._reject(HTTPStatus.CONFLICT, detail)
        return self._send_json({"accepted": True})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dealer-host", default="127.0.0.1")
    parser.add_argument("--dealer-port", type=int, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    token = args.token_file.read_text(encoding="utf-8").strip()
    if len(token) < 24:
        raise SystemExit("session token is missing or too short")
    bridge = HumanBridge(args.dealer_host, args.dealer_port, args.events)
    bridge.start()
    server = PokerHTTPServer((args.host, args.port), bridge, token, args.report)
    server.serve_forever()


if __name__ == "__main__":
    main()
