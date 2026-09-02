"""Slumbot HTTP API adapter.

Slumbot (https://www.slumbot.com) plays heads-up no-limit Texas hold'em with
20,000-chip stacks and 50/100 blinds, the same game DyypHoldem is configured
for. This module turns each Slumbot API response into the ACPC ``MATCHSTATE``
string DyypHoldem already understands, and converts DyypHoldem's cumulative
ACPC actions back into Slumbot's street-local ``b<amount>``, ``c``, ``k`` and
``f`` encoding.

Slumbot positions: ``client_pos`` 1 is the small blind and acts first before
the flop; ``client_pos`` 0 is the big blind and acts first on every later
street. The bundled ACPC ``holdem.nolimit.2p.reverse_blinds.game`` uses the
same numbering (player 0 posts the big blind), so the position is copied
verbatim into the ``MATCHSTATE``.

Slumbot bet amounts are the street-local "bet to" level (before the flop that
level starts at the big blind), while ACPC raise amounts are cumulative hand
commitments. Both directions of that conversion are derived from Slumbot's own
reference action parser, never from locally tracked commitments.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

import settings.arguments as arguments
import settings.constants as constants
import settings.game_settings as game_settings

import server.protocol_to_node as protocol_to_node

HOST = "slumbot.com"
NUM_STREETS = constants.streets_count
SMALL_BLIND = 50
BIG_BLIND = 100
STACK_SIZE = 20000
game_settings.small_blind = SMALL_BLIND
game_settings.big_blind = BIG_BLIND
game_settings.stack = STACK_SIZE

# Backwards-compatible module alias used by the original client.
host = HOST


class SlumbotError(RuntimeError):
    """Base class for Slumbot session failures."""


class SlumbotTransportError(SlumbotError):
    """The Slumbot API could not be reached after the configured retries."""


class SlumbotProtocolError(SlumbotError):
    """Slumbot rejected a request or returned a state DyypHoldem cannot act on."""


class SlumbotGame(object):
    """One Slumbot API session presented through DyypHoldem's ACPC state model."""

    def __init__(
        self,
        host: str = HOST,
        request_timeout: float = 30.0,
        max_attempts: int = 5,
        backoff_seconds: float = 2.0,
        sleep=time.sleep,
        opener=urllib.request.urlopen,
    ):
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self.host = host
        self.request_timeout = float(request_timeout)
        self.max_attempts = int(max_attempts)
        self.backoff_seconds = float(backoff_seconds)
        self._sleep = sleep
        self._opener = opener
        self.request_retries = 0
        self.last_action_string: str | None = None
        self.last_correction: str | None = None
        self.reset_hand()

    def reset_hand(self) -> None:
        self.last_response: dict | None = None
        self.acpc_actions = ""
        self.current_state: dict | None = None
        self.hand_number = 0
        self.last_action_string = None
        self.last_correction = None

    # -- transport ---------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        last_error = "unknown transport failure"
        for attempt in range(1, self.max_attempts + 1):
            request = urllib.request.Request(
                f"https://{self.host}{path}",
                data=body,
                method="POST",
                headers={"Content-Type": "application/json", "Accept": "application/json"},
            )
            try:
                with self._opener(request, timeout=self.request_timeout) as response:
                    raw = response.read().decode("utf-8")
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:500]
                if 500 <= error.code < 600 and attempt < self.max_attempts:
                    last_error = f"HTTP {error.code} from {path}"
                    self._retry_delay(attempt, last_error)
                    continue
                if 500 <= error.code < 600:
                    raise SlumbotTransportError(f"HTTP {error.code} from {path}: {detail}") from error
                raise SlumbotProtocolError(f"HTTP {error.code} from {path}: {detail}") from error
            except (urllib.error.URLError, TimeoutError, OSError) as error:
                last_error = f"{type(error).__name__}: {getattr(error, 'reason', error)}"
                if attempt < self.max_attempts:
                    self._retry_delay(attempt, last_error)
                    continue
                raise SlumbotTransportError(f"{path} failed after {attempt} attempts: {last_error}") from error

            try:
                parsed = json.loads(raw)
            except ValueError as error:
                raise SlumbotProtocolError(f"{path} returned non-JSON content") from error
            if not isinstance(parsed, dict):
                raise SlumbotProtocolError(f"{path} returned a non-object JSON payload")
            if "error_msg" in parsed:
                raise SlumbotProtocolError(f"Slumbot rejected {path}: {parsed['error_msg']}")
            return parsed
        raise SlumbotTransportError(f"{path} failed: {last_error}")

    def _retry_delay(self, attempt: int, reason: str) -> None:
        self.request_retries += 1
        delay = self.backoff_seconds * (2 ** (attempt - 1))
        arguments.logger.warning(f"Slumbot request retry {attempt}: {reason}; sleeping {delay:.1f}s")
        self._sleep(delay)

    # -- hand lifecycle ----------------------------------------------------

    def new_hand(self, token: str | None, hand_number: int = 0) -> dict:
        self.reset_hand()
        self.hand_number = int(hand_number)
        data = {}
        if token:
            data["token"] = token
        response = self._post("/api/new_hand", data)
        self.last_response = response
        return response

    def get_next_situation(self, response: dict, hand_number: int | None = None):
        """Convert a Slumbot response into DyypHoldem's state and tree node.

        Raises ``SlumbotProtocolError`` unless the reconstructed state has
        DyypHoldem to act, so an encoding mismatch can never silently drive
        the solver from the wrong seat.
        """
        if hand_number is not None:
            self.hand_number = int(hand_number)
        arguments.logger.trace(f"Message from server: {repr(response)}")
        action = response.get("action")
        if not isinstance(action, str):
            raise SlumbotProtocolError("Slumbot response has no action string")
        self.current_state = self.parse_action(action)
        if "error" in self.current_state:
            raise SlumbotProtocolError(
                f"could not parse Slumbot action {action!r}: {self.current_state['error']}"
            )
        client_pos = response.get("client_pos")
        if client_pos not in (0, 1):
            raise SlumbotProtocolError(f"Slumbot response has invalid client_pos {client_pos!r}")
        if self.current_state["pos"] != client_pos:
            raise SlumbotProtocolError(
                f"Slumbot state {action!r} has player {self.current_state['pos']} to act, "
                f"not the client seat {client_pos}"
            )

        msg = self.convert_state(response, self.current_state, self.hand_number)
        arguments.logger.info(f"New state received from server: {msg}")
        parsed_state = protocol_to_node.parse_state(msg)
        if parsed_state.acting_player != parsed_state.player:
            raise SlumbotProtocolError(
                f"reconstructed ACPC state {msg!r} does not have DyypHoldem to act"
            )
        node = protocol_to_node.parsed_state_to_node(parsed_state)
        self.last_response = response
        return parsed_state, node

    def convert_state(self, response: dict, state: dict, hand_number: int | None = None) -> str:
        prefix = "MATCHSTATE"
        position = response.get("client_pos")
        hole_cards_list = response.get("hole_cards") or []
        board_cards_list = response.get("board") or []
        hole_cards = "".join(str(card) for card in hole_cards_list)
        if position == 0:
            hole_cards += "|"
        else:
            hole_cards = "|" + hole_cards
        board_cards = ""
        for i, card in enumerate(board_cards_list):
            if i == 0 or i == 3 or i == 4:
                board_cards += "/"
            board_cards += str(card)
        self.acpc_actions, self.max_bet = self.acpcify_actions(response.get("action") or "")
        hand_id = self.hand_number if hand_number is None else int(hand_number)
        return f"{prefix}:{position}:{hand_id}:{self.acpc_actions}:{hole_cards}{board_cards}"

    # -- action encoding ---------------------------------------------------

    def encode_action(self, advised_action: protocol_to_node.Action) -> tuple[str, str | None]:
        """Return Slumbot's action string plus an optional legality correction note."""
        state = self.current_state
        if state is None or "error" in state:
            raise SlumbotProtocolError("no parsed Slumbot state is available for action encoding")

        if advised_action.action == constants.ACPCActions.fold:
            return "f", None

        if advised_action.action == constants.ACPCActions.ccall:
            return ("c" if state["last_bet_size"] > 0 else "k"), None

        if advised_action.action != constants.ACPCActions.rraise:
            raise SlumbotProtocolError(f"unsupported DyypHoldem action {advised_action.action!r}")

        raise_to = int(advised_action.raise_amount)
        street_last_bet_to = int(state["street_last_bet_to"])
        previous_streets = int(state["total_last_bet_to"]) - street_last_bet_to
        remaining = STACK_SIZE - street_last_bet_to
        if remaining <= 0:
            raise SlumbotProtocolError("DyypHoldem tried to raise with no chips behind")
        last_bet_size = int(state["last_bet_size"])
        min_bet_size = max(last_bet_size, BIG_BLIND) if last_bet_size > 0 else BIG_BLIND
        min_bet_size = min(min_bet_size, remaining)

        street_bet_to = raise_to - previous_streets
        requested_size = street_bet_to - street_last_bet_to
        correction = None
        if requested_size > remaining:
            street_bet_to = street_last_bet_to + remaining
            correction = f"raise_to_{raise_to}_capped_to_all_in"
        elif requested_size < min_bet_size:
            street_bet_to = street_last_bet_to + min_bet_size
            correction = f"raise_to_{raise_to}_lifted_to_min_raise"
        if correction is not None:
            arguments.logger.warning(
                f"Slumbot legality correction: {correction} (street bet-to {street_bet_to})"
            )
        return f"b{street_bet_to}", correction

    def play_action(self, token: str | None, advised_action: protocol_to_node.Action) -> dict:
        next_action, correction = self.encode_action(advised_action)
        self.last_action_string = next_action
        self.last_correction = correction
        arguments.logger.debug(f"Sending action to server: {next_action}")
        data = {"incr": next_action}
        if token:
            data["token"] = token
        response = self._post("/api/act", data)
        self.last_response = response
        return response

    # -- pure conversions --------------------------------------------------

    @staticmethod
    def acpcify_actions(actions: str):
        """Convert Slumbot's street-local action string to cumulative ACPC actions."""
        actions = actions.replace("b", "r")
        actions = actions.replace("k", "c")
        streets = actions.split("/")
        max_bet = 0
        for i, street_actions in enumerate(streets):
            bets = street_actions.split("r")
            max_street_bet = max_bet
            for j, betstr in enumerate(bets):
                try:
                    flag_c = False
                    flag_f = False
                    if len(betstr) > 1 and betstr[-1] == 'c':
                        flag_c = True
                        betstr = betstr.replace("c", "")
                    elif len(betstr) > 1 and betstr[-1] == 'f':
                        flag_f = True
                        betstr = betstr.replace("f", "")
                    bet = int(betstr)
                    bet += max_bet
                    max_street_bet = max(max_street_bet, bet)
                    bets[j] = str(bet)
                    if flag_c:
                        bets[j] += "c"
                    elif flag_f:
                        bets[j] += "f"
                    bets[j] = "r" + bets[j]
                except ValueError:
                    continue
            max_bet = max_street_bet
            if max_bet == 0:
                max_bet = BIG_BLIND
            good_string = "".join(bets)
            streets[i] = good_string
        return "/".join(streets), max_bet

    @staticmethod
    def parse_action(action: str) -> dict:
        """Slumbot's reference action parser.

        Returns a dict with information about the action passed in, or a dict
        with an ``error`` key if there was a problem parsing the action.
        ``pos`` is -1 if the hand is over; otherwise the position of the player
        next to act. ``street_last_bet_to`` only counts chips bet on this
        street, ``total_last_bet_to`` counts all chips put into the pot.
        Handles action with or without a final '/'; e.g., "ck" or "ck/".
        """
        st = 0
        street_last_bet_to = BIG_BLIND
        total_last_bet_to = BIG_BLIND
        last_bet_size = BIG_BLIND - SMALL_BLIND
        last_bettor = 0
        sz = len(action)
        pos = 1
        if sz == 0:
            return {
                'st': st,
                'pos': pos,
                'street_last_bet_to': street_last_bet_to,
                'total_last_bet_to': total_last_bet_to,
                'last_bet_size': last_bet_size,
                'last_bettor': last_bettor,
            }

        check_or_call_ends_street = False
        i = 0
        while i < sz:
            if st >= NUM_STREETS:
                return {'error': 'Unexpected error'}
            c = action[i]
            i += 1
            if c == 'k':
                if last_bet_size > 0:
                    return {'error': 'Illegal check'}
                if check_or_call_ends_street:
                    # After a check that ends a pre-river street, expect either a '/' or end of string.
                    if st < NUM_STREETS - 1 and i < sz:
                        if action[i] != '/':
                            return {'error': 'Missing slash'}
                        i += 1
                    if st == NUM_STREETS - 1:
                        # Reached showdown
                        pos = -1
                    else:
                        pos = 0
                        st += 1
                    street_last_bet_to = 0
                    check_or_call_ends_street = False
                else:
                    pos = (pos + 1) % 2
                    check_or_call_ends_street = True
            elif c == 'c':
                if last_bet_size == 0:
                    return {'error': 'Illegal call'}
                if total_last_bet_to == STACK_SIZE:
                    # Call of an all-in bet
                    # Either allow no slashes, or slashes terminating all streets prior to the river.
                    if i != sz:
                        for st1 in range(st, NUM_STREETS - 1):
                            if i == sz:
                                return {'error': 'Missing slash (end of string)'}
                            else:
                                c = action[i]
                                i += 1
                                if c != '/':
                                    return {'error': 'Missing slash'}
                    if i != sz:
                        return {'error': 'Extra characters at end of action'}
                    st = NUM_STREETS - 1
                    pos = -1
                    last_bet_size = 0
                    return {
                        'st': st,
                        'pos': pos,
                        'street_last_bet_to': street_last_bet_to,
                        'total_last_bet_to': total_last_bet_to,
                        'last_bet_size': last_bet_size,
                        'last_bettor': last_bettor,
                    }
                if check_or_call_ends_street:
                    # After a call that ends a pre-river street, expect either a '/' or end of string.
                    if st < NUM_STREETS - 1 and i < sz:
                        if action[i] != '/':
                            return {'error': 'Missing slash'}
                        i += 1
                    if st == NUM_STREETS - 1:
                        # Reached showdown
                        pos = -1
                    else:
                        pos = 0
                        st += 1
                    street_last_bet_to = 0
                    check_or_call_ends_street = False
                else:
                    pos = (pos + 1) % 2
                    check_or_call_ends_street = True
                last_bet_size = 0
                last_bettor = -1
            elif c == 'f':
                if last_bet_size == 0:
                    return {'error': 'Illegal fold'}
                if i != sz:
                    return {'error': 'Extra characters at end of action'}
                pos = -1
                return {
                    'st': st,
                    'pos': pos,
                    'street_last_bet_to': street_last_bet_to,
                    'total_last_bet_to': total_last_bet_to,
                    'last_bet_size': last_bet_size,
                    'last_bettor': last_bettor,
                }
            elif c == 'b':
                j = i
                while i < sz and action[i] >= '0' and action[i] <= '9':
                    i += 1
                if i == j:
                    return {'error': 'Missing bet size'}
                try:
                    new_street_last_bet_to = int(action[j:i])
                except (TypeError, ValueError):
                    return {'error': 'Bet size not an integer'}
                new_last_bet_size = new_street_last_bet_to - street_last_bet_to
                # Validate that the bet is legal
                remaining = STACK_SIZE - street_last_bet_to
                if last_bet_size > 0:
                    min_bet_size = last_bet_size
                    # Make sure minimum opening bet is the size of the big blind.
                    if min_bet_size < BIG_BLIND:
                        min_bet_size = BIG_BLIND
                else:
                    min_bet_size = BIG_BLIND
                # Can always go all-in
                if min_bet_size > remaining:
                    min_bet_size = remaining
                if new_last_bet_size < min_bet_size:
                    return {'error': f"Bet too small - remaining={remaining}, min_bet_size={min_bet_size}, new_last_bet_size={new_last_bet_size}"}
                max_bet_size = remaining
                if new_last_bet_size > max_bet_size:
                    return {'error': f"Bet too big - remaining={remaining}, max_bet_size={max_bet_size}, new_last_bet_size={new_last_bet_size}"}
                last_bet_size = new_last_bet_size
                street_last_bet_to = new_street_last_bet_to
                total_last_bet_to += last_bet_size
                last_bettor = pos
                pos = (pos + 1) % 2
                check_or_call_ends_street = True
            else:
                return {'error': 'Unexpected character in action'}

        return {
            'st': st,
            'pos': pos,
            'street_last_bet_to': street_last_bet_to,
            'total_last_bet_to': total_last_bet_to,
            'last_bet_size': last_bet_size,
            'last_bettor': last_bettor,
        }
