#!/usr/bin/env python3
"""PokerKit-backed legality for DyypHoldem's live ACPC browser seat.

The ACPC dealer remains authoritative.  This module reconstructs only the
public betting state in PokerKit so the browser can display and submit exact
no-limit raise sizes.  PokerKit is imported lazily so the legacy DyypHoldem
test/runtime environment can still import the web bridge without installing a
modern Python-only dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any, Optional


POKERKIT_VERSION = "0.7.5"


class PokerKitUnavailable(RuntimeError):
    """Raised when the pinned PokerKit runtime cannot be loaded."""


class PokerKitStateError(ValueError):
    """Raised when an ACPC public state cannot be mirrored safely."""


@dataclass(frozen=True)
class LegalActionState:
    """Immutable, frontend-safe legality snapshot for one state nonce."""

    can_fold: bool = False
    can_check: bool = False
    can_call: bool = False
    can_raise: bool = False
    call_amount: int = 0
    min_raise_to: Optional[int] = None
    nominal_min_raise_to: Optional[int] = None
    half_pot_raise_to: Optional[int] = None
    three_quarter_pot_raise_to: Optional[int] = None
    pot_raise_to: Optional[int] = None
    max_raise_to: Optional[int] = None
    all_in_only: bool = False

    @property
    def available(self) -> bool:
        return True

    def allows_raise_to(self, amount: object) -> bool:
        """Return whether an integer cumulative ACPC raise-to is legal."""

        if type(amount) is not int or not self.can_raise:
            return False
        if self.min_raise_to is None or self.max_raise_to is None:
            return False
        if self.all_in_only:
            return amount == self.max_raise_to
        return self.min_raise_to <= amount <= self.max_raise_to

    def as_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "can_fold": self.can_fold,
            "can_check": self.can_check,
            "can_call": self.can_call,
            "can_raise": self.can_raise,
            "call_amount": self.call_amount,
            "min_raise_to": self.min_raise_to,
            "nominal_min_raise_to": self.nominal_min_raise_to,
            "half_pot_raise_to": self.half_pot_raise_to,
            "three_quarter_pot_raise_to": self.three_quarter_pot_raise_to,
            "pot_raise_to": self.pot_raise_to,
            "max_raise_to": self.max_raise_to,
            "all_in_only": self.all_in_only,
        }


def unavailable_legal_action_state() -> dict[str, object]:
    """Return the fail-closed public schema when validation is unavailable."""

    payload = LegalActionState().as_dict()
    payload["available"] = False
    return payload


class PokerKitAcpcAdapter:
    """Replay ACPC public actions in PokerKit and expose no-limit legality.

    DyypHoldem's ACPC indices are ``0=SB`` and ``1=BB``.  PokerKit's heads-up
    blind assignment is ``0=BB`` and ``1=SB``, hence every player index is
    mapped with ``1 - index``.  ACPC raise amounts are cumulative commitments
    across the hand, while PokerKit raise-to amounts reset on each street; the
    conversion adds/removes each acting player's prior-street commitment.
    """

    def __init__(
        self,
        stack: int = 20_000,
        small_blind: int = 50,
        big_blind: int = 100,
        expected_version: str = POKERKIT_VERSION,
    ) -> None:
        try:
            installed_version = version("PokerKit")
        except PackageNotFoundError as error:
            raise PokerKitUnavailable(
                f"PokerKit=={expected_version} is required for live action validation"
            ) from error
        if installed_version != expected_version:
            raise PokerKitUnavailable(
                f"PokerKit=={expected_version} is required; found {installed_version}"
            )
        try:
            pokerkit = import_module("pokerkit")
        except Exception as error:
            raise PokerKitUnavailable(
                f"PokerKit=={expected_version} could not be imported"
            ) from error

        self.stack = int(stack)
        self.small_blind = int(small_blind)
        self.big_blind = int(big_blind)
        self._Automation = pokerkit.Automation
        self._NoLimitTexasHoldem = pokerkit.NoLimitTexasHoldem

    @staticmethod
    def _pk_player(acpc_player: int) -> int:
        if acpc_player not in (0, 1):
            raise PokerKitStateError("invalid ACPC player index")
        return 1 - acpc_player

    def _new_state(self) -> Any:
        return self._NoLimitTexasHoldem.create_state(
            tuple(self._Automation),
            False,
            0,
            (self.small_blind, self.big_blind),
            self.big_blind,
            (self.stack, self.stack),
            2,
        )

    def _prior_street_commitment(self, state: Any, acpc_player: int) -> int:
        pk_player = self._pk_player(acpc_player)
        total_commitment = self.stack - int(state.stacks[pk_player])
        return total_commitment - int(state.bets[pk_player])

    def _acpc_raise_to(self, state: Any, acpc_player: int, local_amount: Any) -> Optional[int]:
        if local_amount is None:
            return None
        return self._prior_street_commitment(state, acpc_player) + int(local_amount)

    @staticmethod
    def _action_int(action: dict[str, object], key: str) -> int:
        value = action.get(key)
        if type(value) is not int:
            raise PokerKitStateError(f"public action {key} must be an integer")
        return value

    def replay(self, public_state: Any) -> Any:
        """Rebuild a PokerKit state from a parsed ACPC public action state."""

        try:
            actions = list(public_state.all_actions)
        except (AttributeError, TypeError) as error:
            raise PokerKitStateError("public action state is missing actions") from error
        if len(actions) < 2:
            raise PokerKitStateError("public action state is missing blinds")

        expected_blinds = (
            (0, self.small_blind),
            (1, self.big_blind),
        )
        for action, (player, amount) in zip(actions[:2], expected_blinds):
            if (
                action.get("kind") != "raise"
                or action.get("player") != player
                or action.get("raise_to") != amount
            ):
                raise PokerKitStateError("public action state has invalid blinds")

        state = self._new_state()
        for action in actions[2:]:
            if not state.status or state.actor_index is None:
                raise PokerKitStateError("public action state continues after hand completion")
            acpc_player = self._action_int(action, "player")
            expected_actor = self._pk_player(acpc_player)
            if int(state.actor_index) != expected_actor:
                raise PokerKitStateError("public action actor does not match PokerKit")
            action_street = self._action_int(action, "street")
            if int(state.street_index) + 1 != action_street:
                raise PokerKitStateError("public action street does not match PokerKit")

            kind = action.get("kind")
            if kind == "fold":
                if not state.can_fold():
                    raise PokerKitStateError("public fold is not legal")
                state.fold()
            elif kind == "call":
                if not state.can_check_or_call():
                    raise PokerKitStateError("public check/call is not legal")
                state.check_or_call()
            elif kind == "raise":
                cumulative_amount = self._action_int(action, "raise_to")
                local_amount = cumulative_amount - self._prior_street_commitment(
                    state, acpc_player
                )
                if not state.can_complete_bet_or_raise_to(local_amount):
                    raise PokerKitStateError("public raise is not legal")
                state.complete_bet_or_raise_to(local_amount)
            else:
                raise PokerKitStateError("unknown public action kind")

        terminal = bool(getattr(public_state, "terminal", False))
        if terminal:
            return state
        if not state.status or state.actor_index is None:
            raise PokerKitStateError("PokerKit completed a nonterminal public state")
        if int(state.street_index) + 1 != int(public_state.current_street):
            raise PokerKitStateError("current public street does not match PokerKit")

        acpc_actor = self._action_int(
            {"player": getattr(public_state, "acting_player", None)}, "player"
        )
        if int(state.actor_index) != self._pk_player(acpc_actor):
            raise PokerKitStateError("current public actor does not match PokerKit")

        public_commitments = (int(public_state.bet1), int(public_state.bet2))
        pokerkit_commitments = tuple(
            self.stack - int(state.stacks[self._pk_player(player)])
            for player in (0, 1)
        )
        if pokerkit_commitments != public_commitments:
            raise PokerKitStateError("public commitments do not match PokerKit")
        return state

    def legal_actions(self, public_state: Any) -> LegalActionState:
        """Return authoritative legal actions for a nonterminal public state."""

        if bool(getattr(public_state, "terminal", False)):
            return LegalActionState()
        state = self.replay(public_state)
        acpc_actor = int(public_state.acting_player)
        can_check_or_call = bool(state.can_check_or_call())
        call_amount = int(state.checking_or_calling_amount) if can_check_or_call else 0
        can_raise = bool(state.can_complete_bet_or_raise_to())

        min_raise_to = None
        nominal_min_raise_to = None
        half_pot_raise_to = None
        three_quarter_pot_raise_to = None
        pot_raise_to = None
        max_raise_to = None
        all_in_only = False
        if can_raise:
            effective_min_raise_to = self._acpc_raise_to(
                state,
                acpc_actor,
                state.min_completion_betting_or_raising_to_amount,
            )
            nominal_local_minimum = max(
                int(state.completion_betting_or_raising_amount),
                int(state.street.min_completion_betting_or_raising_amount),
            )
            if not state.completion_status:
                nominal_local_minimum += max(int(value) for value in state.bets)
            nominal_min_raise_to = self._acpc_raise_to(
                state,
                acpc_actor,
                nominal_local_minimum,
            )
            pot_raise_to = self._acpc_raise_to(
                state,
                acpc_actor,
                state.pot_completion_betting_or_raising_to_amount,
            )
            max_raise_to = self._acpc_raise_to(
                state,
                acpc_actor,
                state.max_completion_betting_or_raising_to_amount,
            )
            if (
                effective_min_raise_to is None
                or nominal_min_raise_to is None
                or pot_raise_to is None
                or max_raise_to is None
            ):
                raise PokerKitStateError("PokerKit returned inconsistent raise bounds")
            all_in_only = max_raise_to < nominal_min_raise_to
            min_raise_to = effective_min_raise_to
            if all_in_only and min_raise_to != max_raise_to:
                raise PokerKitStateError("PokerKit returned an invalid short all-in bound")

            prior_commitment = self._prior_street_commitment(state, acpc_actor)
            highest_bet = max(int(value) for value in state.bets)
            pot_after_call = int(state.total_pot_amount) + call_amount

            def fractional_raise_to(numerator: int, denominator: int) -> int:
                local_amount = highest_bet + numerator * pot_after_call // denominator
                cumulative_amount = prior_commitment + local_amount
                return max(min_raise_to, min(max_raise_to, cumulative_amount))

            half_pot_raise_to = fractional_raise_to(1, 2)
            three_quarter_pot_raise_to = fractional_raise_to(3, 4)
            pot_raise_to = max(min_raise_to, min(max_raise_to, pot_raise_to))

        return LegalActionState(
            can_fold=bool(state.can_fold()),
            can_check=can_check_or_call and call_amount == 0,
            can_call=can_check_or_call and call_amount > 0,
            can_raise=can_raise,
            call_amount=call_amount,
            min_raise_to=min_raise_to,
            nominal_min_raise_to=nominal_min_raise_to,
            half_pot_raise_to=half_pot_raise_to,
            three_quarter_pot_raise_to=three_quarter_pot_raise_to,
            pot_raise_to=pot_raise_to,
            max_raise_to=max_raise_to,
            all_in_only=all_in_only,
        )
