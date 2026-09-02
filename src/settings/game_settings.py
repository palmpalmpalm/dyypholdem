from math import comb
import os

# definition of a deck of poker cards
suit_count = 4
rank_count = 13
card_count = suit_count * rank_count

# names for suit and rank of poker card
suit_table = ['c', 'd', 'h', 's']
rank_table = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

# hole cards per hand
hand_card_count = 2
hand_count = comb(card_count, hand_card_count)

# board structure for texas hold'em
board_card_count = [0, 3, 4, 5]

# the blind structure, in chips
ante = 100
small_blind = 50
big_blind = 100
# the size of each player's stack, in chips
stack = 20000
# list of pot-scaled bet sizes to use in tree
# orig: bet_sizing = [[1], [1], [1]]
bet_sizing = [[1], [1], [1]]


def _parse_opponent_bet_sizing(raw):
    """Parse DYYPHOLDEM_OPPONENT_BET_SIZING into a per-level fraction table.

    The value is a comma-separated, ascending list of pot fractions the
    opponent may use for the first bet of a street; the opponent's raises and
    every action of the re-solving player keep the default pot-only menu.
    An empty or unset value keeps the original tree for both players.
    """
    text = (raw or "").strip()
    if not text:
        return None
    fractions = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError as error:
            raise RuntimeError(f"DYYPHOLDEM_OPPONENT_BET_SIZING has a non-numeric pot fraction: {item!r}") from error
        if not value > 0:
            raise RuntimeError("DYYPHOLDEM_OPPONENT_BET_SIZING fractions must be positive")
        fractions.append(value)
    if not fractions:
        raise RuntimeError("DYYPHOLDEM_OPPONENT_BET_SIZING has no pot fractions")
    if fractions != sorted(fractions) or len(set(fractions)) != len(fractions):
        raise RuntimeError("DYYPHOLDEM_OPPONENT_BET_SIZING must be strictly ascending")
    return [fractions, [1], [1]]


# Optional richer bet menu for the opponent only: the player who is not acting
# at the lookahead root. None means the opponent uses ``bet_sizing``.
opponent_bet_sizing = _parse_opponent_bet_sizing(os.environ.get("DYYPHOLDEM_OPPONENT_BET_SIZING"))

