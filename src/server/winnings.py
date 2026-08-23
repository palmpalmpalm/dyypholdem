"""Small payout helpers shared by ACPC clients and lightweight tests."""


def showdown_winnings(my_strength, opponent_strength, my_bet, opponent_bet):
    """Return net chips for an evaluator where lower strength is better."""
    if my_strength == opponent_strength:
        return 0
    return opponent_bet if my_strength < opponent_strength else -my_bet
