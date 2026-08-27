import time

import torch

import settings.arguments as arguments

from terminal_equity.terminal_equity import TerminalEquity
from tree.tree_builder import PokerTreeBuilder
from tree.tree_node import TreeNode, BuildTreeParams
from lookahead.lookahead import Lookahead
from lookahead.resolve_results import ResolveResults
import game.card_tools as card_tools


class Resolving(object):

    tree_builder: PokerTreeBuilder
    terminal_equity: TerminalEquity
    player_range: arguments.Tensor
    opponent_range: arguments.Tensor
    opponent_cfvs: object
    lookahead_tree: TreeNode
    lookahead: Lookahead
    resolve_results: ResolveResults

    @staticmethod
    def _synchronize():
        if arguments.use_gpu and torch.cuda.is_available():
            torch.cuda.synchronize()

    @classmethod
    def _started(cls):
        cls._synchronize()
        return time.perf_counter()

    @classmethod
    def _elapsed(cls, started):
        cls._synchronize()
        return time.perf_counter() - started

    def __init__(self, terminal_equity):
        self.tree_builder = PokerTreeBuilder()
        self.terminal_equity = terminal_equity

    # --- Builds a depth-limited public tree rooted at a given game node.
    # -- @param node the root of the tree
    # -- @local
    def _create_lookahead_tree(self, node):
        build_tree_params = BuildTreeParams(root_node=node, limit_to_street=True)
        self.lookahead_tree = self.tree_builder.build_tree(build_tree_params)

    def _bucketing_cache_telemetry(self):
        transition_box = getattr(self.lookahead, "next_street_boxes", None)
        cache_hit = getattr(transition_box, "bucketing_cache_hit", None)
        return {
            "bucketing_cache_hit": cache_hit,
            "bucketing_transform_bytes": int(
                getattr(transition_box, "bucketing_transform_bytes", 0)
            ),
        }

    def _cuda_graph_telemetry(self):
        return self.lookahead.get_cuda_graph_telemetry()

    # -- Re-solves a depth-limited lookahead using input ranges.
    # --
    # -- Uses the input range for the opponent instead of a gadget range, so only
    # -- appropriate for re-solving the root node of the game tree (where ranges
    # -- are fixed).
    # --
    # -- @param node the public node at which to re-solve
    # -- @param player_range a range vector for the re-solving player
    # -- @param opponent_range a range vector for the opponent
    def resolve_first_node(self, node, player_range, opponent_range) -> ResolveResults:
        # Get current street name for logging
        street_name = self._get_street_name(node.board)
        arguments.logger.debug(f"Resolving first node ({street_name}) with {arguments.cfr_iters} iterations")

        self.player_range = player_range
        self.opponent_range = opponent_range
        self.opponent_cfvs = None

        total_started = self._started()
        started = self._started()
        self._create_lookahead_tree(node)
        public_tree_seconds = self._elapsed(started)

        if player_range.dim() == 1:
            player_range = player_range.view(1, player_range.size(0))
            opponent_range = opponent_range.view(1, opponent_range.size(0))

        started = self._started()
        self.lookahead = Lookahead(self.terminal_equity, player_range.size(0))
        lookahead_tensor_seconds = self._elapsed(started)

        arguments.timer.split_start("Building lookahead tree", log_level="TRACE")
        started = self._started()
        self.lookahead.build_lookahead(self.lookahead_tree)
        lookahead_build_seconds = self._elapsed(started)
        arguments.timer.split_stop("Lookahead tree build time", log_level="TIMING")

        arguments.timer.split_start(f"Resolving {street_name} tree", log_level="TRACE")
        started = self._started()
        self.lookahead.resolve_first_node(player_range, opponent_range)
        cfr_seconds = self._elapsed(started)
        arguments.timer.split_stop(f"{street_name} tree resolution time", log_level="TIMING")

        started = self._started()
        self.resolve_results = self.lookahead.get_results()
        results_seconds = self._elapsed(started)
        self.last_timing = {
            "public_tree_seconds": public_tree_seconds,
            "lookahead_tensor_seconds": lookahead_tensor_seconds,
            "lookahead_build_seconds": lookahead_build_seconds,
            "cfr_seconds": cfr_seconds,
            "results_seconds": results_seconds,
            "resolve_total_seconds": self._elapsed(total_started),
            **self._bucketing_cache_telemetry(),
            **self._cuda_graph_telemetry(),
        }

        return self.resolve_results

    # -- Re-solves a depth-limited lookahead using an input range for the player and
    # -- the @{cfrd_gadget|CFRDGadget} to generate ranges for the opponent.
    # --
    # -- @param node the public node at which to re-solve
    # -- @param player_range a range vector for the re-solving player
    # -- @param opponent_cfvs a vector of cfvs achieved by the opponent
    # -- before re-solving
    def resolve(self, node, player_range, opponent_cfvs):
        assert card_tools.is_valid_range(player_range, node.board)
        
        # Get current street name for logging
        street_name = self._get_street_name(node.board)
        arguments.logger.debug(f"Resolving node ({street_name})")

        self.player_range = player_range
        self.opponent_cfvs = opponent_cfvs
        total_started = self._started()
        started = self._started()
        self._create_lookahead_tree(node)
        public_tree_seconds = self._elapsed(started)

        if player_range.dim() == 1:
            player_range = player_range.view(1, player_range.size(0))

        arguments.timer.split_start("Building lookahead tree", log_level="TRACE")
        started = self._started()
        self.lookahead = Lookahead(self.terminal_equity, player_range.size(0))
        lookahead_tensor_seconds = self._elapsed(started)
        started = self._started()
        self.lookahead.build_lookahead(self.lookahead_tree)
        lookahead_build_seconds = self._elapsed(started)
        arguments.timer.split_stop("Tree build time", log_level="TIMING")

        arguments.timer.split_start(f"Resolving {street_name} node", log_level="TRACE")
        started = self._started()
        self.lookahead.resolve(player_range, opponent_cfvs)
        cfr_seconds = self._elapsed(started)
        arguments.timer.split_stop(f"{street_name} resolve time", log_level="TIMING")

        started = self._started()
        self.resolve_results = self.lookahead.get_results()
        results_seconds = self._elapsed(started)
        self.last_timing = {
            "public_tree_seconds": public_tree_seconds,
            "lookahead_tensor_seconds": lookahead_tensor_seconds,
            "lookahead_build_seconds": lookahead_build_seconds,
            "cfr_seconds": cfr_seconds,
            "results_seconds": results_seconds,
            "resolve_total_seconds": self._elapsed(total_started),
            **self._bucketing_cache_telemetry(),
            **self._cuda_graph_telemetry(),
        }
        return self.resolve_results

    # --- Gives a list of possible actions at the node being re-solved.
    # --
    # -- The node must first be re-solved with @{resolve} or @{resolve_first_node}.
    # -- @return a list of legal actions
    def get_possible_actions(self):
        return self.lookahead_tree.actions

    # --- Gives the average counterfactual values that the re-solve player received
    # -- at the node during re-solving.
    # --
    # -- The node must first be re-solved with @{resolve_first_node}.
    # --
    # -- @return a vector of cfvs
    def get_root_cfv(self):
        return self.resolve_results.root_cfvs

    # --- Gives the average counterfactual values that each player received
    # -- at the node during re-solving.
    # --
    # -- Useful for data generation for neural net training
    # --
    # -- The node must first be re-solved with @{resolve_first_node}.
    # --
    # -- @return a 2xK tensor of cfvs, where K is the range size
    def get_root_cfv_both_players(self):
        return self.resolve_results.root_cfvs_both_players

    # --- Gives the average counterfactual values that the opponent received
    # -- during re-solving after the re-solve player took a given action.
    # --
    # -- Used during continual re-solving to track opponent cfvs. The node must
    # -- first be re-solved with @{resolve} or @{resolve_first_node}.
    # --
    # -- @param action the action taken by the re-solve player at the node being
    # -- re-solved
    # -- @return a vector of cfvs
    def get_action_cfv(self, action):
        action_id = self._action_to_action_id(action)
        return self.resolve_results.children_cfvs[action_id]

    # --- Gives the average counterfactual values that the opponent received
    # -- during re-solving after a chance event (the betting round changes and
    # -- more cards are dealt).
    # --
    # -- Used during continual re-solving to track opponent cfvs. The node must
    # -- first be re-solved with @{resolve} or @{resolve_first_node}.
    # --
    # -- @param action the action taken by the re-solve player at the node being
    # -- re-solved
    # -- @param board a vector of board cards which were updated by the chance event
    # -- @return a vector of cfvs
    def get_chance_action_cfv(self, action, board):
        started = self._started()
        replayed_flop = False
        captured_flop = False
        # resolve to get next_board chance actions if flop
        if board.dim() == 1 and board.size(0) == 3:
            captured_flop = (
                self.lookahead.has_captured_preflop_inputs(action) is True
            )
            if not captured_flop:
                replayed_flop = True
                # The terminal-equity object is shared across hands and streets.
                # A cached-root decision does not otherwise reset it, so a previous
                # hand's board can leak into this preflop replay. Restore the board
                # at which this lookahead was originally solved before recomputing
                # its CFR trajectory for the observed flop.
                self.terminal_equity.set_board(self.lookahead.tree.board)
                self.lookahead.reset()
                board_idx = card_tools.get_flop_board_index(board)
                self.lookahead.next_board_idx = board_idx

                if self.opponent_cfvs is not None:
                    self.lookahead.resolve(self.player_range, self.opponent_cfvs)
                else:
                    self.lookahead.resolve_first_node(self.player_range, self.opponent_range)
                self.lookahead.next_board_idx = None
        out = self.lookahead.get_chance_action_cfv(action, board)
        self.last_chance_timing = {
            "seconds": self._elapsed(started),
            "replayed_flop": replayed_flop,
            "captured_flop": captured_flop,
        }
        return out

    # --- Gives the probability that the re-solved strategy takes a given action.
    # --
    # -- The node must first be re-solved with @{resolve} or @{resolve_first_node}.
    # --
    # -- @param action a legal action at the re-solve node
    # -- @return a vector giving the probability of taking the action with each
    # -- private hand
    def get_action_strategy(self, action):
        action_id = self._action_to_action_id(action)
        return self.resolve_results.strategy[action_id][0]

    # --- Gives the index of the given action at the node being re-solved.
    # --
    # -- The node must first be re-solved with @{resolve} or @{resolve_first_node}.
    # -- @param action a legal action at the node
    # -- @return the index of the action
    # -- @local
    def _action_to_action_id(self, action):
        actions = self.get_possible_actions()
        action_id = -1
        for i in range(actions.size(0)):
            if action == actions[i]:
                action_id = i
        assert action_id != -1
        return action_id

    # --- Helper method to get the current street name based on board cards
    # -- @param board the current board cards
    # -- @return a string representing the current street (preflop, flop, turn, river)
    # -- @local
    def _get_street_name(self, board):
        if board is None or board.size(0) == 0:
            return "preflop"
        elif board.size(0) == 3:
            return "flop"
        elif board.size(0) == 4:
            return "turn"
        elif board.size(0) == 5:
            return "river"
        else:
            return "unknown"
