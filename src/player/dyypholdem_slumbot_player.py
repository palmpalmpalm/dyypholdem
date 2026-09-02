import argparse
import os
from pathlib import Path
import platform
import sys
sys.path.append(os.getcwd())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Play with DyypHoldem against Slumbot')
    parser.add_argument('hands', type=int, help="Number of hands to play against Slumbot")
    parser.add_argument("--host", type=str, default="slumbot.com", help="Slumbot API host")
    parser.add_argument("--seed", type=int, default=None, help="seed Torch and Python action sampling")
    parser.add_argument("--telemetry", type=Path, default=None, help="private decision JSONL output")
    parser.add_argument("--report", type=Path, default=None, help="safe live JSON timing report")
    parser.add_argument("--text-report", type=Path, default=None, help="safe live text timing report")
    parser.add_argument("--events", type=Path, default=None, help="safe per-hand JSONL events")
    parser.add_argument("--summary", type=Path, default=None, help="safe live JSON match summary")
    parser.add_argument("--max-consecutive-errors", type=int, default=3,
                        help="abort after this many consecutive failed hands")
    return parser


if __name__ == '__main__':
    args = build_parser().parse_args()
    if args.hands < 1:
        raise SystemExit("hands must be at least 1")

    import gc

    import torch

    import settings.arguments as arguments

    from server.slumbot_game import SlumbotGame
    from lookahead.continual_resolving import ContinualResolving
    from player.slumbot_match import SlumbotMatch
    from utils.decision_telemetry import DecisionTelemetryWriter, model_manifest

    import utils.pseudo_random as random_

    if args.seed is not None:
        if not 0 <= args.seed <= 2_147_483_647:
            raise SystemExit("seed must be between 0 and 2147483647")
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        random_.rng.seed(args.seed)

    slumbot_game = SlumbotGame(host=args.host)
    continual_resolving = ContinualResolving()

    telemetry_writer = None
    if args.telemetry is not None:
        report_path = args.report or args.telemetry.with_name("timing_report.json")
        text_report_path = args.text_report or args.telemetry.with_name("timing_report.txt")
        compact_root_raw = os.environ.get("DYYPHOLDEM_COMPACT_MODEL_PATH")
        compact_root = Path(compact_root_raw).resolve() if compact_root_raw else None
        gpu_name = torch.cuda.get_device_name(0) if arguments.use_gpu and torch.cuda.is_available() else None
        telemetry_writer = DecisionTelemetryWriter(
            args.telemetry,
            report_path,
            text_report_path,
            {
                "source_commit": os.environ.get("DYYPHOLDEM_SOURCE_COMMIT"),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "gpu_name": gpu_name,
                "cfr_iterations": arguments.cfr_iters,
                "cfr_skip_iterations": arguments.cfr_skip_iters,
                "bot_seed": args.seed,
                "opponent": "slumbot",
                "slumbot_host": args.host,
                "expected_hands": args.hands,
                "compact_models": model_manifest(compact_root),
            },
        )
        telemetry_writer.append(continual_resolving.initialization_telemetry)

    arguments.logger.success(
        f"AI_READY initialization_seconds={continual_resolving.initialization_seconds:.6f} "
        f"device={arguments.device}"
    )

    if arguments.use_pseudo_random:
        random_.manual_seed(0)

    def collect_garbage() -> None:
        gc.collect()
        if arguments.use_gpu and torch.cuda.is_available():
            torch.cuda.empty_cache()

    match = SlumbotMatch(
        slumbot_game,
        continual_resolving,
        args.hands,
        events_path=args.events,
        summary_path=args.summary,
        telemetry_writer=telemetry_writer,
        logger=arguments.logger,
        after_hand=collect_garbage,
        max_consecutive_errors=args.max_consecutive_errors,
        host=args.host,
        seed=args.seed,
    )
    sys.exit(match.run())
