"""Backend commands: run the service, replay the dataset as NDTP, check the online hint."""

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from transport_backend.config import Settings


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "transport_backend.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
        access_log=False,
    )


def _send_ndtp(args: argparse.Namespace) -> None:
    from transport_backend.ndtp_client import replay_to_ndtp

    stats = asyncio.run(
        replay_to_ndtp(
            data_root=args.data,
            split=args.split,
            host=args.host,
            port=args.port,
            speed=args.speed,
            start_at=args.start,
            duration_s=args.duration,
            vehicles=args.vehicles.split(",") if args.vehicles else None,
            time_offset_s=args.time_offset,
        )
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))
    if stats.errors:
        # A unit that could not be delivered after all retries fails the command.
        raise SystemExit(1)


def _check_hint(args: argparse.Namespace) -> None:
    from transport_backend.hint_check import check_hint

    print(json.dumps(check_hint(args.data, args.split, args.radius), ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    serve = commands.add_parser("serve", help="Run the dispatcher API and the ingest engine")
    serve.add_argument("--host", default="0.0.0.0")  # noqa: S104 - container service
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--log-level", default="info")
    serve.set_defaults(handler=_serve)

    send = commands.add_parser("send-ndtp", help="Replay dataset telemetry as NDTP packets")
    send.add_argument("--data", type=Path, default=Path("dataset"))
    send.add_argument("--split", default="test")
    send.add_argument("--host", default="127.0.0.1")
    send.add_argument("--port", type=int, default=9201)
    send.add_argument("--speed", type=float, default=30.0)
    send.add_argument(
        "--start", default=None, help="Source time to start from, e.g. '2026-01-06 12:00:00'"
    )
    send.add_argument("--duration", type=float, default=None, help="Source seconds to send")
    send.add_argument("--vehicles", default=None, help="Comma separated tr_id filter")
    send.add_argument("--time-offset", type=float, default=0.0)
    send.set_defaults(handler=_send_ndtp)

    check = commands.add_parser(
        "check-hint", help="Compare the estimated online deviation with the supplied cur_dev_s"
    )
    check.add_argument("--data", type=Path, default=Path("dataset"))
    check.add_argument("--split", default="test")
    check.add_argument("--radius", type=float, default=60.0)
    check.set_defaults(handler=_check_hint)

    settings = commands.add_parser("settings", help="Print the effective configuration")
    settings.set_defaults(
        handler=lambda args: print(
            json.dumps(Settings.from_env().to_dict(), ensure_ascii=False, indent=2)
        )
    )

    args = parser.parse_args()
    args.handler(args)
