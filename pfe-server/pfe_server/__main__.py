"""Entry point for running pfe_server as a module."""

import argparse
import sys
from typing import Optional

from pfe_server.app import serve


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="PFE Server")
    parser.add_argument("--port", type=int, default=8921, help="Port to run server on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--workspace", type=str, default=None, help="Workspace name")
    parser.add_argument("--adapter", type=str, default="latest", help="Adapter to use")
    parser.add_argument("--api-key", type=str, default=None, help="API key for authentication")
    parser.add_argument("--allow-remote-access", action="store_true", help="Allow remote clients to access management endpoints")
    parser.add_argument("--cors-origins", type=str, default=None, help="Comma-separated list of allowed CORS origins (e.g. 'http://localhost:3000,http://localhost:5173')")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    args = parser.parse_args(argv)

    cors_origins = [o.strip() for o in args.cors_origins.split(",")] if args.cors_origins else None

    result = serve(
        port=args.port,
        host=args.host,
        workspace=args.workspace,
        adapter=args.adapter,
        api_key=args.api_key,
        allow_remote_access=args.allow_remote_access,
        cors_origins=cors_origins,
        dry_run=args.dry_run,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
