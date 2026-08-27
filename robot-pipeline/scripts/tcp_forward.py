#!/usr/bin/env python3
"""Small TCP forwarder used to expose a service from an existing container."""

from __future__ import annotations

import argparse
import select
import socket
import socketserver


class ForwardHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        upstream = socket.create_connection(self.server.target, timeout=5)
        sockets = (self.request, upstream)
        try:
            while True:
                readable, _, _ = select.select(sockets, [], [], 30)
                if not readable:
                    continue
                for source in readable:
                    data = source.recv(65536)
                    if not data:
                        return
                    destination = upstream if source is self.request else self.request
                    destination.sendall(data)
        finally:
            upstream.close()


class ForwardServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--target-host", required=True)
    parser.add_argument("--target-port", type=int, required=True)
    options = parser.parse_args()
    server = ForwardServer(
        (options.listen_host, options.listen_port), ForwardHandler
    )
    server.target = (options.target_host, options.target_port)
    server.serve_forever()


if __name__ == "__main__":
    main()
