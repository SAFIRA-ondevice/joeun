"""UDP helpers for SAFIRA SPK0 audio over WireGuard.

WireGuard itself is configured outside this module. Pass the WireGuard peer
address and ports at runtime; no network address is hard-coded here.
"""
from __future__ import annotations

import socket
from typing import Optional

import numpy as np

from spk0 import PACKET_BYTES, pack_spk0, unpack_spk0


class Spk0UdpSender:
    def __init__(self, peer_host: str, peer_port: int):
        self.peer = (peer_host, int(peer_port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sequence = 0

    def send(self, pcm: np.ndarray) -> int:
        packet = pack_spk0(self.sequence, pcm)
        sent = self.sock.sendto(packet, self.peer)
        if sent != PACKET_BYTES:
            raise OSError(f"partial UDP send: {sent}/{PACKET_BYTES}")
        seq = self.sequence
        self.sequence = (self.sequence + 1) & 0xFFFF
        return seq

    def close(self):
        self.sock.close()


class Spk0UdpReceiver:
    def __init__(
        self,
        bind_host: str,
        bind_port: int,
        expected_peer_host: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.expected_peer_host = expected_peer_host
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind_host, int(bind_port)))
        self.sock.settimeout(timeout)

    def recv(self) -> tuple[int, np.ndarray, tuple[str, int]]:
        packet, peer = self.sock.recvfrom(PACKET_BYTES + 64)
        if self.expected_peer_host is not None and peer[0] != self.expected_peer_host:
            raise ValueError(f"unexpected SPK0 peer={peer[0]}")
        seq, pcm = unpack_spk0(packet)
        return seq, pcm, peer

    def close(self):
        self.sock.close()
