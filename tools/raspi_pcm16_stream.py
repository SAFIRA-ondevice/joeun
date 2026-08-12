#!/usr/bin/env python3
"""Play the three-channel SAFIRA M3C0 stream as a real-time mono ALSA mix."""

import argparse
import array
import shutil
import struct
import subprocess
import sys
import time


MIC_MAGIC = b"M3C0"
DBG_MAGIC = b"DBG0"
HEADER = struct.Struct("<4sHH")
DBG_PAYLOAD = struct.Struct("<IIIIII")
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320
INPUT_CHANNELS = 3
SERIAL_BAUD = 2_000_000
MIC_PAYLOAD_BYTES = FRAME_SAMPLES * INPUT_CHANNELS * 2
MIC_PACKET_BYTES = HEADER.size + MIC_PAYLOAD_BYTES


class PacketReader:
    def __init__(self):
        self.buffer = bytearray()
        self.bad_headers = 0
        self.resync_bytes = 0
        self.debug_packets = 0
        self.last_debug = None

    def _pop(self):
        while True:
            locations = [
                (at, magic)
                for magic in (MIC_MAGIC, DBG_MAGIC)
                if (at := self.buffer.find(magic)) >= 0
            ]
            if not locations:
                keep = len(MIC_MAGIC) - 1
                if len(self.buffer) > keep:
                    self.resync_bytes += len(self.buffer) - keep
                    del self.buffer[:-keep]
                return None

            magic_at, magic = min(locations, key=lambda item: item[0])
            if magic_at:
                self.resync_bytes += magic_at
                del self.buffer[:magic_at]
            if len(self.buffer) < HEADER.size:
                return None

            _, sequence, count = HEADER.unpack_from(self.buffer)
            if magic == DBG_MAGIC:
                if count != DBG_PAYLOAD.size:
                    self.bad_headers += 1
                    del self.buffer[0]
                    continue
                packet_bytes = HEADER.size + DBG_PAYLOAD.size
                if len(self.buffer) < packet_bytes:
                    return None
                self.last_debug = DBG_PAYLOAD.unpack_from(self.buffer, HEADER.size)
                self.debug_packets += 1
                del self.buffer[:packet_bytes]
                continue

            if count != FRAME_SAMPLES:
                self.bad_headers += 1
                del self.buffer[0]
                continue
            if len(self.buffer) < MIC_PACKET_BYTES:
                return None

            payload = bytes(self.buffer[HEADER.size:MIC_PACKET_BYTES])
            del self.buffer[:MIC_PACKET_BYTES]
            return sequence, payload

    def read(self, port):
        prefetched = getattr(port, "_safira_prefetched", b"")
        if prefetched:
            self.buffer.extend(prefetched)
            port._safira_prefetched = b""

        while True:
            packet = self._pop()
            if packet is not None:
                return packet
            chunk = port.read(max(1, getattr(port, "in_waiting", 0) or 1))
            if not chunk:
                raise TimeoutError("M3C0 packet timeout")
            self.buffer.extend(chunk)


def start_stream(port):
    """Start firmware output, or accept a device that is already sending binary."""
    command = b"START_INMP_ONLY\n"
    port.write(command)
    deadline = time.monotonic() + 8.0
    received = bytearray()

    while time.monotonic() < deadline:
        chunk = port.readline()
        if not chunk:
            continue
        received.extend(chunk)
        magic_at = received.find(MIC_MAGIC)
        if magic_at >= 0:
            port._safira_prefetched = bytes(received[magic_at:])
            return
        for line in received.splitlines():
            if line.strip() == b"STARTED":
                return
        if len(received) > 4096:
            del received[:-3]

    raise TimeoutError("ESP32 did not answer STARTED or send an M3C0 packet")


def mix_channels(payload, volume=1.0):
    if len(payload) != MIC_PAYLOAD_BYTES:
        raise ValueError(f"expected {MIC_PAYLOAD_BYTES} PCM bytes, got {len(payload)}")
    if volume < 0:
        raise ValueError("volume must be non-negative")

    samples = array.array("h")
    samples.frombytes(payload)
    if sys.byteorder != "little":
        samples.byteswap()

    mono = array.array("h")
    for index in range(0, len(samples), INPUT_CHANNELS):
        mixed = round(sum(samples[index:index + INPUT_CHANNELS]) / INPUT_CHANNELS * volume)
        mono.append(max(-32768, min(32767, mixed)))
    if sys.byteorder != "little":
        mono.byteswap()
    return mono.tobytes()


def open_aplay(device=None):
    if shutil.which("aplay") is None:
        raise RuntimeError("aplay not found; install it with: sudo apt install alsa-utils")
    command = ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", str(SAMPLE_RATE), "-c", "1"]
    if device:
        command.extend(["-D", device])
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description="Play the SAFIRA ESP32 microphone mix through Raspberry Pi ALSA.")
    parser.add_argument("--port", required=True, help="Example: /dev/ttyUSB0 or /dev/ttyACM0")
    parser.add_argument("--volume", type=float, default=1.0)
    parser.add_argument("--device", help="Optional ALSA device, for example plughw:0,0")
    args = parser.parse_args()

    try:
        import serial
    except ImportError:
        sys.exit("pyserial is missing; run: python3 -m pip install -r requirements.txt")

    player = open_aplay(args.device)
    reader = PacketReader()
    previous_sequence = None
    missing = 0
    frames = 0

    try:
        with serial.Serial(args.port, SERIAL_BAUD, timeout=0.2, write_timeout=1) as port:
            time.sleep(0.5)
            port.reset_input_buffer()
            port.reset_output_buffer()
            start_stream(port)
            port.timeout = 2.0
            print("Playing 16 kHz mono mix. Press Ctrl+C to stop.")
            while True:
                sequence, payload = reader.read(port)
                if previous_sequence is not None:
                    gap = (sequence - previous_sequence - 1) & 0xFFFF
                    if 0 < gap < 1000:
                        missing += gap
                previous_sequence = sequence
                player.stdin.write(mix_channels(payload, args.volume))
                frames += 1
                if frames % 50 == 0:
                    player.stdin.flush()
                    print(f"audio={frames * 0.02:.0f}s missing={missing} resync={reader.resync_bytes}B")
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        if player.stdin:
            player.stdin.close()
        player.wait(timeout=3)


if __name__ == "__main__":
    main()
