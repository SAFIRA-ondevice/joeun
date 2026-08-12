#!/usr/bin/env python3
"""Record the SAFIRA ESP32 M3C0 serial stream as mono PCM16 WAV files."""

import argparse
import struct
import sys
import time
import wave
from pathlib import Path


MIC_MAGIC = b"M3C0"
DBG_MAGIC = b"DBG0"
HEADER = struct.Struct("<4sHH")
DBG_PAYLOAD = struct.Struct("<IIIIII")
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 320
INPUT_CHANNELS = 3
MIC_PAYLOAD_BYTES = FRAME_SAMPLES * INPUT_CHANNELS * 2
MIC_PACKET_BYTES = HEADER.size + MIC_PAYLOAD_BYTES
SERIAL_BAUD = 2_000_000
CHANNEL_NAMES = ("ambient_left", "ambient_right", "voice")


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
                values = DBG_PAYLOAD.unpack_from(self.buffer, HEADER.size)
                self.last_debug = values
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
        while True:
            packet = self._pop()
            if packet is not None:
                return packet
            chunk = port.read(max(1, port.in_waiting or 1))
            if not chunk:
                raise TimeoutError("M3C0 packet timeout")
            self.buffer.extend(chunk)


def start_stream(port):
    deadline = time.monotonic() + 8.0
    command = b"START_INMP_ONLY\n"
    last_send = 0.0
    while time.monotonic() < deadline:
        now = time.monotonic()
        if now - last_send >= 1.0:
            port.write(command)
            last_send = now
        line = port.readline().decode("ascii", errors="ignore").strip()
        if line:
            print(f"ESP32: {line}")
        if line == "STARTED":
            port.timeout = 2.0
            return
    raise TimeoutError("ESP32 did not answer STARTED")


def split_pcm16(payload):
    samples = struct.unpack("<" + "h" * (FRAME_SAMPLES * INPUT_CHANNELS), payload)
    channels = [bytearray(), bytearray(), bytearray()]
    for index in range(FRAME_SAMPLES):
        base = index * INPUT_CHANNELS
        for channel in range(INPUT_CHANNELS):
            channels[channel] += struct.pack("<h", samples[base + channel])
    return channels


def open_wavs(output_dir, prefix):
    output_dir.mkdir(parents=True, exist_ok=True)
    opened = []
    for name in CHANNEL_NAMES:
        path = output_dir / f"{prefix}_{name}.wav"
        wav = wave.open(str(path), "wb")
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        opened.append((path, wav))
    return opened


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", required=True, help="Example: /dev/ttyUSB0 or /dev/ttyACM0")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")

    try:
        import serial
    except ImportError:
        sys.exit("pyserial is missing; run: python3 -m pip install pyserial")

    prefix = time.strftime("esp32_%Y%m%d_%H%M%S")
    wavs = open_wavs(args.output_dir, prefix)
    target_frames = max(1, round(args.seconds * SAMPLE_RATE / FRAME_SAMPLES))
    received = 0
    missing = 0
    previous_sequence = None
    reader = PacketReader()
    silence = b"\0" * (FRAME_SAMPLES * 2)

    try:
        with serial.Serial(args.port, SERIAL_BAUD, timeout=0.2, write_timeout=1) as port:
            time.sleep(0.5)
            port.reset_input_buffer()
            port.reset_output_buffer()
            start_stream(port)
            started = time.monotonic()

            while received < target_frames:
                sequence, payload = reader.read(port)
                if previous_sequence is not None:
                    gap = (sequence - previous_sequence - 1) & 0xFFFF
                    if 0 < gap < 1000:
                        missing += gap
                        for _ in range(min(gap, target_frames - received)):
                            for _, wav in wavs:
                                wav.writeframesraw(silence)
                            received += 1
                previous_sequence = sequence

                for (_, wav), pcm in zip(wavs, split_pcm16(payload)):
                    wav.writeframesraw(pcm)
                received += 1
                if received % 50 == 0:
                    print(f"audio={received * 0.02:.1f}s missing={missing} resync={reader.resync_bytes}B")

            elapsed = time.monotonic() - started
    finally:
        for _, wav in wavs:
            wav.close()

    print(f"Recorded {received * 0.02:.2f}s in {elapsed:.2f}s; missing frames={missing}")
    print(f"bad headers={reader.bad_headers}, resync bytes={reader.resync_bytes}")
    for path, _ in wavs:
        print(path.resolve())


if __name__ == "__main__":
    main()
