#!/usr/bin/env python3
"""Play the ESP32 three-channel M3C0 PCM16 stream on a Raspberry Pi."""

import argparse
import struct
import subprocess
import sys
import time
import wave
from contextlib import ExitStack


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
CHANNEL_NAMES = ("ambient_left", "ambient_right", "voice")


def clamp16(value):
    return max(-32768, min(32767, int(value)))


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
                (offset, magic)
                for magic in (MIC_MAGIC, DBG_MAGIC)
                if (offset := self.buffer.find(magic)) >= 0
            ]
            if not locations:
                keep = max(len(MIC_MAGIC), len(DBG_MAGIC)) - 1
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
            chunk = port.read(max(1, port.in_waiting or 1))
            if not chunk:
                raise TimeoutError("M3C0 packet timeout")
            self.buffer.extend(chunk)


def _print_startup_line(line_buffer):
    if not line_buffer:
        return False
    line = line_buffer.decode("ascii", errors="ignore").strip()
    if line:
        print(f"ESP32: {line}")
    return line == "STARTED"


def start_stream(port):
    deadline = time.monotonic() + 8.0
    command = b"START_INMP_ONLY\n"
    last_send = 0.0
    line_buffer = bytearray()
    packet_probe = bytearray()

    while time.monotonic() < deadline:
        now = time.monotonic()
        if now - last_send >= 1.0:
            port.write(command)
            last_send = now

        chunk = port.read(max(1, port.in_waiting or 1))
        if not chunk:
            continue

        packet_probe.extend(chunk)
        magic_offsets = [
            offset
            for magic in (MIC_MAGIC, DBG_MAGIC)
            if (offset := packet_probe.find(magic)) >= 0
        ]
        if magic_offsets:
            magic_at = min(magic_offsets)
            port._safira_prefetched = bytes(packet_probe[magic_at:])
            print("ESP32 is already streaming M3C0 audio; continuing.")
            return
        del packet_probe[:-3]

        for value in chunk:
            if value in (10, 13):
                if _print_startup_line(line_buffer):
                    return
                line_buffer.clear()
            elif 32 <= value <= 126:
                if len(line_buffer) < 128:
                    line_buffer.append(value)
            else:
                line_buffer.clear()

    raise TimeoutError("ESP32 did not answer STARTED or begin M3C0 streaming")


def split_channels(payload):
    samples = struct.unpack("<" + "h" * (FRAME_SAMPLES * INPUT_CHANNELS), payload)
    return tuple(samples[channel::INPUT_CHANNELS] for channel in range(INPUT_CHANNELS))


def pack_pcm16(samples):
    return struct.pack("<" + "h" * len(samples), *samples)


def mix_channels(payload, volume):
    samples = struct.unpack("<" + "h" * (FRAME_SAMPLES * INPUT_CHANNELS), payload)
    volume = max(0.0, volume)
    mixed = []
    for index in range(FRAME_SAMPLES):
        base = index * INPUT_CHANNELS
        value = sum(samples[base:base + INPUT_CHANNELS]) // INPUT_CHANNELS
        mixed.append(clamp16(value * volume))
    return pack_pcm16(mixed)


class AlsaPlayback:
    def __init__(self, device=""):
        command = ["aplay", "-q", "-t", "raw"]
        if device:
            command.extend(("-D", device))
        command.extend(("-f", "S16_LE", "-c", "1", "-r", str(SAMPLE_RATE)))
        try:
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
        except FileNotFoundError as error:
            raise RuntimeError("aplay is not installed; install alsa-utils on the Raspberry Pi") from error

    def write(self, pcm):
        if self.process.poll() is not None:
            raise RuntimeError("aplay exited before audio could be played")
        try:
            self.process.stdin.write(pcm)
            self.process.stdin.flush()
        except BrokenPipeError as error:
            raise RuntimeError("aplay stopped accepting PCM16 audio") from error

    def close(self):
        if self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.close()
        try:
            return self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            return self.process.wait(timeout=2)


def open_wavs(stack, prefix):
    if not prefix:
        return []
    base = prefix[:-4] if prefix.lower().endswith(".wav") else prefix
    wavs = []
    for name in CHANNEL_NAMES:
        wav = stack.enter_context(wave.open(f"{base}_{name}.wav", "wb"))
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wavs.append(wav)
    return wavs


def write_wavs(wavs, payload):
    if not wavs:
        return
    for wav, samples in zip(wavs, split_channels(payload)):
        wav.writeframesraw(pack_pcm16(samples))


def print_status(received, missing, reader, started_at):
    audio_seconds = received * FRAME_SAMPLES / SAMPLE_RATE
    elapsed = time.monotonic() - started_at
    print(
        f"audio={audio_seconds:.1f}s wall={elapsed:.1f}s missing={missing} "
        f"bad_headers={reader.bad_headers} resync={reader.resync_bytes}B "
        f"debug={reader.debug_packets}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Receive ESP32 M3C0 PCM16 frames and play a 3-channel mix on Raspberry Pi."
    )
    parser.add_argument("--port", default="/dev/ttyUSB0", help="ESP32 serial port, for example /dev/ttyUSB0")
    parser.add_argument("--seconds", type=float, default=0.0, help="0 means play until Ctrl+C")
    parser.add_argument("--save-prefix", default="", help="Optional WAV filename prefix for all 3 channels")
    parser.add_argument("--speaker", choices=("none", "mix"), default="mix")
    parser.add_argument("--speaker-device", default="", help="Optional ALSA device passed to aplay -D")
    parser.add_argument("--volume", type=float, default=1.0, help="3-channel mix gain; default: 1.0")
    args = parser.parse_args()
    if args.seconds < 0:
        parser.error("--seconds must be zero or positive")

    try:
        import serial
    except ImportError:
        sys.exit("pyserial is missing; run: python3 -m pip install pyserial")

    target_frames = None if args.seconds == 0 else max(1, round(args.seconds * SAMPLE_RATE / FRAME_SAMPLES))
    reader = PacketReader()
    received = 0
    missing = 0
    previous_sequence = None
    started_at = time.monotonic()
    last_status_at = started_at

    with ExitStack() as stack:
        wavs = open_wavs(stack, args.save_prefix)
        playback = AlsaPlayback(args.speaker_device) if args.speaker == "mix" else None
        try:
            with serial.Serial(args.port, SERIAL_BAUD, timeout=0.2, write_timeout=1) as port:
                time.sleep(0.5)
                port.reset_input_buffer()
                port.reset_output_buffer()
                start_stream(port)

                while target_frames is None or received < target_frames:
                    sequence, payload = reader.read(port)
                    if previous_sequence is not None:
                        gap = (sequence - previous_sequence - 1) & 0xFFFF
                        if 0 < gap < 1000:
                            missing += gap
                    previous_sequence = sequence

                    write_wavs(wavs, payload)
                    if playback:
                        playback.write(mix_channels(payload, args.volume))
                    received += 1

                    if time.monotonic() - last_status_at >= 1.0:
                        print_status(received, missing, reader, started_at)
                        last_status_at = time.monotonic()
        except KeyboardInterrupt:
            print("\nstopped")
        finally:
            if playback:
                return_code = playback.close()
                if return_code not in (0, None):
                    print(f"warning: aplay exited with status {return_code}", file=sys.stderr)

    print_status(received, missing, reader, started_at)


if __name__ == "__main__":
    main()
