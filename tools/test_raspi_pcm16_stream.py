import pathlib
import struct
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch


TOOLS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS_DIR))

import raspi_pcm16_stream as stream


TARGET_FRAME_SAMPLES = 320
TARGET_INPUT_CHANNELS = 3


class FakePort:
    def __init__(self, data=b""):
        self.buffer = bytearray(data)
        self.writes = []

    @property
    def in_waiting(self):
        return len(self.buffer)

    def read(self, size=1):
        chunk = bytes(self.buffer[:size])
        del self.buffer[:size]
        return chunk

    def readline(self):
        if not self.buffer:
            return b""
        newline = self.buffer.find(b"\n")
        size = len(self.buffer) if newline < 0 else newline + 1
        return self.read(size)

    def write(self, data):
        self.writes.append(data)


class PacketReaderTests(unittest.TestCase):
    def test_returns_one_pcm16_frame(self):
        payload = b"\0" * stream.MIC_PAYLOAD_BYTES
        reader = stream.PacketReader()
        reader.buffer.extend(
            stream.HEADER.pack(stream.MIC_MAGIC, 12, stream.FRAME_SAMPLES) + payload
        )

        self.assertEqual(reader._pop(), (12, payload))


class MixerTests(unittest.TestCase):
    def test_averages_all_three_channels(self):
        triples = [(30000, 30000, 30000), (-30000, -30000, -30000)] * 160
        payload = struct.pack(
            "<" + "h" * (TARGET_FRAME_SAMPLES * TARGET_INPUT_CHANNELS),
            *sum(triples, ()),
        )

        mixed = stream.mix_channels(payload, 1.0)

        self.assertEqual(
            mixed,
            struct.pack("<" + "h" * 320, *([30000, -30000] * 160)),
        )

    def test_clips_amplified_pcm16(self):
        payload = struct.pack(
            "<" + "h" * 960,
            *([30000, 30000, 30000] * 320),
        )

        mixed = stream.mix_channels(payload, 2.0)

        self.assertEqual(struct.unpack_from("<h", mixed)[0], 32767)


class StartupTests(unittest.TestCase):
    def test_accepts_already_streaming_m3c0_packets(self):
        port = FakePort(b"M3C0" + b"\0" * 32)

        stream.start_stream(port)

        self.assertEqual(port.writes, [b"START_INMP_ONLY\n"])

    def test_preserves_already_streaming_packet_for_reader(self):
        payload = b"\0" * stream.MIC_PAYLOAD_BYTES
        packet = stream.HEADER.pack(stream.MIC_MAGIC, 21, stream.FRAME_SAMPLES) + payload
        port = FakePort(packet)

        stream.start_stream(port)

        self.assertEqual(stream.PacketReader().read(port), (21, payload))

    def test_startup_does_not_require_usb_purge_methods(self):
        port = FakePort(b"STARTED\n")

        stream.start_stream(port)

        self.assertEqual(port.writes, [b"START_INMP_ONLY\n"])


class AlsaPlaybackTests(unittest.TestCase):
    @patch("raspi_pcm16_stream.subprocess.Popen")
    def test_launches_aplay_for_mono_pcm16(self, popen):
        process = Mock()
        process.poll.return_value = None
        process.stdin.closed = False
        popen.return_value = process

        playback = stream.AlsaPlayback("default")

        popen.assert_called_once_with(
            ["aplay", "-q", "-t", "raw", "-D", "default", "-f", "S16_LE", "-c", "1", "-r", "16000"],
            stdin=subprocess.PIPE,
        )
        playback.close()

    @patch("raspi_pcm16_stream.subprocess.Popen")
    def test_raises_when_aplay_exits_early(self, popen):
        process = Mock()
        process.poll.return_value = 1
        popen.return_value = process
        playback = stream.AlsaPlayback()

        with self.assertRaisesRegex(RuntimeError, "aplay exited"):
            playback.write(b"\0\0")


if __name__ == "__main__":
    unittest.main()
