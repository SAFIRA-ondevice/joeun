import importlib.util
import struct
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("raspi_pcm16_stream.py")
SPEC = importlib.util.spec_from_file_location("raspi_pcm16_stream", MODULE_PATH)
stream = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stream)


class FakePort:
    def __init__(self, incoming=b""):
        self.incoming = bytearray(incoming)
        self.writes = []
        self._safira_prefetched = b""

    @property
    def in_waiting(self):
        return len(self.incoming)

    def write(self, data):
        self.writes.append(data)

    def read(self, size=1):
        data = bytes(self.incoming[:size])
        del self.incoming[:size]
        return data

    def readline(self):
        if not self.incoming:
            return b""
        newline = self.incoming.find(b"\n")
        size = newline + 1 if newline >= 0 else len(self.incoming)
        return self.read(size)


class StreamTests(unittest.TestCase):
    def test_packet_reader_returns_one_320_sample_m3c0_frame(self):
        payload = b"\0" * stream.MIC_PAYLOAD_BYTES
        reader = stream.PacketReader()
        reader.buffer.extend(stream.HEADER.pack(stream.MIC_MAGIC, 12, stream.FRAME_SAMPLES) + payload)
        self.assertEqual(reader._pop(), (12, payload))

    def test_mix_channels_averages_three_pcm16_channels(self):
        triples = [(30000, 30000, 30000), (-30000, -30000, -30000)] * 160
        payload = struct.pack("<" + "h" * 960, *sum(triples, ()))
        expected = struct.pack("<" + "h" * 320, *([30000, -30000] * 160))
        self.assertEqual(stream.mix_channels(payload, 1.0), expected)

    def test_mix_channels_clips_amplified_pcm16(self):
        payload = struct.pack("<" + "h" * 960, *([30000, 30000, 30000] * 320))
        self.assertEqual(struct.unpack_from("<h", stream.mix_channels(payload, 2.0))[0], 32767)

    def test_start_stream_accepts_already_streaming_packets(self):
        port = FakePort(b"M3C0" + b"\0" * 32)
        stream.start_stream(port)
        self.assertEqual(port.writes, [b"START_INMP_ONLY\n"])
        self.assertTrue(port._safira_prefetched.startswith(b"M3C0"))

    def test_start_stream_accepts_started_text(self):
        port = FakePort(b"READY\nSTARTED\n")
        stream.start_stream(port)
        self.assertEqual(port.writes, [b"START_INMP_ONLY\n"])


if __name__ == "__main__":
    unittest.main()
