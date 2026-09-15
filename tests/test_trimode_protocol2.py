import os
import sys
import struct
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings
from app.server import _StreamState, _Slot, resolve_route, _on_control_frame, _on_audio_frame

class DummyWebSocket:
    def __init__(self):
        self.sent = []
    async def send_json(self, data):
        self.sent.append(data)

class TestTriModeProtocol2(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.settings = Settings(sample_rate=16000, sentence_streaming=True, warmup=False, mt_backend='qwen_hf')
        self.ws = DummyWebSocket()
        self.state = _StreamState(self.settings, self.ws)
        self.state.languages = ['en', 'ar']
        self.state.channel_map = {'0': 'L', '1': 'R'}

    def test_resolve_route_hybrid_a_earbud_mic(self):
        self.state.mode = 'hybrid_a'
        r = resolve_route(self.state, capture='earbud_mic')
        self.assertEqual(r['speaker_id'], 0)
        self.assertEqual(r['sink'], 'speaker')
        self.assertEqual(r['attribution'], 'capture_source')
        self.assertEqual(r['source'], 'en')
        self.assertEqual(r['target'], 'ar')

    def test_resolve_route_hybrid_a_phone_mic(self):
        self.state.mode = 'hybrid_a'
        r = resolve_route(self.state, capture='phone_mic')
        self.assertEqual(r['speaker_id'], 1)
        self.assertEqual(r['sink'], 'earbuds')
        self.assertEqual(r['attribution'], 'capture_source')
        self.assertEqual(r['source'], 'ar')
        self.assertEqual(r['target'], 'en')

    def test_resolve_route_share_b(self):
        self.state.mode = 'share_b'
        r0 = resolve_route(self.state, lid_winner='en', lid_conf=0.92)
        self.assertEqual(r0['speaker_id'], 0)
        self.assertEqual(r0['sink'], 'earbud_r')
        self.assertEqual(r0['attribution'], 'lid')
        self.assertEqual(r0['lid_conf'], 0.92)

        r1 = resolve_route(self.state, lid_winner='ar', lid_conf=0.88)
        self.assertEqual(r1['speaker_id'], 1)
        self.assertEqual(r1['sink'], 'earbud_l')
        self.assertEqual(r1['attribution'], 'lid')
        self.assertEqual(r1['lid_conf'], 0.88)

    def test_resolve_route_listen_c(self):
        self.state.mode = 'listen_c'
        self.state.source = 'en'
        self.state.target = 'ar'
        r = resolve_route(self.state)
        self.assertEqual(r['sink'], 'earbuds')
        self.assertEqual(r['attribution'], 'pinned')
        self.assertEqual(r['direction'], 'en->ar')

    def test_resolve_route_pair_auto_backward_compat(self):
        self.state.mode = 'pair_auto'
        r = resolve_route(self.state, lid_winner='en')
        self.assertEqual(r['speaker_id'], 0)
        self.assertEqual(r['channel'], 'L')

    async def test_config_validation_valid_protocol2(self):
        await _on_control_frame(self.state, {
            'protocol': 2,
            'mode': 'hybrid_a',
            'languages': ['en', 'ar'],
            'speakers': [{'id': 0, 'role': 'me', 'lang': 'en'}, {'id': 1, 'role': 'guest', 'lang': 'ar'}]
        })
        self.assertEqual(len(self.ws.sent), 1)
        conf = self.ws.sent[0]
        self.assertEqual(conf.get('event'), 'config')
        self.assertEqual(conf.get('protocol'), 2)
        self.assertEqual(conf.get('mode'), 'hybrid_a')
        self.assertEqual(len(conf.get('speakers', [])), 2)

    async def test_config_validation_invalid_mode(self):
        self.state.protocol_version = 2
        await _on_control_frame(self.state, {
            'protocol': 2,
            'mode': 'magic_telepathy'
        })
        self.assertEqual(len(self.ws.sent), 1)
        err = self.ws.sent[0]
        self.assertEqual(err.get('event'), 'config_error')
        self.assertEqual(err.get('reason'), 'unsupported_mode')

    async def test_config_validation_missing_languages(self):
        self.state.protocol_version = 2
        self.state.languages = []
        await _on_control_frame(self.state, {
            'protocol': 2,
            'mode': 'share_b',
            'languages': []
        })
        self.assertEqual(len(self.ws.sent), 1)
        err = self.ws.sent[0]
        self.assertEqual(err.get('event'), 'config_error')
        self.assertEqual(err.get('reason'), 'missing_languages')

    async def test_utt_meta_action(self):
        slot = _Slot(1, 'en', 'ar')
        self.state.slots[1] = slot
        await _on_control_frame(self.state, {
            'action': 'utt_meta',
            'utt': 1,
            'capture': 'earbud_mic'
        })
        self.assertEqual(slot.capture, 'earbud_mic')
        self.assertEqual(self.ws.sent[-1].get('event'), 'utt_meta_ack')

    async def test_audio_frame_capture_flags(self):
        self.state.protocol_version = 2
        # Header <BBHH: version=2, flags=0x08 (earbud_mic), utt=1, seq=0
        header = struct.pack('<BBHH', 2, 0x08, 1, 0)
        pcm = b'\x00\x00' * 160
        await _on_audio_frame(self.state, header + pcm)
        self.assertIn(1, self.state.slots)
        self.assertEqual(self.state.slots[1].capture, 'earbud_mic')

        # Header with flags=0x10 (phone_mic), utt=2, seq=0
        header2 = struct.pack('<BBHH', 2, 0x10, 2, 0)
        await _on_audio_frame(self.state, header2 + pcm)
        self.assertIn(2, self.state.slots)
        self.assertEqual(self.state.slots[2].capture, 'phone_mic')

if __name__ == '__main__':
    unittest.main()
