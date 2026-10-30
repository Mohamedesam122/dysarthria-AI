"""Run from the repo root:  python -m unittest discover -s tests   (or pytest)
No dataset, model, or running backend needed."""
import io
import os
import sys
import unittest
import wave
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

import requests  # noqa: E402

import api_client as ac  # noqa: E402

S = ac.Settings()


def make_wav(seconds=1.0, rate=16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(seconds * rate))
    return buf.getvalue()


GOOD = {
    "status": "success",
    "transcript": "hello",
    "confidence": None,
    "confidence_type": None,
    "model_version": "baseline-v1",
    "explanation": {"available": False, "limitations": ["Not supported."]},
    "warnings": [],
    "disclaimer": "AI-generated prediction; it may be incorrect and is not a medical diagnosis.",
}


def fake_resp(ok=True, status=200, json_data=None, json_error=False):
    r = MagicMock()
    r.ok, r.status_code = ok, status
    if json_error:
        r.json.side_effect = ValueError
    else:
        r.json.return_value = json_data
    return r


class ValidateUpload(unittest.TestCase):
    def test_valid_wav(self):
        self.assertIsNone(ac.validate_upload("a.wav", make_wav(), S))

    def test_empty(self):
        self.assertIn("empty", ac.validate_upload("a.wav", b"", S))

    def test_wrong_extension(self):
        self.assertIn("isn't supported", ac.validate_upload("a.mp3", make_wav(), S))

    def test_corrupt_wav(self):
        self.assertIn("couldn't be read", ac.validate_upload("a.wav", b"not audio", S))

    def test_too_long(self):
        self.assertIn("seconds", ac.validate_upload("a.wav", make_wav(40), S))

    def test_too_large(self):
        tiny = ac.Settings(max_upload_mb=0.001)
        self.assertIn("MB", ac.validate_upload("a.wav", make_wav(1), tiny))


class Normalize(unittest.TestCase):
    def test_good(self):
        out = ac.normalize_response(GOOD)
        self.assertEqual(out["transcript"], "hello")
        self.assertIsNone(out["confidence"])
        self.assertFalse(out["explanation"]["available"])
        self.assertFalse(out["is_mock"])

    def test_missing_explanation_defaults_to_unavailable(self):
        out = ac.normalize_response({"transcript": "x"})
        self.assertFalse(out["explanation"]["available"])
        self.assertEqual(out["disclaimer"], ac.DEFAULT_DISCLAIMER)

    def test_bad_payload(self):
        with self.assertRaises(ac.ApiError):
            ac.normalize_response({"nope": 1})
        with self.assertRaises(ac.ApiError):
            ac.normalize_response("text")


class Predict(unittest.TestCase):
    def test_mock_is_labelled_and_skips_network(self):
        with patch("api_client.requests.post") as post:
            out = ac.predict("a.wav", b"x", ac.with_mock(S, True))
        post.assert_not_called()
        self.assertTrue(out["is_mock"])
        self.assertEqual(out["model_version"], "mock")

    def test_success(self):
        with patch("api_client.requests.post", return_value=fake_resp(json_data=GOOD)):
            out = ac.predict("a.wav", b"x", S)
        self.assertEqual(out["transcript"], "hello")

    def test_backend_error_message_is_shown(self):
        r = fake_resp(ok=False, status=422, json_data={"message": "Unsupported audio."})
        with patch("api_client.requests.post", return_value=r):
            with self.assertRaises(ac.ApiError) as cm:
                ac.predict("a.wav", b"x", S)
        self.assertEqual(str(cm.exception), "Unsupported audio.")

    def test_server_error_without_message(self):
        r = fake_resp(ok=False, status=500, json_error=True)
        with patch("api_client.requests.post", return_value=r):
            with self.assertRaises(ac.ApiError) as cm:
                ac.predict("a.wav", b"x", S)
        self.assertIn("500", str(cm.exception))

    def test_connection_error_hides_internal_text(self):
        with patch("api_client.requests.post", side_effect=requests.ConnectionError("boom")):
            with self.assertRaises(ac.ApiError) as cm:
                ac.predict("a.wav", b"x", S)
        self.assertNotIn("boom", str(cm.exception))

    def test_timeout(self):
        with patch("api_client.requests.post", side_effect=requests.Timeout()):
            with self.assertRaises(ac.ApiError) as cm:
                ac.predict("a.wav", b"x", S)
        self.assertIn("too long", str(cm.exception))

    def test_invalid_json(self):
        with patch("api_client.requests.post", return_value=fake_resp(json_error=True)):
            with self.assertRaises(ac.ApiError):
                ac.predict("a.wav", b"x", S)

    def test_error_status_in_body(self):
        r = fake_resp(json_data={"status": "error", "message": "Model failed."})
        with patch("api_client.requests.post", return_value=r):
            with self.assertRaises(ac.ApiError) as cm:
                ac.predict("a.wav", b"x", S)
        self.assertEqual(str(cm.exception), "Model failed.")


class Health(unittest.TestCase):
    def test_up(self):
        with patch("api_client.requests.get", return_value=fake_resp()):
            self.assertTrue(ac.check_health(S)[0])

    def test_down(self):
        with patch("api_client.requests.get", side_effect=requests.ConnectionError()):
            self.assertFalse(ac.check_health(S)[0])


if __name__ == "__main__":
    unittest.main()
