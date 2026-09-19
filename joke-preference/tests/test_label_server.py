from __future__ import annotations

import http.client
import tempfile
import threading
import unittest
import urllib.parse
from pathlib import Path

from joke_pref.data import load_labels
from joke_pref.label_server import serve
from tests.fakes import write_sample_premises


class LabelServerTests(unittest.TestCase):
    def setUp(self) -> None:
        d = Path(tempfile.mkdtemp())
        self.labels_dir = d / "labels"
        self.server = serve(write_sample_premises(d), self.labels_dir, port=0)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.cookie = ""

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _req(self, method: str, path: str, form: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        headers = {"Cookie": self.cookie} if self.cookie else {}
        body = None
        if form is not None:
            body = urllib.parse.urlencode(form)
            headers["Content-Type"] = "application/x-www-form-urlencoded"
        conn.request(method, path, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8")
        set_cookie = resp.getheader("Set-Cookie")
        if set_cookie:
            self.cookie = set_cookie.split(";")[0]
        conn.close()
        return resp.status, data

    def test_flow(self) -> None:
        status, page = self._req("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("Who is labeling?", page)

        status, _ = self._req("POST", "/start", {"user": "Nav"})
        self.assertEqual(status, 303)
        self.assertEqual(self.cookie, "user=nav")

        status, page = self._req("GET", "/")
        self.assertIn("Why did the chicken cross the road?", page)
        self.assertIn("0 / 9 punchlines", page)

        status, frag = self._req("POST", "/label", {"punchline_id": "t-001.1", "label": "great"})
        self.assertEqual(status, 200)
        self.assertIn("great on", frag.replace("class='great on'", "great on"))
        self.assertEqual(load_labels(self.labels_dir, "nav"), {"t-001.1": "great"})

        for pid in ("t-001.0", "t-001.2"):
            self._req("POST", "/label", {"punchline_id": pid, "label": "bad"})
        status, frag = self._req("GET", "/card?after=t-001")
        self.assertIn("My therapist", frag)

        status, page = self._req("GET", "/review")
        self.assertIn("3/3", page)

        status, _ = self._req("POST", "/label", {"punchline_id": "nope.0", "label": "bad"})
        self.assertEqual(status, 400)
        status, _ = self._req("POST", "/label", {"punchline_id": "t-001.0", "label": "meh"})
        self.assertEqual(status, 400)

    def test_label_requires_user(self) -> None:
        status, _ = self._req("POST", "/label", {"punchline_id": "t-001.0", "label": "bad"})
        self.assertEqual(status, 401)


if __name__ == "__main__":
    unittest.main()
