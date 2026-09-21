import urllib.request
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8080"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=3) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        return resp.read().decode("utf-8"), dict(resp.headers)

def post_json(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=3) as resp:
        assert resp.status == 200, f"Expected 200, got {resp.status}"
        return json.loads(resp.read().decode("utf-8"))

def main():
    print("[1] Testing GET / (Minimal HTML Interface)...")
    html, _ = get("/")
    assert "<title>Radar Shader Controls</title>" in html
    assert "scanning_speed" in html
    assert "decay_time" in html
    assert "scale" in html
    assert "auto_speed" in html
    assert "auto_freq" in html
    # Check that non-essential elements like canvas are removed
    assert "<canvas" not in html, "Canvas element should be removed for a minimal UI"
    print("  -> Passed. Minimal HTML served without non-essential elements.")

    print("[2] Testing GET /api/status...")
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    print(f"  -> Initial status: {status}")
    assert "scale" in status
    assert "scale_type" in status
    assert "auto_speed" in status
    assert "rhythm_division" in status
    assert "auto_freq" in status
    print("  -> Passed. Status API returns all autotune and scale fields.")

    print("[3] Testing Frequency Scales (log, mel, linear)...")
    res = post_json("/api/controls", {"scale": "mel"})
    assert res["scale"] == "mel" and res["scale_type"] == 1, f"Expected mel/1, got {res}"
    print("  -> Mel scale verified.")

    res = post_json("/api/controls", {"scale": "linear"})
    assert res["scale"] == "linear" and res["scale_type"] == 2, f"Expected linear/2, got {res}"
    print("  -> Linear scale verified.")

    res = post_json("/api/controls", {"scale": "log"})
    assert res["scale"] == "log" and res["scale_type"] == 0, f"Expected log/0, got {res}"
    print("  -> Log scale verified.")

    print("[4] Testing Autotune Scanning Speed (Musical Rhythms)...")
    res = post_json("/api/controls", {"auto_speed": True, "rhythm_division": 1})
    assert res["auto_speed"] is True
    assert res["rhythm_division"] == 1
    print(f"  -> Auto speed enabled with 1/2 bar rhythm division. Detected BPM: {res.get('detected_bpm')}")

    print("[5] Testing Autotune Frequency Range & Scaling...")
    res = post_json("/api/controls", {"auto_freq": True})
    assert res["auto_freq"] is True
    assert "auto_gain" in res
    print(f"  -> Auto frequency enabled. Auto gain: {res.get('auto_gain')}")

    print("[6] Testing manual override of speed and decay...")
    res = post_json("/api/controls", {
        "auto_speed": False,
        "scanning_speed": 0.75,
        "decay_time": 0.9850,
        "auto_freq": False,
        "freq_min": 80.0,
        "freq_max": 9500.0,
        "scale": "mel"
    })
    assert abs(res["scanning_speed"] - 0.75) < 0.001
    assert abs(res["decay_time"] - 0.9850) < 0.001
    assert abs(res["freq_min"] - 80.0) < 0.1
    assert abs(res["freq_max"] - 9500.0) < 0.1
    assert res["scale"] == "mel"
    assert res["auto_speed"] is False
    assert res["auto_freq"] is False
    print("  -> Manual controls override verified.")

    print("\nALL ENHANCED RADAR WEB INTERFACE TESTS PASSED!")

if __name__ == "__main__":
    main()
