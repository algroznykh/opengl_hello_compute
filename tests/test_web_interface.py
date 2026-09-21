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
    print("[1] Testing GET / (HTML Interface)...")
    html, headers = get("/")
    assert "<title>Radar Shader Controller</title>" in html
    assert "scanning_speed" in html
    assert "decay_time" in html
    assert "freq_min" in html
    assert "radarCanvas" in html
    print("  -> Passed. HTML served correctly.")

    print("[2] Testing GET /api/status...")
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    print(f"  -> Initial status: {status}")
    assert "scanning_speed" in status
    assert "decay_time" in status
    assert "freq_min" in status
    assert "freq_max" in status
    print("  -> Passed. Status API returns all required fields.")

    print("[3] Testing POST /api/controls for Scanning Speed...")
    res = post_json("/api/controls", {"scanning_speed": 1.45})
    assert abs(res["scanning_speed"] - 1.45) < 0.001, f"Expected 1.45, got {res['scanning_speed']}"
    print(f"  -> Scanning speed updated to {res['scanning_speed']} rev/s")

    print("[4] Testing POST /api/controls for Decay Time...")
    res = post_json("/api/controls", {"decay_time": 0.9825})
    assert abs(res["decay_time"] - 0.9825) < 0.001, f"Expected 0.9825, got {res['decay_time']}"
    print(f"  -> Decay time updated to {res['decay_time']}")

    print("[5] Testing POST /api/controls for Freq Range...")
    res = post_json("/api/controls", {"freq_min": 65.0, "freq_max": 7500.0})
    assert abs(res["freq_min"] - 65.0) < 0.1, f"Expected 65.0, got {res['freq_min']}"
    assert abs(res["freq_max"] - 7500.0) < 0.1, f"Expected 7500.0, got {res['freq_max']}"
    print(f"  -> Frequency range updated to {res['freq_min']} Hz - {res['freq_max']} Hz")

    print("[6] Testing simultaneous update of all controls...")
    res = post_json("/api/controls", {
        "scanning_speed": 0.85,
        "decay_time": 0.992,
        "freq_min": 120.0,
        "freq_max": 12000.0,
        "freq_range_low": 0.1,
        "freq_range_high": 0.9
    })
    assert abs(res["scanning_speed"] - 0.85) < 0.001
    assert abs(res["decay_time"] - 0.992) < 0.001
    assert abs(res["freq_min"] - 120.0) < 0.1
    assert abs(res["freq_max"] - 12000.0) < 0.1
    assert abs(res["freq_range_low"] - 0.1) < 0.01
    assert abs(res["freq_range_high"] - 0.9) < 0.01
    print("  -> Passed. All controls updated and verified.")

    print("\nALL WEB CONTROLLER TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
