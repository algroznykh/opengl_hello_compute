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
        res = json.loads(resp.read().decode("utf-8"))
    time.sleep(0.02)
    return res

def main():
    print("[1] Testing GET / (Minimal HTML Interface)...")
    html, _ = get("/")
    assert "<title>Radar Shader Controls</title>" in html
    assert "scanning_speed" in html
    assert "decay_time" in html
    assert "scale" in html
    assert "auto_speed" in html
    assert "auto_freq" in html
    assert "nca_perturb" in html
    assert "multiplySpeed(2.0)" in html
    assert "multiplySpeed(0.25)" in html
    assert "<canvas" not in html, "Canvas element should remain removed for minimal UI"
    print("  -> Passed. Minimal HTML served with speed multipliers and NCA perturbation.")

    print("[2] Testing GET /api/status...")
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    print(f"  -> Initial status: {status}")
    assert "scale" in status
    assert "scale_type" in status
    assert "auto_speed" in status
    assert "rhythm_division" in status
    assert "auto_freq" in status
    assert "nca_perturb" in status
    assert "nca_seed" in status
    assert "nca_step_scale" in status
    print("  -> Passed. Status API returns all autotune, scale, and NCA fields.")

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

    print("[4] Testing Rotation Speed Multipliers (2x, 4x faster / slower)...")
    # Set known base speed first
    post_json("/api/controls", {"scanning_speed": 0.40})
    
    # Test 2x faster
    res = post_json("/api/controls", {"speed_multiplier": 2.0})
    assert abs(res["scanning_speed"] - 0.80) < 0.01, f"Expected 0.80, got {res['scanning_speed']}"
    print(f"  -> 2x faster multiplier: speed = {res['scanning_speed']} rev/s")

    # Test 4x faster
    res = post_json("/api/controls", {"speed_multiplier": 4.0})
    assert abs(res["scanning_speed"] - 3.20) < 0.02, f"Expected 3.20, got {res['scanning_speed']}"
    print(f"  -> 4x faster multiplier: speed = {res['scanning_speed']} rev/s")

    # Test 1/2x slower (0.5x)
    res = post_json("/api/controls", {"speed_multiplier": 0.5})
    assert abs(res["scanning_speed"] - 1.60) < 0.02, f"Expected 1.60, got {res['scanning_speed']}"
    print(f"  -> 1/2x slower multiplier: speed = {res['scanning_speed']} rev/s")

    # Test 1/4x slower (0.25x)
    res = post_json("/api/controls", {"speed_multiplier": 0.25})
    assert abs(res["scanning_speed"] - 0.40) < 0.02, f"Expected 0.40, got {res['scanning_speed']}"
    print(f"  -> 1/4x slower multiplier: speed = {res['scanning_speed']} rev/s")

    print("[5] Testing NCA Weights Perturbation controls...")
    res = post_json("/api/controls", {
        "nca_perturb": 0.35,
        "nca_seed": 77.0,
        "nca_step_scale": 1.4
    })
    assert abs(res["nca_perturb"] - 0.35) < 0.01, f"Expected 0.35, got {res['nca_perturb']}"
    assert abs(res["nca_seed"] - 77.0) < 0.01, f"Expected 77.0, got {res['nca_seed']}"
    assert abs(res["nca_step_scale"] - 1.4) < 0.01, f"Expected 1.4, got {res['nca_step_scale']}"
    print(f"  -> NCA perturb={res['nca_perturb']}, seed={res['nca_seed']}, step_scale={res['nca_step_scale']}")

    # Reset NCA
    res = post_json("/api/controls", {"nca_perturb": 0.0, "nca_seed": 0.0, "nca_step_scale": 1.0})
    assert abs(res["nca_perturb"] - 0.0) < 0.01
    assert abs(res["nca_step_scale"] - 1.0) < 0.01
    print("  -> NCA reset verified.")

    print("[6] Testing Autotune options...")
    res = post_json("/api/controls", {"auto_speed": True, "rhythm_division": 2, "auto_freq": True})
    assert res["auto_speed"] is True
    assert res["auto_freq"] is True
    assert res["rhythm_division"] == 2
    print(f"  -> Auto speed & auto freq enabled. Detected BPM: {res.get('detected_bpm')}")

    print("[7] Testing Waveform Visualization Modes (Auto fallback, Always, Off)...")
    # Test setting waveform_mode to "auto"
    res = post_json("/api/controls", {"waveform_mode": "auto"})
    assert res["waveform_mode"] == 1, f"Expected 1, got {res['waveform_mode']}"
    assert res["waveform_mode_name"] == "auto", f"Expected auto, got {res['waveform_mode_name']}"
    assert "rhythm_established" in res
    assert "waveform_active" in res
    print(f"  -> Waveform mode 'auto' verified. rhythm_established={res['rhythm_established']}, waveform_active={res['waveform_active']}")

    # Test setting waveform_mode to "always"
    res = post_json("/api/controls", {"waveform_mode": "always"})
    assert res["waveform_mode"] == 2, f"Expected 2, got {res['waveform_mode']}"
    assert res["waveform_mode_name"] == "always", f"Expected always, got {res['waveform_mode_name']}"
    assert res["waveform_active"] is True
    print("  -> Waveform mode 'always' verified (waveform_active=True).")

    # Test setting waveform_mode to "off"
    res = post_json("/api/controls", {"waveform_mode": "off"})
    assert res["waveform_mode"] == 0, f"Expected 0, got {res['waveform_mode']}"
    assert res["waveform_mode_name"] == "off", f"Expected off, got {res['waveform_mode_name']}"
    assert res["waveform_active"] is False
    print("  -> Waveform mode 'off' verified (waveform_active=False).")

    # Test integer mode
    res = post_json("/api/controls", {"waveform_mode": 1})
    assert res["waveform_mode"] == 1 and res["waveform_mode_name"] == "auto"
    print("  -> Waveform integer mode 1 verified.")

    # Check that HTML contains waveform controls
    html, _ = get("/")
    assert "setWaveformMode" in html
    assert "wfBtnAuto" in html
    assert "wfBtnAlways" in html
    assert "wfBtnOff" in html
    print("  -> Minimal HTML interface contains all waveform controls and status indicators.")

    print("\nALL CONTROLLER, NCA PERTURBATION & WAVEFORM TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
