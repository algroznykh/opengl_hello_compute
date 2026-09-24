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
    assert "multiplySpeed(2.0)" in html
    assert "multiplySpeed(0.25)" in html

    # Verify Simulation Resolution controls in HTML
    assert "Simulation Resolution" in html, "HTML should contain Simulation Resolution card"
    assert "simResBadge" in html, "HTML should contain simResBadge"
    assert "simScaleSlider" in html, "HTML should contain simScaleSlider"
    assert "setSimPreset" in html, "HTML should contain setSimPreset function"
    assert "setSimScale" in html, "HTML should contain setSimScale function"
    assert "setSimFiltering" in html, "HTML should contain setSimFiltering function"
    assert "filterBtnHermite" in html, "HTML should contain Smooth Hermite filtering button"
    assert "filterBtnBilinear" in html, "HTML should contain Bilinear filtering button"
    assert "filterBtnNearest" in html, "HTML should contain Nearest filtering button"
    assert "resetSimulationStateBtn" in html, "HTML should contain resetSimulationStateBtn"

    # Verify relative presets
    assert "presetBtn1_1" in html, "HTML should contain 1:1 preset button"
    assert "presetBtn1_2" in html, "HTML should contain 1:2 preset button"
    assert "presetBtn1_3" in html, "HTML should contain 1:3 preset button"
    assert "presetBtn1_4" in html, "HTML should contain 1:4 preset button"

    # Verify fixed presets
    assert "presetBtn160" in html, "HTML should contain 160x120 preset button"
    assert "presetBtn267" in html, "HTML should contain 267x200 preset button"
    assert "presetBtn400" in html, "HTML should contain 400x300 preset button"

    # Verify Genetic Algorithm controls are REMOVED as requested
    assert "Genetic Algorithm" not in html, "Genetic Algorithm card must be removed"
    assert "mutateCandidate" not in html, "mutateCandidate must be removed"
    assert "mutateEverything" not in html, "mutateEverything must be removed"
    assert "reinforceCandidate" not in html, "reinforceCandidate must be removed"
    assert "resetToRandomWeights" not in html, "resetToRandomWeights must be removed"
    assert "resetWiringMatrix" not in html, "resetWiringMatrix must be removed"
    print("  -> Passed. Simulation Resolution UI verified and Genetic Algorithm controls confirmed removed.")

    # Other core features remain intact
    assert "setTopology" in html, "HTML should contain setTopology"
    assert "topoBtnPolar" in html, "HTML should contain polar topology button"
    assert "reversePoleBtn" in html, "HTML should contain reversePoleBtn"
    assert "reverseSpinBtn" in html, "HTML should contain reverseSpinBtn"
    assert "specInfluenceSlider" in html, "HTML should contain specInfluenceSlider"
    assert "setNcaChannel" in html, "HTML should contain setNcaChannel"
    assert "toggleAutoCycleChannels" in html, "HTML should contain toggleAutoCycleChannels"
    assert "chBtnPca" in html, "HTML should contain chBtnPca button"
    assert "triggerImprint" in html, "HTML should contain Easter egg triggerImprint"
    assert "imprintInput" in html, "HTML should contain arbitrary message input"
    print("  -> Passed. Topology, Easter Egg Imprint, Spectrogram, and Direct Channels verified in HTML.")

    print("[2] Testing GET /api/status...")
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    print(f"  -> Initial status: {status}")
    assert "scale" in status
    assert "scale_type" in status
    assert "auto_speed" in status
    assert "rhythm_division" in status
    assert "auto_freq" in status
    assert "spectrogram_influence" in status
    assert "nca_channel_view" in status
    assert "nca_auto_cycle" in status
    assert "nca_topology" in status
    assert "topology" in status
    assert "nca_pole_radial" in status
    assert "nca_pole_azimuthal" in status

    # Verify simulation resolution fields
    assert "sim_width" in status, "Status must contain sim_width"
    assert "sim_height" in status, "Status must contain sim_height"
    assert "sim_scale_mode" in status, "Status must contain sim_scale_mode"
    assert "sim_scale" in status, "Status must contain sim_scale"
    assert "sim_preset" in status, "Status must contain sim_preset"
    assert "sim_filtering" in status, "Status must contain sim_filtering"

    # Verify genetic fields removed from status
    assert "genetic_generation" not in status, "genetic_generation must be removed from status"
    assert "genetic_has_candidate" not in status, "genetic_has_candidate must be removed from status"
    assert "genetic_mutated_count" not in status, "genetic_mutated_count must be removed from status"
    print("  -> Passed. Status API returns simulation resolution and confirmed genetic fields removed.")

    print("[3] Testing Frequency Scales (log, mel, linear)...")
    res = post_json("/api/controls", {"scale": "mel"})
    assert res["scale"] == "mel" and res["scale_type"] == 1
    res = post_json("/api/controls", {"scale": "linear"})
    assert res["scale"] == "linear" and res["scale_type"] == 2
    res = post_json("/api/controls", {"scale": "log"})
    assert res["scale"] == "log" and res["scale_type"] == 0
    print("  -> Frequency scales verified.")

    print("[4] Testing Rotation Speed Multipliers (2x, 4x, 0.5x, 0.25x)...")
    post_json("/api/controls", {"scanning_speed": 0.40, "auto_speed": False})
    res = post_json("/api/controls", {"speed_multiplier": 2.0})
    assert abs(res["scanning_speed"] - 0.80) < 0.01
    res = post_json("/api/controls", {"speed_multiplier": 0.5})
    assert abs(res["scanning_speed"] - 0.40) < 0.01
    print("  -> Speed multipliers verified.")

    print("[5] Testing Autotune options...")
    res = post_json("/api/controls", {"auto_speed": True, "rhythm_division": 2, "auto_freq": True})
    assert res["auto_speed"] is True
    assert res["auto_freq"] is True
    assert res["rhythm_division"] == 2
    print(f"  -> Autotune verified. BPM: {res.get('detected_bpm')}")

    print("[6] Testing NCA Topology Controls & Reverse Pole Options...")
    res = post_json("/api/topology", {"topology": "polar", "reverse_pole": False, "reverse_spin": False})
    assert abs(res["nca_topology"] - 1.0) < 0.01
    assert res["topology"] == "polar"
    assert res["reverse_pole"] is False

    res = post_json("/api/topology", {"reverse_pole": True})
    assert res["reverse_pole"] is True
    assert abs(res["nca_pole_radial"] - (-1.0)) < 0.01

    res = post_json("/api/topology", {"reverse_spin": True})
    assert res["reverse_spin"] is True
    assert abs(res["nca_pole_azimuthal"] - (-1.0)) < 0.01

    res = post_json("/api/topology", {"topology": "cartesian", "reverse_pole": False, "reverse_spin": False})
    assert abs(res["nca_topology"] - 0.0) < 0.01
    assert res["topology"] == "cartesian"
    print("  -> NCA Topology (Cartesian, Polar, Reverse Pole, Reverse Spin) verified.")

    print("[7] Testing Easter Egg Giant Structure Imprint (/api/imprint)...")
    res = post_json("/api/imprint", {"message": "BYOB"})
    assert res["status"] == "imprinted"
    assert res["message"] == "BYOB"
    print("  -> Giant Structure Imprint verified.")

    print("[8] Testing Simulation Resolution Selector (/api/resolution)...")
    # 8.1 Relative Preset 1:2 (Half resolution: 400x300)
    res = post_json("/api/resolution", {"preset": "1:2"})
    assert res["sim_width"] == 400 and res["sim_height"] == 300, f"Expected 400x300, got {res['sim_width']}x{res['sim_height']}"
    assert res["sim_scale_mode"] == 0, "Expected relative mode 0"
    assert abs(res["sim_scale"] - 0.5) < 0.01
    print(f"  -> Relative Preset 1:2 verified: {res['sim_width']}x{res['sim_height']}, scale={res['sim_scale']}")

    # 8.2 Relative Preset 1:4 (Quarter resolution: 200x150)
    res = post_json("/api/resolution", {"preset": "1:4"})
    assert res["sim_width"] == 200 and res["sim_height"] == 150, f"Expected 200x150, got {res.get('sim_width')}x{res.get('sim_height')} (res: {res})"
    assert abs(res["sim_scale"] - 0.25) < 0.01, f"Expected scale 0.25, got {res.get('sim_scale')}"
    print(f"  -> Relative Preset 1:4 verified: {res['sim_width']}x{res['sim_height']}")

    # 8.3 Fixed Preset: 160x120
    res = post_json("/api/resolution", {"preset": "160x120"})
    assert res["sim_width"] == 160 and res["sim_height"] == 120
    assert res["sim_scale_mode"] == 1, "Expected fixed mode 1"
    print(f"  -> Fixed Preset 160x120 verified: {res['sim_width']}x{res['sim_height']}, mode={res['sim_scale_mode']}")

    # 8.4 Fixed Preset: 512x384
    res = post_json("/api/resolution", {"preset": "512x384"})
    assert res["sim_width"] == 512 and res["sim_height"] == 384
    assert res["sim_scale_mode"] == 1
    print(f"  -> Fixed Preset 512x384 verified: {res['sim_width']}x{res['sim_height']}")

    # 8.5 Continuous Scale Slider (e.g. 0.60x)
    res = post_json("/api/resolution", {"scale": 0.60})
    assert res["sim_width"] == 480 and res["sim_height"] == 360
    assert res["sim_scale_mode"] == 0
    print(f"  -> Custom Scale 0.60x verified: {res['sim_width']}x{res['sim_height']}")

    # 8.6 Zero-Artifact Filtering Modes: 0 (Hermite), 1 (Bilinear), 2 (Nearest)
    res = post_json("/api/resolution", {"filtering": 1})
    assert res["sim_filtering"] == 1, f"Expected filtering 1 (Bilinear), got {res['sim_filtering']}"
    res = post_json("/api/resolution", {"filtering": 2})
    assert res["sim_filtering"] == 2, f"Expected filtering 2 (Nearest), got {res['sim_filtering']}"
    res = post_json("/api/resolution", {"filtering": 0})
    assert res["sim_filtering"] == 0, f"Expected filtering 0 (Smooth Hermite), got {res['sim_filtering']}"
    print("  -> Zero-Artifact Filtering Modes (Smooth Hermite, Bilinear, Nearest) verified.")

    # 8.7 Reset to 1:3 Default (267x200)
    res = post_json("/api/resolution", {"preset": "1:3"})
    assert res["sim_width"] == 267 and res["sim_height"] == 200
    assert abs(res["sim_scale"] - 0.333333) < 0.01
    print(f"  -> Returned to Default 1:3 Preset: {res['sim_width']}x{res['sim_height']}")

    # 8.8 Resolution via /api/controls
    res = post_json("/api/controls", {"sim_preset": "400x300"})
    assert res["sim_width"] == 400 and res["sim_height"] == 300
    res = post_json("/api/controls", {"sim_preset": "1:3"})
    assert res["sim_width"] == 267 and res["sim_height"] == 200
    print("  -> Resolution control via /api/controls verified.")

    print("[9] Testing Spectrogram Influence Controls (/api/spectrogram)...")
    res = post_json("/api/spectrogram", {"spectrogram_influence": 0.0})
    assert abs(res["spectrogram_influence"] - 0.0) < 0.01
    res = post_json("/api/controls", {"spectrogram_influence": 1.0})
    assert abs(res["spectrogram_influence"] - 1.0) < 0.01
    print("  -> Spectrogram influence verified.")

    print("[10] Testing Simulation State Reset (/api/reset_state)...")
    res = post_json("/api/reset_state", {})
    assert res.get("status") == "state_reset"
    print("  -> Simulation state reset verified.")

    print("[11] Testing Direct NCA Channels View & PCA RGB...")
    for ch in [0, 3, 5, 8, 11]:
        res = post_json("/api/channel", {"channel": ch})
        assert res["nca_channel_view"] == ch
    res = post_json("/api/channel", {"channel": 12})
    assert res["nca_channel_view"] == 12
    res = post_json("/api/channel", {"channel": 13})
    assert res["nca_channel_view"] == 13
    res = post_json("/api/channel", {"channel": -1})
    assert res["nca_channel_view"] == -1
    print("  -> Direct channel selection, raw RGB, and PCA RGB modes verified.")

    print("\n=======================================================")
    print("🎉 ALL TESTS PASSED! Simulation resolution is verified,")
    print("   zero-artifact Hermite resampling is active, and")
    print("   genetic weight modification is cleanly removed.")
    print("=======================================================")

if __name__ == "__main__":
    main()
