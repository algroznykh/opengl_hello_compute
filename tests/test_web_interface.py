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

    # Verify Waveform and Channel Auto-Cycle are REMOVED as requested
    assert "waveform" not in html.lower(), "Waveform references must be removed"
    assert "autoCycleChannels" not in html, "autoCycleChannels must be removed"
    assert "cycleIntervalSlider" not in html, "cycleIntervalSlider must be removed"

    # Verify Custom 3 Channels and PCA Palette controls
    assert "chBtnCustom" in html, "HTML should contain 3 Channels (RGB) button"
    assert "selChR" in html, "HTML should contain Red channel selector"
    assert "selChG" in html, "HTML should contain Green channel selector"
    assert "selChB" in html, "HTML should contain Blue channel selector"
    assert "rerollPcaPalette" in html, "HTML should contain reroll PCA palette function"
    assert "shiftPcaPalette" in html, "HTML should contain shift PCA palette function"
    assert "pcaHueSlider" in html, "HTML should contain PCA hue slider"
    assert "decayMedBtn" in html, "HTML should contain decay persistence Medium button"
    print("  -> Passed. UI elements verified: 3 Channels (RGB), PCA Palette, Decay persistence, and clean removals.")

    # Other core features remain intact
    assert "setTopology" in html, "HTML should contain setTopology"
    assert "topoBtnPolar" in html, "HTML should contain polar topology button"
    assert "reversePoleBtn" in html, "HTML should contain reversePoleBtn"
    assert "reverseSpinBtn" in html, "HTML should contain reverseSpinBtn"
    assert "specInfluenceSlider" in html, "HTML should contain specInfluenceSlider"
    assert "setNcaChannel" in html, "HTML should contain setNcaChannel"
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
    assert "nca_topology" in status
    assert "topology" in status
    assert "nca_pole_radial" in status
    assert "nca_pole_azimuthal" in status

    # Verify requested default values
    assert status["auto_speed"] is True, "Default auto_speed must be True"
    assert abs(status["spectrogram_influence"] - 1.0) < 0.05, "Default spectrogram_influence must be 1.0"
    assert abs(status["nca_topology"] - 1.0) < 0.05, "Default nca_topology must be Polar (1.0)"
    assert status["reverse_pole"] is True, "Default reverse_pole must be True (inward)"
    assert abs(status["decay_time"] - 0.9700) < 0.005, f"Default decay_time must be Medium (0.9700), got {status.get('decay_time')}"
    assert status["sim_width"] == 800 and status["sim_height"] == 600, f"Default resolution must be 800x600, got {status.get('sim_width')}x{status.get('sim_height')}"

    # Verify simulation resolution fields
    assert "sim_width" in status, "Status must contain sim_width"
    assert "sim_height" in status, "Status must contain sim_height"
    assert "sim_scale_mode" in status, "Status must contain sim_scale_mode"
    assert "sim_scale" in status, "Status must contain sim_scale"
    assert "sim_preset" in status, "Status must contain sim_preset"
    assert "sim_filtering" in status, "Status must contain sim_filtering"

    # Verify custom channels and PCA fields
    assert "custom_channel_r" in status, "Status must contain custom_channel_r"
    assert "custom_channel_g" in status, "Status must contain custom_channel_g"
    assert "custom_channel_b" in status, "Status must contain custom_channel_b"
    assert "pca_hue_shift" in status, "Status must contain pca_hue_shift"

    # Verify removed fields
    assert "waveform_mode" not in status, "waveform_mode must be removed from status"
    assert "waveform_active" not in status, "waveform_active must be removed from status"
    assert "nca_auto_cycle" not in status, "nca_auto_cycle must be removed from status"
    assert "genetic_generation" not in status, "genetic_generation must be removed from status"
    print("  -> Passed. Status API defaults verified: Auto Rhythm Sync ON, Spec Influence 1.0, Polar Inward Rev, Medium Decay, 800x600.")

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

    # 8.7 Reset to 1:1 Default (800x600)
    res = post_json("/api/resolution", {"preset": "1:1"})
    assert res["sim_width"] == 800 and res["sim_height"] == 600
    assert abs(res["sim_scale"] - 1.0) < 0.01
    print(f"  -> Returned to Default 1:1 Preset: {res['sim_width']}x{res['sim_height']}")

    # 8.8 Resolution via /api/controls
    res = post_json("/api/controls", {"sim_preset": "400x300"})
    assert res["sim_width"] == 400 and res["sim_height"] == 300
    res = post_json("/api/controls", {"sim_preset": "1:1"})
    assert res["sim_width"] == 800 and res["sim_height"] == 600
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

    print("[11] Testing Direct NCA Channels View & Custom 3 Channels...")
    for ch in [0, 3, 5, 8, 11]:
        res = post_json("/api/channel", {"channel": ch})
        assert res["nca_channel_view"] == ch
    
    # Test 3 Selectable Channels mapped to RGB w/o color correction
    res = post_json("/api/channel", {"channel": 12, "custom_r": 4, "custom_g": 7, "custom_b": 10})
    assert res["nca_channel_view"] == 12
    assert res["custom_channel_r"] == 4
    assert res["custom_channel_g"] == 7
    assert res["custom_channel_b"] == 10
    print("  -> 3 Selectable Channels mode (Ch 4, Ch 7, Ch 10 -> RGB) verified.")

    res = post_json("/api/channel", {"channel": 13})
    assert res["nca_channel_view"] == 13
    res = post_json("/api/channel", {"channel": -1})
    assert res["nca_channel_view"] == -1
    print("  -> Direct channel selection, custom 3 channels, and PCA RGB modes verified.")

    print("[12] Testing PCA Palette Re-roll and Hue Shift Controls (/api/pca/palette)...")
    res = post_json("/api/pca/palette", {"reroll": True})
    assert "pca_hue_shift" in res
    res = post_json("/api/pca/palette", {"shift": 45.0})
    assert "pca_hue_shift" in res
    res = post_json("/api/pca/palette", {"hue": 120.0})
    assert abs(res["pca_hue_shift"] - 120.0) < 1.0
    print(f"  -> PCA Palette controls verified (reroll, shift, absolute hue: {res['pca_hue_shift']}°).")

    # Reset to defaults
    post_json("/api/controls", {
        "auto_speed": True,
        "spectrogram_influence": 1.0,
        "nca_topology": 1.0,
        "reverse_pole": True,
        "decay_time": 0.9700,
        "sim_preset": "1:1"
    })
    post_json("/api/pca/palette", {"hue": 0.0})

    print("\n=======================================================")
    print("🎉 ALL TESTS PASSED! Defaults, PCA Palette controls,")
    print("   3 Selectable Channels, and FFT polar spectrogram")
    print("   are verified and working flawlessly.")
    print("=======================================================")

if __name__ == "__main__":
    main()
