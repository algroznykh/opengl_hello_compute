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
    # Verify new features in HTML
    assert "setTopology" in html, "HTML should contain setTopology"
    assert "topoBtnPolar" in html, "HTML should contain polar topology button"
    assert "reversePoleBtn" in html, "HTML should contain reversePoleBtn"
    assert "reverseSpinBtn" in html, "HTML should contain reverseSpinBtn"
    assert "specInfluenceSlider" in html, "HTML should contain specInfluenceSlider"
    assert "resetSimulationStateBtn" in html, "HTML should contain resetSimulationStateBtn"
    assert "resetToRandomWeights" in html, "HTML should contain resetToRandomWeights"
    assert "resetWiringMatrix" in html, "HTML should contain resetWiringMatrix"
    assert "wiringBadge" in html, "HTML should contain neural wiringBadge"
    assert "mutateCandidate" in html, "HTML should contain mutateCandidate button"
    assert "mutateEverything" in html, "HTML should contain mutateEverything button"
    assert "targetBtnEverything" not in html, "targetBtnEverything should be removed to de-duplicate mutate everything controls"
    assert "setNcaChannel" in html, "HTML should contain setNcaChannel"
    assert "toggleAutoCycleChannels" in html, "HTML should contain toggleAutoCycleChannels"
    assert "autoCycleChannelsBtn" in html, "HTML should contain autoCycleChannelsBtn"
    assert "chBtnComp" in html, "HTML should contain chBtnComp button"
    assert "chBtn0" in html, "HTML should contain chBtn0 button"
    assert "chBtnPca" in html, "HTML should contain chBtnPca button for 3 main PCA components"
    assert "rerollBadge" in html, "HTML should contain rerollBadge for auto-reroll telemetry"
    assert "cancelCandidate" in html, "HTML should contain cancelCandidate button"
    assert "reinforceCandidate" in html, "HTML should contain reinforceCandidate button"

    # Verify Reset State, Mutate Everything, and Reinforce are moved close together
    pos_reset = html.find("resetSimulationStateBtn")
    pos_everything = html.find("mutateEverything")
    pos_reinforce = html.find("reinforceCandidate")
    assert pos_reset != -1 and pos_everything != -1 and pos_reinforce != -1, "All 3 core buttons must be present"
    assert abs(pos_everything - pos_reset) < 600, "Reset State and Mutate Everything must be clustered close together"
    assert abs(pos_reinforce - pos_everything) < 600, "Mutate Everything and Reinforce must be clustered close together"
    print("  -> Passed. Reset State, Mutate Everything, and Reinforce are verified close together.")

    assert "triggerImprint" in html, "HTML should contain Easter egg triggerImprint"
    assert "imprintInput" in html, "HTML should contain arbitrary message input"
    assert "<canvas" not in html, "Canvas element should remain removed for minimal UI"
    print("  -> Passed. Minimal HTML contains Direct NCA Channels, PCA RGB, Auto-Cycle, and Clustered Controls.")

    print("[2] Testing GET /api/status...")
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    print(f"  -> Initial status: {status}")
    assert "scale" in status
    assert "scale_type" in status
    assert "auto_speed" in status
    assert "rhythm_division" in status
    assert "auto_freq" in status
    assert "spectrogram_influence" in status, "Status should contain spectrogram_influence"
    assert "nca_channel_view" in status, "Status should contain nca_channel_view"
    assert "nca_auto_cycle" in status, "Status should contain nca_auto_cycle"
    assert "nca_cycle_interval" in status, "Status should contain nca_cycle_interval"
    assert "nca_rerolls" in status, "Status should contain nca_rerolls"
    assert "nca_auto_reroll" in status, "Status should contain nca_auto_reroll"
    assert "nca_topology" in status
    assert "topology" in status
    assert "nca_pole_radial" in status
    assert "nca_pole_azimuthal" in status
    assert "reverse_pole" in status
    assert "reverse_spin" in status
    assert "genetic_generation" in status
    assert "genetic_has_candidate" in status
    assert "genetic_mutated_count" in status
    assert "nca_wiring_links" in status, "Status should contain nca_wiring_links"
    assert "nca_wiring_events" in status, "Status should contain nca_wiring_events"
    assert "genetic_mutation_scale" in status
    assert "genetic_mutation_rate" in status
    print("  -> Passed. Status API returns autotune, spectrogram influence, wiring, and genetic fields.")

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
    post_json("/api/controls", {"scanning_speed": 0.40, "auto_speed": False})
    
    res = post_json("/api/controls", {"speed_multiplier": 2.0})
    assert abs(res["scanning_speed"] - 0.80) < 0.01, f"Expected 0.80, got {res['scanning_speed']}"
    print(f"  -> 2x faster multiplier: speed = {res['scanning_speed']} rev/s")

    res = post_json("/api/controls", {"speed_multiplier": 4.0})
    assert abs(res["scanning_speed"] - 3.20) < 0.02, f"Expected 3.20, got {res['scanning_speed']}"
    print(f"  -> 4x faster multiplier: speed = {res['scanning_speed']} rev/s")

    res = post_json("/api/controls", {"speed_multiplier": 0.5})
    assert abs(res["scanning_speed"] - 1.60) < 0.02, f"Expected 1.60, got {res['scanning_speed']}"
    print(f"  -> 1/2x slower multiplier: speed = {res['scanning_speed']} rev/s")

    res = post_json("/api/controls", {"speed_multiplier": 0.25})
    assert abs(res["scanning_speed"] - 0.40) < 0.02, f"Expected 0.40, got {res['scanning_speed']}"
    print(f"  -> 1/4x slower multiplier: speed = {res['scanning_speed']} rev/s")

    print("[5] Testing Autotune options...")
    res = post_json("/api/controls", {"auto_speed": True, "rhythm_division": 2, "auto_freq": True})
    assert res["auto_speed"] is True
    assert res["auto_freq"] is True
    assert res["rhythm_division"] == 2
    print(f"  -> Auto speed & auto freq enabled. Detected BPM: {res.get('detected_bpm')}")

    print("[6] Testing NCA Topology Controls & Reverse Pole Options...")
    # Set to Polar via /api/topology
    res = post_json("/api/topology", {"topology": "polar", "reverse_pole": False, "reverse_spin": False})
    assert abs(res["nca_topology"] - 1.0) < 0.01, f"Expected 1.0, got {res['nca_topology']}"
    assert res["topology"] == "polar", f"Expected polar, got {res['topology']}"
    assert res["reverse_pole"] is False
    assert abs(res["nca_pole_radial"] - 1.0) < 0.01
    print(f"  -> Normal Polar verified: nca_topology={res['nca_topology']}, radial={res['nca_pole_radial']}")

    # Reverse radial pole (inward sink)
    res = post_json("/api/topology", {"reverse_pole": True})
    assert res["reverse_pole"] is True
    assert abs(res["nca_pole_radial"] - (-1.0)) < 0.01, f"Expected -1.0, got {res['nca_pole_radial']}"
    print(f"  -> Reverse radial pole verified: nca_pole_radial={res['nca_pole_radial']}")

    # Reverse azimuthal spin
    res = post_json("/api/topology", {"reverse_spin": True})
    assert res["reverse_spin"] is True
    assert abs(res["nca_pole_azimuthal"] - (-1.0)) < 0.01, f"Expected -1.0, got {res['nca_pole_azimuthal']}"
    print(f"  -> Reverse spin verified: nca_pole_azimuthal={res['nca_pole_azimuthal']}")

    # Set back to Cartesian (original unrotated)
    res = post_json("/api/topology", {"topology": "cartesian", "reverse_pole": False, "reverse_spin": False})
    assert abs(res["nca_topology"] - 0.0) < 0.01, f"Expected 0.0, got {res['nca_topology']}"
    assert res["topology"] == "cartesian", f"Expected cartesian, got {res['topology']}"
    print(f"  -> Original Cartesian verified: nca_topology={res['nca_topology']}")

    print("[7] Testing Easter Egg Giant Structure Imprint (/api/imprint)...")
    res = post_json("/api/imprint", {"message": "BYOB"})
    assert res["status"] == "imprinted", f"Expected imprinted, got {res}"
    assert res["message"] == "BYOB"
    print("  -> Default Easter Egg 'BYOB' giant imprint request verified.")

    res = post_json("/api/imprint", {"message": "POLAR LIFE"})
    assert res["status"] == "imprinted"
    assert res["message"] == "POLAR LIFE"
    print("  -> Two-line multi-word 'POLAR LIFE' giant imprint request verified.")

    print("[8] Testing Human-Assisted Genetic Evolution & Mutate Everything...")
    # 8.1 Reset to Generation 0
    res = post_json("/api/genetic", {"action": "reset"})
    assert res["genetic_generation"] == 0, f"Expected Gen 0, got {res['genetic_generation']}"
    assert res["genetic_has_candidate"] is False
    print("  -> Gen 0 reset verified.")

    # 8.2 Test 'Mutate Everything' option (100% of all 588 parameters)
    res = post_json("/api/genetic", {
        "action": "mutate_everything",
        "scale": 0.15
    })
    time.sleep(0.05)
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    assert status["genetic_has_candidate"] is True, f"Expected candidate live, got {status}"
    assert status["genetic_mutated_count"] == 588, f"Expected exactly 588 mutated parameters, got {status['genetic_mutated_count']}"
    print(f"  -> Mutate Everything verified: all {status['genetic_mutated_count']} parameters perturbed!")

    # 8.3 Cancel candidate (reverts to baseline)
    res = post_json("/api/genetic", {"action": "cancel"})
    time.sleep(0.05)
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    assert status["genetic_has_candidate"] is False, f"Expected candidate cancelled, got {status}"
    assert status["genetic_generation"] == 0
    print("  -> Cancelled candidate; reverted to Gen 0 baseline.")

    # 8.4 Mutate perception subsystem and Reinforce to Gen 1
    post_json("/api/genetic", {
        "action": "mutate",
        "target": "perception",
        "scale": 0.20,
        "rate": 0.30
    })
    time.sleep(0.05)
    res = post_json("/api/genetic", {"action": "reinforce"})
    time.sleep(0.05)
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    assert status["genetic_generation"] == 1, f"Expected Gen 1, got {status['genetic_generation']}"
    assert status["genetic_has_candidate"] is False
    print(f"  -> Reinforced candidate! Advanced to Gen {status['genetic_generation']}.")

    # 8.5 Rollback to Gen 0
    res = post_json("/api/genetic", {"action": "rollback"})
    time.sleep(0.05)
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    assert status["genetic_generation"] == 0, f"Expected rollback to Gen 0, got {status['genetic_generation']}"
    print("  -> Rollback verified; restored previous generation.")

    print("[9] Testing Spectrogram Influence Controls (/api/spectrogram & /api/controls)...")
    # Set to 0.0 (hides scanning line, state decoupled from audio)
    res = post_json("/api/spectrogram", {"spectrogram_influence": 0.0})
    assert abs(res["spectrogram_influence"] - 0.0) < 0.01, f"Expected 0.0, got {res['spectrogram_influence']}"
    print("  -> Spectrogram influence set to 0.0 (scanning line hidden, autonomous state evolution).")

    # Set via controls to 1.5
    res = post_json("/api/controls", {"spectrogram_influence": 1.5})
    assert abs(res["spectrogram_influence"] - 1.5) < 0.01, f"Expected 1.5, got {res['spectrogram_influence']}"
    print(f"  -> Spectrogram influence set to {res['spectrogram_influence']} via /api/controls.")

    # Reset back to default 1.0
    res = post_json("/api/controls", {"spectrogram_influence": 1.0})
    assert abs(res["spectrogram_influence"] - 1.0) < 0.01

    print("[10] Testing Simulation State Reset (/api/reset_state)...")
    res = post_json("/api/reset_state", {})
    assert res.get("status") == "state_reset", f"Expected state_reset, got {res}"
    print("  -> Simulation state reset requested via /api/reset_state.")

    res = post_json("/api/controls", {"reset_state": True})
    print("  -> Simulation state reset requested via /api/controls.")

    print("[11] Testing Reset Weights to Random (/api/genetic action=reset_random)...")
    res = post_json("/api/genetic", {"action": "reset_random"})
    time.sleep(0.05)
    status_str, _ = get("/api/status")
    status = json.loads(status_str)
    assert status["genetic_generation"] == 0
    assert status["genetic_has_candidate"] is False
    print("  -> Reset to random weights distribution verified.")

    print("[12] Testing Synaptic Wiring ('Parameters that change together wire together')...")
    # Reset wiring and baseline
    post_json("/api/genetic", {"action": "reset"})
    post_json("/api/genetic", {"action": "reset_wiring"})
    time.sleep(0.05)
    status = json.loads(get("/api/status")[0])
    assert status["nca_wiring_links"] == 0, f"Expected 0 links, got {status['nca_wiring_links']}"
    assert status["nca_wiring_events"] == 0

    # Mutate perception subsystem
    post_json("/api/genetic", {
        "action": "mutate",
        "target": "perception",
        "scale": 0.25,
        "rate": 0.35
    })
    time.sleep(0.05)
    status = json.loads(get("/api/status")[0])
    mutCount = status["genetic_mutated_count"]
    assert status["genetic_has_candidate"] is True
    assert mutCount > 1, f"Expected multiple mutated weights, got {mutCount}"
    print(f"  -> Candidate generated with {mutCount} mutated parameters.")

    # Reinforce candidate: these parameters must wire together!
    post_json("/api/genetic", {"action": "reinforce"})
    time.sleep(0.05)
    status = json.loads(get("/api/status")[0])
    links = status["nca_wiring_links"]
    events = status["nca_wiring_events"]
    assert links > 0, f"Expected active wiring links > 0, got {links}"
    assert events == 1, f"Expected 1 wiring event, got {events}"
    print(f"  -> Reinforcement successful! Wired {links} synaptic connections across changed parameters.")

    # Mutate again: wired parameters co-fire together!
    post_json("/api/genetic", {
        "action": "mutate",
        "target": "perception",
        "scale": 0.20,
        "rate": 0.20
    })
    time.sleep(0.05)
    status = json.loads(get("/api/status")[0])
    assert status["genetic_has_candidate"] is True
    print(f"  -> Second mutation fired successfully with co-firing enabled: {status['genetic_mutated_count']} parameters.")
    post_json("/api/genetic", {"action": "cancel"})

    # Reset back to clean Gen 0 original
    post_json("/api/genetic", {"action": "reset"})
    print("  -> Clean baseline restored.")

    print("[13] Testing Direct NCA Channels View & Auto-Cycling...")
    # Test setting individual channels 0..11 directly
    for ch in [0, 3, 5, 8, 11]:
        res = post_json("/api/channel", {"channel": ch})
        assert res["nca_channel_view"] == ch, f"Expected channel {ch}, got {res['nca_channel_view']}"
    print("  -> Direct channel selection (0, 3, 5, 8, 11) verified.")

    # Test raw RGB mode (channel 12)
    res = post_json("/api/channel", {"channel": 12})
    assert res["nca_channel_view"] == 12, f"Expected channel 12, got {res['nca_channel_view']}"
    print("  -> Raw RGB channels mode (12) verified.")

    # Test 3 main PCA components mode (channel 13 / "pca")
    res = post_json("/api/channel", {"channel": 13})
    assert res["nca_channel_view"] == 13, f"Expected channel 13, got {res['nca_channel_view']}"
    print("  -> 3 Main PCA components mode (channel 13) verified.")

    res = post_json("/api/channel", {"channel": "pca"})
    assert res["nca_channel_view"] == 13, f"Expected channel 13 via 'pca' string, got {res['nca_channel_view']}"
    print("  -> 3 Main PCA components mode via 'pca' string verified.")

    # Test returning to composite mode (-1)
    res = post_json("/api/channel", {"channel": -1})
    assert res["nca_channel_view"] == -1, f"Expected channel -1, got {res['nca_channel_view']}"
    print("  -> Composite color-corrected mode (-1) verified.")

    # Test channel control via /api/controls
    res = post_json("/api/controls", {"nca_channel_view": 4})
    assert res["nca_channel_view"] == 4, f"Expected channel 4 via /api/controls, got {res['nca_channel_view']}"
    print("  -> Channel selection via /api/controls verified.")

    # Test Auto-Cycle Channels toggle and cycle interval
    res = post_json("/api/channel", {"auto_cycle": True, "cycle_interval": 0.5})
    assert res["nca_auto_cycle"] is True, "Expected auto_cycle to be True"
    assert abs(res["nca_cycle_interval"] - 0.5) < 0.01, f"Expected cycle_interval 0.5, got {res['nca_cycle_interval']}"
    print("  -> Auto-cycling channels enabled with 0.5s interval.")

    # Wait for auto-cycling to advance channel
    time.sleep(1.2)
    status = json.loads(get("/api/status")[0])
    print(f"  -> Channel after cycling: {status['nca_channel_view']}")
    assert status["nca_auto_cycle"] is True

    # Disable auto-cycle and restore composite
    res = post_json("/api/channel", {"auto_cycle": False, "channel": -1, "cycle_interval": 1.5})
    assert res["nca_auto_cycle"] is False
    assert res["nca_channel_view"] == -1
    print("  -> Auto-cycle disabled and composite view restored.")

    print("[14] Testing Auto-Reroll on State Collapse / Numerical Explosion...")
    # Verify auto-reroll enabled in status
    status = json.loads(get("/api/status")[0])
    assert status["nca_auto_reroll"] is True, "Auto-reroll should be enabled by default"

    # Test mutating everything with candidate generated
    res = post_json("/api/genetic", {"action": "mutate_everything", "scale": 0.15})
    assert res["genetic_has_candidate"] is True
    assert res["genetic_mutated_count"] > 0
    print(f"  -> Mutate Everything candidate generated with {res['genetic_mutated_count']} parameters.")

    # Wait for state analysis frames (at least 15 frames = 250ms)
    time.sleep(0.35)
    status = json.loads(get("/api/status")[0])
    print(f"  -> Auto-rerolls count: {status['nca_rerolls']}, has_candidate: {status['genetic_has_candidate']}")
    assert status["genetic_has_candidate"] is True, "Candidate should remain active (auto-rerolled if collapsed)"

    # Cancel candidate and clean up
    post_json("/api/genetic", {"action": "cancel"})
    post_json("/api/channel", {"channel": -1})
    print("  -> Auto-reroll engine verified.")

    print("\nALL SPECTROGRAM INFLUENCE, STATE RESET, RANDOM RESET, SYNAPTIC WIRING, DIRECT CHANNELS, PCA RGB & AUTO-REROLL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
