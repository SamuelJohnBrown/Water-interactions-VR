#include "water_coll_det.h"
#include "helper.h"
#include "config.h"
#include <thread>
#include <chrono>
#include <atomic>
#include <SKSE/SKSE.h>
#include <RE/Skyrim.h>
#include <cmath>
#include <algorithm>
#include <functional>
#include <deque>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#undef max
#undef min

// NOTE: enable real logging (use IW_LOG_* macros and AppendToPluginLog)
//#define AppendToPluginLog(...) do { (void)0; } while(0)

namespace InteractiveWaterVR {

static std::atomic<bool> s_running{false};
static std::thread s_thread;
// Persisted prev-moving flags used for transition logging; reset on start/stop
static std::atomic<bool> s_prevLeftMoving{false};
static std::atomic<bool> s_prevRightMoving{false};

// Per-controller ripple emission state
static std::atomic<bool> s_leftRippleEmitted{false};
static std::atomic<bool> s_rightRippleEmitted{false};

// Per-controller last ripple time (for adaptive scaling)
static std::chrono::steady_clock::time_point s_leftLastRippleTime{};
static std::chrono::steady_clock::time_point s_rightLastRippleTime{};
// Per-controller last wake scheduling time (ms since epoch)
static std::atomic<long long> s_leftLastWakeMs{0};
static std::atomic<long long> s_rightLastWakeMs{0};
// Per-controller last entry-splash sound start time (ms since epoch)
static std::atomic<long long> s_leftLastEntrySoundMs{0};
static std::atomic<long long> s_rightLastEntrySoundMs{0};
// Window (ms) during which an exit sound must be suppressed if an entry sound just played
constexpr long long kEntrySoundGuardMs =1500;
// Wake-move sound played once when hand starts moving underwater after delay
static RE::BGSSoundDescriptorForm* s_wakeMoveSoundDesc = nullptr;
static std::atomic<uint32_t> s_leftWakeMoveSoundHandle{0};
static std::atomic<uint32_t> s_rightWakeMoveSoundHandle{0};
static std::atomic<long long> s_leftLastWakeMoveMs{0};
static std::atomic<long long> s_rightLastWakeMoveMs{0};

// Hardcoded maximum speeds (m/s) beyond which entries/exits are ignored - this prevents phantom splashes in bodies of water
constexpr float kMaxEntryDownSpeed =1500.0f; // ignore entry impacts faster than this (lowered)
constexpr float kMaxExitUpSpeed =900.0f; // ignore exit impacts faster than this (lowered)

// Polling interval (ms) - lowered to sample closer to higher frame rates (e.g.90Hz VR ~11ms)
// Reduced to6 ms to improve responsiveness for fast controller motion (tradeoff: higher CPU).
constexpr int kPollIntervalMs =6;

// Movement detection thresholds (m/s)
constexpr float kStationaryThreshold =0.02f; // below => consider stationary (instant threshold)
constexpr float kMovingThreshold =0.05f; // above => consider moving (instant)
constexpr float kJitterThreshold =0.01f; // tiny jitter to update last movement time
constexpr float kMaxValidSpeed =50.0f; // ignore implausible spikes
constexpr float kStationaryConfirmSeconds =2.0f; // require this much time without movement to mark stationary

// Flag set while a game load is in progress to avoid using engine objects.
static std::atomic<bool> s_gameLoadInProgress{false};

// Persist player swimming state for transition logging
static std::atomic<bool> s_prevPlayerSwimming{false};

// Player depth logging state
static std::atomic<long long> s_lastPlayerDepthLogMs{0};
static std::atomic<float> s_prevPlayerDepth{0.0f};
constexpr long long kPlayerDepthLogIntervalMs =1000; // log at most once per second (or on significant change)
constexpr float kPlayerDepthLogDelta =0.05f; // meters change to force log

// Per-controller last measured depth (meters) when submerged
static std::atomic<float> s_leftControllerDepth{0.0f};
static std::atomic<float> s_rightControllerDepth{0.0f};

// Notify functions
void NotifyGameLoadStart() { s_gameLoadInProgress.store(true); }
void NotifyGameLoadEnd() { s_gameLoadInProgress.store(false); }

bool IsGameLoadInProgress() { return s_gameLoadInProgress.load(); }

// Movement state (true == moving)
bool leftMoving = false;
bool rightMoving = false;

// Last time any meaningful movement was seen (used to confirm stationary after delay)
auto leftLastMovementTime = std::chrono::steady_clock::now();
auto rightLastMovementTime = std::chrono::steady_clock::now();

// Movement start candidate timers (require sustained movement before reporting 'started moving')
std::chrono::steady_clock::time_point leftMovementCandidateTime{};
std::chrono::steady_clock::time_point rightMovementCandidateTime{};

 // Submerged state: set when controller enters water, cleared on exit
 static std::atomic<bool> leftSubmerged{false};
 static std::atomic<bool> rightSubmerged{false};
 // Transition timestamps (ms since steady_clock epoch) used to allow forced entry/exit ripples
 static std::atomic<long long> lastLeftTransitionMs{0};
 static std::atomic<long long> lastRightTransitionMs{0};
 // Per-controller submerged start times (ms since epoch) used to decide sound suppression
 static std::atomic<long long> leftSubmergedStartMs{0};
 static std::atomic<long long> rightSubmergedStartMs{0};
 // Window (ms) after transition during which a forced ripple is allowed
 constexpr long long kForcedRippleWindowMs =250;

// Movement thresholds are configurable via Data\SKSE\Plugins\Interactive_Water_VR.ini (see src/config.cpp)

static RE::NiAVObject* GetPlayerHandNode(bool rightHand) {
	auto player = RE::PlayerCharacter::GetSingleton();
	if (!player) return nullptr;
	auto root = player->Get3D();
	if (!root) return nullptr;

#if defined(ENABLE_SKYRIM_VR)
	const char* nodeName = rightHand ? "NPC R Hand [RHnd]" : "NPC L Hand [LHnd]";
#else
	const char* nodeName = nullptr;
#endif
	if (nodeName) {
		return root->GetObjectByName(nodeName);
	}
	return root;
}

static RE::NiPoint3 GetControllerWorldPosition(bool rightHand) {
	auto handNode = GetPlayerHandNode(rightHand);
	if (handNode) return handNode->world.translate;
	return {0.0f,0.0f,0.0f};
}

static float VecLen(const RE::NiPoint3& v) {
	return std::sqrt(v.x*v.x + v.y*v.y + v.z*v.z);
}
static RE::NiPoint3 Normalize(const RE::NiPoint3& v) {
	float l = VecLen(v);
	if (l <=1e-6f) return {0.0f,1.0f,0.0f};
	return {v.x / l, v.y / l, v.z / l};
}

static RE::NiPoint3 GetControllerForward(bool rightHand) {
	auto node = GetPlayerHandNode(rightHand);
	if (!node) return {0.0f,1.0f,0.0f};
	// assume Y axis is forward for controller model
	RE::NiPoint3 localForward{0.0f,1.0f,0.0f};
	RE::NiPoint3 worldF = node->world.rotate * localForward;
	return Normalize(worldF);
}

// New: compute splash amplitude from downSpeed (magnitude units used by estimator)
static float ComputeEntrySplashAmount(float downSpeed) {
	// Very small impacts produce no visible ripple
	if (downSpeed <=0.1f) {
		return 0.0f;
	}

	// Use configurable band thresholds/amplitudes from config.cpp (now including a 'light' band)
	float amt =0.0f;
	if (downSpeed <= cfgSplashVeryLightMax) {
		amt = cfgSplashVeryLightAmt;
	} else if (downSpeed <= cfgSplashLightMax) {
		amt = cfgSplashLightAmt;
	} else if (downSpeed <= cfgSplashNormalMax) {
		amt = cfgSplashNormalAmt;
	} else if (downSpeed <= cfgSplashHardMax) {
		amt = cfgSplashHardAmt;
	} else {
		amt = cfgSplashVeryHardAmt;
	}

	// Apply global scale
	amt *= cfgSplashScale;

	return amt;
}

// New: compute splash amplitude for exit (upward) using exit-specific bands (including 'light')
static float ComputeExitSplashAmount(float upSpeed) {
	if (upSpeed <=0.1f) {
		return 0.0f;
	}
	float amt =0.0f;
	if (upSpeed <= cfgSplashExitVeryLightMax) {
		amt = cfgSplashExitVeryLightAmt;
	} else if (upSpeed <= cfgSplashExitLightMax) {
		amt = cfgSplashExitLightAmt;
	} else if (upSpeed <= cfgSplashExitNormalMax) {
		amt = cfgSplashExitNormalAmt;
	} else if (upSpeed <= cfgSplashExitHardMax) {
		amt = cfgSplashExitHardAmt;
	} else {
		amt = cfgSplashExitVeryHardAmt;
	}
	amt *= cfgSplashScale;
	return amt;
}

// --- Sound playback helpers for splash bands ---
enum class SplashBand {
	VeryLight =0,
	Light,
	Normal,
	Hard,
	VeryHard,
	Count
};

// Map user-provided base form IDs (from SpellInteractionsVR.esp) to bands
static const std::uint32_t kSplashFormBaseIDs[static_cast<size_t>(SplashBand::Count)] = {
	0x01000806u, // VeryLight (user provided0X1000806)
	0x01000806u, // Light (same in user's mapping)
	0x01000807u, // Normal/Medium
	0x01000808u, // Hard/Large
	0x01000808u // VeryHard/VeryLarge
};

// Cache loaded descriptors to avoid repeated lookups
static RE::BGSSoundDescriptorForm* s_splashSounds[static_cast<size_t>(SplashBand::Count)] = {nullptr};

static RE::BGSSoundDescriptorForm* LoadSplashSoundDescriptor(SplashBand band) {
	auto idx = static_cast<size_t>(band);
	if (idx >= static_cast<size_t>(SplashBand::Count)) return nullptr;
	if (s_splashSounds[idx]) return s_splashSounds[idx];

	std::uint32_t fullId =0;
	// Use helper to load and log; helper will compose full form id from plugin name
	auto form = LoadFormAndLog<RE::BGSSoundDescriptorForm>("SpellInteractionsVR.esp", fullId, kSplashFormBaseIDs[idx], "SplashSound");
	if (form) {
		s_splashSounds[idx] = form;
	} 
	return s_splashSounds[idx];
}

static SplashBand GetSplashBandForDownSpeed(float downSpeed) {
	if (downSpeed <= cfgSplashVeryLightMax) return SplashBand::VeryLight;
	if (downSpeed <= cfgSplashLightMax) return SplashBand::Light;
	if (downSpeed <= cfgSplashNormalMax) return SplashBand::Normal;
	if (downSpeed <= cfgSplashHardMax) return SplashBand::Hard;
	return SplashBand::VeryHard;
}

static uint32_t PlaySoundAtNode(RE::BGSSoundDescriptorForm* sound, RE::NiAVObject* node, const RE::NiPoint3& location, float volume)
{
	if (!sound) return 0;
	auto audio = RE::BSAudioManager::GetSingleton();
	if (!audio) return 0;

	RE::BSSoundHandle handle;
	// Build sound data from descriptor; pass a reasonable flag value similar to example (16)
	if (!audio->BuildSoundDataFromDescriptor(handle, static_cast<RE::BSISoundDescriptor*>(sound),16)) {
		return 0;
	}
	if (handle.soundID == static_cast<uint32_t>(-1)) {
		return 0;
	}

	// log the requested volume
	handle.SetPosition(location);
	handle.SetObjectToFollow(node);
	// apply per-band volume (SetVolume expects a float)
	handle.SetVolume(volume);
	if (handle.Play()) {
		return handle.soundID;
	}

	return 0;
}

static void PlaySplashSoundForDownSpeed(bool isLeft, float downSpeed) {
	// Choose band
	SplashBand band = GetSplashBandForDownSpeed(downSpeed);
	auto desc = LoadSplashSoundDescriptor(band);
	if (!desc) return;
	// Get hand node for appropriate hand
	auto node = GetPlayerHandNode(!isLeft ? true : false); // left: false, right: true
	if (!node) {
		return;
	}

	// If the controller is already submerged and this is NOT an immediate entry event, suppress sounds.
	// Allow entry sounds only if the last transition for this hand was an 'inWater' within the forced window.
	long long lastTrans = isLeft ? lastLeftTransitionMs.load() : lastRightTransitionMs.load();
	bool justEntered = false;
	if (lastTrans !=0) {
		long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		if ((nowMs - lastTrans) <= kForcedRippleWindowMs) justEntered = true;
	}
	if (!justEntered) {
		// if currently submerged, suppress any splash sounds (this prevents sounds while moving underwater)
		if ((isLeft && leftSubmerged.load()) || (!isLeft && rightSubmerged.load())) {
			return;
		}
	}

	// select volume based on band and config
	float vol =1.0f;
	switch (band) {
		case SplashBand::VeryLight: vol = cfgSplashVeryLightVol; break;
		case SplashBand::Light: vol = cfgSplashLightVol; break;
		case SplashBand::Normal: vol = cfgSplashNormalVol; break;
		case SplashBand::Hard: vol = cfgSplashHardVol; break;
		case SplashBand::VeryHard: vol = cfgSplashVeryHardVol; break;
		default: vol =1.0f; break;
	}
	// clamp to non-negative
	if (vol <0.0f) vol =0.0f;

	uint32_t id = PlaySoundAtNode(desc, node, node->world.translate, vol);
	if (id == 0) {
	} else {
		// record entry-sound start time so exit sounds can be suppressed while this plays
		long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		if (isLeft) s_leftLastEntrySoundMs.store(nowMs);
		else s_rightLastEntrySoundMs.store(nowMs);
	}
}

// Helper that runs on main thread (or fallback) to add ripple (audio intentionally removed)
static void EmitRipple(const RE::NiPoint3& p, float amt, const char* reason = nullptr, const char* hand = "?") {
	// Only emit ripple; audio intentionally removed
	auto ws = RE::TESWaterSystem::GetSingleton();
	if (ws) {
		ws->AddRipple(p, amt);
	}
}

// Emit ripple if allowed for the given hand. If 'force' is true, emit regardless of submerged state.
// Returns true if a ripple was emitted, false if it was suppressed.
static bool EmitRippleIfAllowed(bool isLeft, const RE::NiPoint3& p, float amt, bool force = false, int requireSubmergedState = -1, const char* reason = nullptr) {
 	 // If forced, only allow if within short window after transition; otherwise treat as not forced
 	 if (force) {
 	 	 auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
 	 	 if (isLeft) {
 	 	 	 long long t = lastLeftTransitionMs.load();
 	 	 	 if (nowMs - t > kForcedRippleWindowMs) {
 	 	 	 	 // forced window expired -> treat as not forced
 	 	 	 	 force = false;
 	 	 	 }
 	 	 } else {
 	 	 	 long long t = lastRightTransitionMs.load();
 	 	 	 if (nowMs - t > kForcedRippleWindowMs) {
 	 	 	 	 force = false;
 	 	 	 }
 	 	 }
 	 }

 	 // If a required submerged state is provided, enforce it now
 	 if (requireSubmergedState != -1) {
 	 	 bool cur = isLeft ? leftSubmerged.load() : rightSubmerged.load();
 	 	 if (requireSubmergedState ==1 && !cur) {
 	 	 	 return false;
 	 	 }
 	 	 if (requireSubmergedState ==0 && cur) {
 	 	 	 return false;
 	 	 }
 	 }

 	 // If not forced, suppress ripples when the controller is already submerged (legacy behaviour)
 	 if (!force) {
 	 	 bool cur = isLeft ? leftSubmerged.load() : rightSubmerged.load();
 	 	 if (cur) {
 	 	 	 // always suppress splashes while submerged; wake ripples are handled separately
 	 	 	 return false;
 	 	 }
 	 }

 	 // use hand string for EmitRipple logging
 	 const char* handStr = isLeft ? "L" : "R";
 	 EmitRipple(p, amt, reason, handStr);
 	 return true;
 }

 // Splash-named wrapper that preserves behavior and is used by call sites
 static bool EmitSplashIfAllowed(bool isLeft, const RE::NiPoint3& p, float amt, bool force = false, int requireSubmergedState = -1, const char* reason = nullptr) {
	 return EmitRippleIfAllowed(isLeft, p, amt, force, requireSubmergedState, reason);
 }

// --- Exit sound support ---
// Map exit-band form IDs: default for exits is0x01000810, except hard/very hard use0x0100080E
 static const std::uint32_t kSplashExitFormBaseIDs[static_cast<size_t>(SplashBand::Count)] = {
0x01000810u, // VeryLight
0x01000810u, // Light
0x01000810u, // Normal
0x0100080Eu, // Hard
0x0100080Eu // VeryHard
 };
 
 // Cache exit descriptors
 static RE::BGSSoundDescriptorForm* s_splashExitSounds[static_cast<size_t>(SplashBand::Count)] = { nullptr };

 static RE::BGSSoundDescriptorForm* LoadSplashExitSoundDescriptor(SplashBand band) {
 auto idx = static_cast<size_t>(band);
 if (idx >= static_cast<size_t>(SplashBand::Count)) return nullptr;
 if (s_splashExitSounds[idx]) return s_splashExitSounds[idx];
 std::uint32_t fullId =0;
 auto form = LoadFormAndLog<RE::BGSSoundDescriptorForm>("SpellInteractionsVR.esp", fullId, kSplashExitFormBaseIDs[idx], "SplashExitSound");
 if (form) {
 s_splashExitSounds[idx] = form;
 IW_LOG_INFO("Loaded splash exit sound form for band %u -> fullId=0x%08X", (unsigned)idx, fullId);
 } else {
 IW_LOG_WARN("Failed to load splash exit sound form for band %u (base0x%08X)", (unsigned)idx, kSplashExitFormBaseIDs[idx]);
 }
 return s_splashExitSounds[idx];
 }

 static SplashBand GetExitSplashBandForUpSpeed(float upSpeed) {
 if (upSpeed <= cfgSplashExitVeryLightMax) return SplashBand::VeryLight;
 if (upSpeed <= cfgSplashExitLightMax) return SplashBand::Light;
 if (upSpeed <= cfgSplashExitNormalMax) return SplashBand::Normal;
 if (upSpeed <= cfgSplashExitHardMax) return SplashBand::Hard;
 return SplashBand::VeryHard;
 }

 static void PlayExitSoundForUpSpeed(bool isLeft, float upSpeed) {
 SplashBand band = GetExitSplashBandForUpSpeed(upSpeed);
 auto desc = LoadSplashExitSoundDescriptor(band);
 if (!desc) return;
 auto node = GetPlayerHandNode(isLeft ? false : true);
 if (!node) return;
 float vol =0.2f;
 switch (band) {
 case SplashBand::VeryLight: vol = cfgSplashExitVeryLightVol; break;
 case SplashBand::Light: vol = cfgSplashExitLightVol; break;
 case SplashBand::Normal: vol = cfgSplashExitNormalVol; break;
 case SplashBand::Hard: vol = cfgSplashExitHardVol; break;
 case SplashBand::VeryHard: vol = cfgSplashExitVeryHardVol; break;
 default: vol =0.2f; break;
 }
 if (vol <0.0f) vol =0.0f;

 // Exit sounds should be played when exiting the water. However, ensure we don't accidentally
 // play sounds while the controller remains submerged. If the hand is currently submerged,
 // skip playback.
 if ((isLeft && leftSubmerged.load()) || (!isLeft && rightSubmerged.load())) {
	 return;
 }

 // If a recent entry splash sound was played for this hand, suppress the exit sound to avoid overlap.
 long long lastEntry = isLeft ? s_leftLastEntrySoundMs.load() : s_rightLastEntrySoundMs.load();
 if (lastEntry !=0) {
	 long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	 if ((nowMs - lastEntry) < kEntrySoundGuardMs) {
		 return;
	 }
 }

 PlaySoundAtNode(desc, node, node->world.translate, vol);
}

// Definition: attempt to play the wake-move sound (once per submerged session)
static bool TryPlayWakeMoveSound(bool isLeft) {
 // Load descriptor if needed
 if (!s_wakeMoveSoundDesc) {
 std::uint32_t fullId =0;
 auto form = LoadFormAndLog<RE::BGSSoundDescriptorForm>("SpellInteractionsVR.esp", fullId,0x01000809u, "WakeMoveSound");
 if (!form) {
 IW_LOG_WARN("TryPlayWakeMoveSound: failed to load wake-move sound form (base0x01000809)");
 return false;
 }
 s_wakeMoveSoundDesc = form;
 IW_LOG_INFO("TryPlayWakeMoveSound: loaded wake-move sound form -> fullId=0x%08X", fullId);
 }

 // NOTE: per request, this sound is immune to guards. Play whenever called and hand node exists.
 auto node = GetPlayerHandNode(isLeft ? false : true);
 if (!node) return false;

 float vol = cfgWakeMoveSoundVol;
 uint32_t id = PlaySoundAtNode(s_wakeMoveSoundDesc, node, node->world.translate, vol);
 if (id ==0) return false;

 // store handle id for potential later stop/cleanup on exit
 if (isLeft) s_leftWakeMoveSoundHandle.store(id);
 else s_rightWakeMoveSoundHandle.store(id);

 return true;
}

static bool IsPointInWater(const RE::NiPoint3& a_pos, float& outWaterHeight) {
 outWaterHeight =0.0f;
 // rate-limit noisy logs to once every N seconds
 static auto s_lastIsPointLog = std::chrono::steady_clock::now() - std::chrono::seconds(10);
 constexpr auto kLogInterval = std::chrono::seconds(2);

 auto nowLog = std::chrono::steady_clock::now();

 auto tes = RE::TES::GetSingleton();
 if (!tes) {
 if (nowLog - s_lastIsPointLog > kLogInterval) {
 s_lastIsPointLog = nowLog;
 }
 return false;
 }
 auto cell = tes->GetCell(a_pos);
 if (!cell) {
 if (nowLog - s_lastIsPointLog > kLogInterval) {
 s_lastIsPointLog = nowLog;
 }
 return false;
 }
 float wh =0.0f;
 if (cell->GetWaterHeight(a_pos, wh)) {
 outWaterHeight = wh;
 if (!std::isfinite(outWaterHeight)) {
 if (nowLog - s_lastIsPointLog > kLogInterval) {
 s_lastIsPointLog = nowLog;
 }
 return false;
 }
 constexpr float kThreshold =0.02f;
 bool inWater = (outWaterHeight - a_pos.z) > kThreshold;
 return inWater;
 }
 if (nowLog - s_lastIsPointLog > kLogInterval) {
 s_lastIsPointLog = nowLog;
 }
 return false;
}

static void LogWaterDetailsAtPosition(const RE::NiPoint3& a_pos) {
 // Intentionally no-op: detailed water logging removed to avoid excessive log spam.
 (void)a_pos;
}

// forward declaration for wake ripple helper
static void EmitWakeRipple(bool isLeft, const RE::NiPoint3& p, float amt);
// forward declaration for wake-move sound helper
static bool TryPlayWakeMoveSound(bool isLeft);

void MonitoringThread() {
	// call loadConfig initially
	loadConfig();

	// Reset persistent transition flags on thread start
	// (these are static inside the loop; ensure they start false)
	// Note: prev flags declared later as static within the loop; reset global atomics if present

	bool lastLeftInWater = false;
	bool lastRightInWater = false;

	RE::NiPoint3 prevLeftPos{0.0f,0.0f,0.0f};
	RE::NiPoint3 prevRightPos{0.0f,0.0f,0.0f};

	// ring buffer sample for velocity estimation
struct Sample { RE::NiPoint3 pos; RE::NiPoint3 forward; std::chrono::steady_clock::time_point t; };
std::deque<Sample> leftSamples;
std::deque<Sample> rightSamples;
std::deque<Sample> playerSamples; // samples for player speed estimation
	
	float prevLeftWaterHeight =0.0f;
	float prevRightWaterHeight =0.0f;
	auto prevLeftTime = std::chrono::steady_clock::now();
	auto prevRightTime = prevLeftTime;
	bool havePrevLeft = false;
	bool havePrevRight = false;

	// Movement state (true == moving)
	bool leftMoving = false;
 bool rightMoving = false;

	// recent speeds for logging
	float recentLeftSpeed =0.0f;
	float recentRightSpeed =0.0f;
 float recentPlayerSpeed =0.0f; // track player speed

 // Player speed log rate control
 long long lastPlayerSpeedLogMs =0;
 float prevLoggedPlayerSpeed =0.0f;
 constexpr long long kPlayerSpeedLogIntervalMs =500; // log at most twice per second
 constexpr float kPlayerSpeedLogDelta =0.1f; // log if speed changes by this much (m/s)

	// Last time any meaningful movement was seen (used to confirm stationary after delay)
	auto leftLastMovementTime = std::chrono::steady_clock::now();
	auto rightLastMovementTime = std::chrono::steady_clock::now();

	// Movement start candidate timers (require sustained movement before reporting 'started moving')
	std::chrono::steady_clock::time_point leftMovementCandidateTime{};
	std::chrono::steady_clock::time_point rightMovementCandidateTime{};

	// Submerged state: set when controller enters water, cleared on exit
	// Use file-scope submerged flags (`leftSubmerged` / `rightSubmerged`) so helper functions
	// (EmitRippleIfAllowed) can observe the current submerged state. Do NOT redeclare locals here.

	while (s_running.load(std::memory_order_acquire)) {
		try {
			// hot-reload config each loop to ensure INI hotload on load game
			loadConfig();

			// If player or world isn't ready (during load), wait and skip this iteration
			auto player = RE::PlayerCharacter::GetSingleton();
			if (!player) {
				std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
				continue;
			}
			auto root = player->Get3D();
			if (!root) {
				std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
				continue;
			}

			// If a game load is in progress, skip sampling and emission to avoid accessing torn-down engine objects
			if (s_gameLoadInProgress.load()) {
				std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
				continue;
			}

			// If the UI is in menu mode or game is paused, wait until menus are closed or game unpaused
			auto ui = RE::UI::GetSingleton();
			if (ui) {
				// Block here until menus are gone or game is unpaused. Use a short sleep to be responsive.
				while (s_running.load(std::memory_order_acquire) && (ui->GameIsPaused() || ui->IsShowingMenus())) {
					std::this_thread::sleep_for(std::chrono::milliseconds(25));
					ui = RE::UI::GetSingleton();
				}
				// If monitoring was stopped while waiting, break out of this iteration
				if (!s_running.load(std::memory_order_acquire)) break;
			}
			
			// proceed with controller sampling

			auto leftPos = GetControllerWorldPosition(false);
			auto rightPos = GetControllerWorldPosition(true);
			// capture player position early for sampling and later use
	auto playerPos = root->world.translate;

			// capture forward directions and timestamp, push into ring buffers for velocity estimation
			RE::NiPoint3 leftForward = GetControllerForward(false);
			RE::NiPoint3 rightForward = GetControllerForward(true);
			auto sampleTime = std::chrono::steady_clock::now();
			leftSamples.push_back(Sample{leftPos, leftForward, sampleTime});
			rightSamples.push_back(Sample{rightPos, rightForward, sampleTime});
			playerSamples.push_back(Sample{playerPos, RE::NiPoint3{0.0f,0.0f,0.0f}, sampleTime});
			constexpr size_t kMaxSamples =7;
			if (leftSamples.size() > kMaxSamples) leftSamples.pop_front();
			if (rightSamples.size() > kMaxSamples) rightSamples.pop_front();
			if (playerSamples.size() > kMaxSamples) playerSamples.pop_front();

			// compute player speed from last two player samples
			if (playerSamples.size() >=2) {
				const auto &sPrev = playerSamples[playerSamples.size() -2];
				const auto &sCur = playerSamples[playerSamples.size() -1];
				double ds = std::chrono::duration<double>(sCur.t - sPrev.t).count();
				if (ds >1e-6) {
					float dx = sCur.pos.x - sPrev.pos.x;
					float dy = sCur.pos.y - sPrev.pos.y;
					float dz = sCur.pos.z - sPrev.pos.z;
					recentPlayerSpeed = std::sqrt(dx * dx + dy * dy + dz * dz) / (float)ds;
					// rate-limited logging
					long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
					if ((nowMs - lastPlayerSpeedLogMs) >= kPlayerSpeedLogIntervalMs || std::fabs(recentPlayerSpeed - prevLoggedPlayerSpeed) >= kPlayerSpeedLogDelta) {
						// Replace verbose player speed logs with water data collection/summaries
						prevLoggedPlayerSpeed = recentPlayerSpeed;
						lastPlayerSpeedLogMs = nowMs;
						auto taskIntfDbg = SKSE::GetTaskInterface();
						if (taskIntfDbg) {
							RE::NiPoint3 pPosCopy = playerPos; // capture
							std::function<void()> dbgFn = [pPosCopy]() {
								if (s_gameLoadInProgress.load()) return;
								LogWaterDetailsAtPosition(pPosCopy);
							};
							taskIntfDbg->AddTask(dbgFn);
						} else {
							if (!s_gameLoadInProgress.load()) {
								LogWaterDetailsAtPosition(playerPos);
							}
						}
					}

					// If player is moving too fast, suspend all water detection until they slow down
					constexpr float kPlayerSpeedShutdown =220.0f; // m/s
					if (recentPlayerSpeed > kPlayerSpeedShutdown) {
						// Clear submerged flags and transition timestamps
						leftSubmerged.store(false);
						rightSubmerged.store(false);
						lastLeftTransitionMs.store(0);
						lastRightTransitionMs.store(0);
						leftSubmergedStartMs.store(0);
						rightSubmergedStartMs.store(0);

						// Reset emission guards and moving-state logs
						s_leftRippleEmitted.store(false);
					 s_rightRippleEmitted.store(false);
					 s_prevLeftMoving.store(false);
					 s_prevRightMoving.store(false);

						// Clear sample buffers used for velocity estimation and mark we have no previous samples
						leftSamples.clear();
						rightSamples.clear();
						playerSamples.clear();
						havePrevLeft = false;
						havePrevRight = false;

						// Reset previous-in-water flags so any pending entry/exit logic is abandoned
						lastLeftInWater = false;
						lastRightInWater = false;

						// Reset recent timers to avoid immediate emissions when player slows
						s_leftLastRippleTime = std::chrono::steady_clock::time_point{};
					 s_rightLastRippleTime = std::chrono::steady_clock::time_point{};

						// Yield the iteration to stop processing immediately
						std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
						continue;
					}
				}
			}

			// determine water heights and in-water flags early (used later)
			float leftWaterHeight =0.0f;
			float rightWaterHeight =0.0f;
			bool leftInWater = IsPointInWater(leftPos, leftWaterHeight);
			bool rightInWater = IsPointInWater(rightPos, rightWaterHeight);

			// If the engine has no active TESWaterForm for the current water system, treat as no-water
			// and suspend all water-detection for this iteration. This prevents ripples/sounds when
			// the water type is undefined (currentWaterType == nullptr).
			auto waterSystemCheck = RE::TESWaterSystem::GetSingleton();
			if (waterSystemCheck && !waterSystemCheck->currentWaterType) {
				// clear global submerged flags so helpers see not-submerged
				leftSubmerged.store(false);
				rightSubmerged.store(false);

				// clear transition timestamps and start times
				lastLeftTransitionMs.store(0);
				lastRightTransitionMs.store(0);
				leftSubmergedStartMs.store(0);
				rightSubmergedStartMs.store(0);

				// reset emission guards and moving-state logs
				s_leftRippleEmitted.store(false);
				s_rightRippleEmitted.store(false);
				s_prevLeftMoving.store(false);
				s_prevRightMoving.store(false);

				// clear sample buffers and reset prev-sample flags
				leftSamples.clear();
				rightSamples.clear();
				playerSamples.clear();
				havePrevLeft = false;
				havePrevRight = false;

				// reset last-in-water flags
				lastLeftInWater = false;
				lastRightInWater = false;

				// reset recent timers
				s_leftLastRippleTime = std::chrono::steady_clock::time_point{};
				s_rightLastRippleTime = std::chrono::steady_clock::time_point{};

				// skip the rest of this monitoring iteration
				std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
				continue;
			}

			// Update controller depth atoms (meters). Zero when not in water.
			float leftControllerDepth =0.0f;
			float rightControllerDepth =0.0f;
			if (leftInWater) {
				leftControllerDepth = leftWaterHeight - leftPos.z;
				if (leftControllerDepth <0.0f) leftControllerDepth =0.0f;
			}
			if (rightInWater) {
				rightControllerDepth = rightWaterHeight - rightPos.z;
				if (rightControllerDepth <0.0f) rightControllerDepth =0.0f;
			}
			s_leftControllerDepth.store(leftControllerDepth);
		 s_rightControllerDepth.store(rightControllerDepth);

			// Per-controller depth shutdown: if a controller goes too deep, suspend detection for that hand only.
			constexpr float kControllerDepthShutdown =15.0f; // meters
			if (leftControllerDepth >= kControllerDepthShutdown) {
				// Treat as not-in-water for this iteration and clear left-specific transient state
				leftInWater = false;
				lastLeftInWater = false;
				leftSubmerged.store(false);
				lastLeftTransitionMs.store(0);
				leftSubmergedStartMs.store(0);
				// Reset left emission/gesture state
			 s_leftRippleEmitted.store(false);
 s_prevLeftMoving.store(false);
				// clear left samples and mark no previous
				leftSamples.clear();
				havePrevLeft = false;
				// reset left wake timers
				s_leftLastWakeMs.store(0);
				s_leftLastEntrySoundMs.store(0);
				s_leftLastWakeMoveMs.store(0);
				s_leftLastRippleTime = std::chrono::steady_clock::time_point{};
			}
			if (rightControllerDepth >= kControllerDepthShutdown) {
				// Treat as not-in-water for this iteration and clear right-specific transient state
				rightInWater = false;
				lastRightInWater = false;
				rightSubmerged.store(false);
				lastRightTransitionMs.store(0);
				rightSubmergedStartMs.store(0);
				// Reset right emission/gesture state
			 s_rightRippleEmitted.store(false);
 s_prevRightMoving.store(false);
				// clear right samples and mark no previous
				rightSamples.clear();
				havePrevRight = false;
				// reset right wake timers
			 s_rightLastWakeMs.store(0);
 s_rightLastEntrySoundMs.store(0);
 s_rightLastWakeMoveMs.store(0);
 s_rightLastRippleTime = std::chrono::steady_clock::time_point{};
			}

			// Update transition timestamps (ms since steady_clock epoch) so forced entry/exit ripples
 			// are permitted only within a short window after transition. Do NOT flip the file-scope
 			// submerged flags here — postpone until after emission logic so queued main-thread tasks
 			// that may execute outside the forced window are not incorrectly suppressed by the
 			// "already submerged" check.
		 if (leftInWater && !lastLeftInWater) {
			 long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
			 lastLeftTransitionMs.store(nowMs);
			 // mark submerged start time
			 leftSubmergedStartMs.store(nowMs);
			 // ignore tiny Z jitter transitions: require previous pos was sufficiently above surface to count as entry
			 if (!(prevLeftPos.z - leftWaterHeight > cfgMinZDiffForEntryExit)) {
			 } else {
			 }
 }
 if (!leftInWater && lastLeftInWater) {
			 long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
			 lastLeftTransitionMs.store(nowMs);
			 // clear submerged start (we are exiting)
			 long long start = leftSubmergedStartMs.load();
			 long long submergedMs =0;
			 if (start !=0) submergedMs = nowMs - start;
			 // Clear start immediately so other logic knows we're no longer submerged
			 leftSubmergedStartMs.store(0);
			 // ignore tiny Z jitter transitions on exit: require current pos sufficiently above previous water height
			 if (!(leftPos.z - prevLeftWaterHeight > cfgMinZDiffForEntryExit)) {
			 } else {
			 }
 }
 if (rightInWater && !lastRightInWater) {
			 long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
			 lastRightTransitionMs.store(nowMs);
			 // mark submerged start time for right
			 rightSubmergedStartMs.store(nowMs);
			 if (!(prevRightPos.z - rightWaterHeight > cfgMinZDiffForEntryExit)) {
			 } else {
			 }
 }
 if (!rightInWater && lastRightInWater) {
			 long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
			 lastRightTransitionMs.store(nowMs);
			 long long start = rightSubmergedStartMs.load();
			 long long submergedMs =0;
			 if (start !=0) submergedMs = nowMs - start;
			 rightSubmergedStartMs.store(0);
			 if (!(rightPos.z - prevRightWaterHeight > cfgMinZDiffForEntryExit)) {
			 } else {
			 }
 }

			auto now = std::chrono::steady_clock::now();
			float leftDt =0.0f;
			float rightDt = 0.0f;
			if (havePrevLeft) leftDt = std::chrono::duration<float>(now - prevLeftTime).count();
			if (havePrevRight) rightDt = std::chrono::duration<float>(now - prevRightTime).count();

			// use cfg variables for thresholds
			float movingConfirm = cfgMovingConfirmSeconds;
			float jitterThreshold = cfgJitterThresholdAdjusted;
			float movingThreshold = cfgMovingThresholdAdjusted;

			// --- Movement detection (use translational delta from sample buffer to ignore rotation) ---
			if (havePrevLeft && leftDt >1e-6f) {
				float speed =0.0f;
				if (leftSamples.size() >=2) {
					const auto &sPrev = leftSamples[leftSamples.size() -2];
					const auto &sCur = leftSamples[leftSamples.size() -1];
					double ds = std::chrono::duration<double>(sCur.t - sPrev.t).count();
					if (ds >1e-6) {
						float dx = sCur.pos.x - sPrev.pos.x;
						float dy = sCur.pos.y - sPrev.pos.y;
						float dz = sCur.pos.z - sPrev.pos.z;
						// use full3D translational magnitude (helps detect vertical movement as well)
						speed = std::sqrt(dx * dx + dy * dy + dz * dz) / (float)ds;
					}
				}

				if (speed <= kMaxValidSpeed) {
					recentLeftSpeed = speed; // full translational speed

					if (speed > movingThreshold) {
						leftLastMovementTime = now;
					}

					// Debug: log computed translational speed occasionally to help tuning
					// (silenced to only log start/stop transitions)
					// if (recentLeftSpeed >=2.0f && recentLeftSpeed <1e4f) {
					// 	AppendToPluginLog("DEBUG", "Left translational speed=%.4f m/s", recentLeftSpeed);
					// }

					if (!leftMoving) {
						if (speed > movingThreshold) {
							if (leftMovementCandidateTime.time_since_epoch().count() ==0) {
								leftMovementCandidateTime = now;
							} else {
								float cand = std::chrono::duration<float>(now - leftMovementCandidateTime).count();
								if (cand >= movingConfirm) {
									leftMoving = true;
									leftMovementCandidateTime = std::chrono::steady_clock::time_point{};
									// mark last movement time when we deem moving
									leftLastMovementTime = now;
								}
							}
						} else {
							// reset candidate if motion drops below threshold
					 leftMovementCandidateTime = std::chrono::steady_clock::time_point{};
						}
					} else {
						// already moving: check for stationary (require kStationaryConfirmSeconds of no significant movement)
						float secsSince = std::chrono::duration<float>(now - leftLastMovementTime).count();
						if (secsSince >= kStationaryConfirmSeconds) {
							leftMoving = false;
						}
					}
				}
			}

			if (havePrevRight && rightDt >1e-6f) {
				float speedR =0.0f;
				if (rightSamples.size() >=2) {
					const auto &sPrev = rightSamples[rightSamples.size() -2];
					const auto &sCur = rightSamples[rightSamples.size() -1];
					double ds = std::chrono::duration<double>(sCur.t - sPrev.t).count();
					if (ds >1e-6) {
						float dx = sCur.pos.x - sPrev.pos.x;
						float dy = sCur.pos.y - sPrev.pos.y;
						float dz = sCur.pos.z - sPrev.pos.z;
						// use full3D translational magnitude
						speedR = std::sqrt(dx * dx + dy * dy + dz * dz) / (float)ds;
					}
				}

				if (speedR <= kMaxValidSpeed) {
					recentRightSpeed = speedR;

					if (speedR > movingThreshold) rightLastMovementTime = now;

					// Debug: occasional speed log
					// (silenced to only log start/stop transitions)
					// if (recentRightSpeed >=2.0f && recentRightSpeed <1e4f) {
					// 	AppendToPluginLog("DEBUG", "Right translational speed=%.4f m/s", recentRightSpeed);
					// }

					if (!rightMoving) {
						if (speedR > movingThreshold) {
							if (rightMovementCandidateTime.time_since_epoch().count() ==0) {
								rightMovementCandidateTime = now;
							} else {
								float candR = std::chrono::duration<float>(now - rightMovementCandidateTime).count();
								if (candR >= movingConfirm) {
									rightMoving = true;
									rightMovementCandidateTime = std::chrono::steady_clock::time_point{};
									rightLastMovementTime = now;
								}
							}
						} else {
						 rightMovementCandidateTime = std::chrono::steady_clock::time_point{};
						}
					} else {
						float secsSinceR = std::chrono::duration<float>(now - rightLastMovementTime).count();
						if (secsSinceR >= kStationaryConfirmSeconds) rightMoving = false;
					}
				}
			}
			// ---------------------------------------------------------

			// Determine wake speed threshold (same used by spawn logic)
			float wakeSpeedThreshold = std::max(0.01f, movingThreshold *0.5f);

			// Determine whether each hand is currently producing wake ripples (submerged && above threshold)
			bool leftProducingWake = (leftInWater && recentLeftSpeed > wakeSpeedThreshold);
			bool rightProducingWake = (rightInWater && recentRightSpeed > wakeSpeedThreshold);

			// Emit logs when producing state changes (use atomics topersist across thread runs)
			{
				bool prev = s_prevLeftMoving.load();
				if (leftProducingWake != prev) {
					// suppress verbose controller wake logs per request
					s_prevLeftMoving.store(leftProducingWake);
				}
			}

			{
				bool prev = s_prevRightMoving.load();
				if (rightProducingWake != prev) {
					// suppress verbose controller wake logs per request
				 s_prevRightMoving.store(rightProducingWake);
				}
			}

			// Spawn wake ripples immediately when submerged AND moving; call helper directly for zero-frame latency
			if (cfgWakeEnabled) {
				// Use instantaneous recent translational speed. Use a more sensitive threshold so wakes spawn immediately
				// float wakeSpeedThreshold = std::max(0.01f, movingThreshold *0.5f); // moved above
				auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
				// LEFT
				if (leftInWater && recentLeftSpeed > wakeSpeedThreshold) {
					long long lastMs = s_leftLastWakeMs.load();
					if (cfgWakeSpawnMs ==0 || nowMs - lastMs >= cfgWakeSpawnMs) {
						// compute multiplier from recent speed
						float mult = cfgWakeMinMultiplier;
						if (cfgWakeScaleMultiplier >0.0f) {
							mult = recentLeftSpeed * cfgWakeScaleMultiplier;
							if (mult < cfgWakeMinMultiplier) mult = cfgWakeMinMultiplier;
							if (mult > cfgWakeMaxMultiplier) mult = cfgWakeMaxMultiplier;
						}
						RE::NiPoint3 wakePos = leftPos;
						wakePos.z = leftWaterHeight;
						float base = cfgWakeAmt;
						float finalAmt = base * mult;
 
 // Schedule emission on main thread via SKSE TaskInterface for safe engine access. The task reads
 // the atomic wake position when it runs so ripples follow the controller even if the task executes
 // a few ms after scheduling.
 auto taskIntfWakeLocal = SKSE::GetTaskInterface();
 if (taskIntfWakeLocal) {
 std::function<void()> wfn = [wakePos, finalAmt]() {
 if (s_gameLoadInProgress.load()) return;
 EmitWakeRipple(true, wakePos, finalAmt);
 };
 taskIntfWakeLocal->AddTask(wfn);
 } else {
 // Fallback: emit directly using the captured wakePos (task interface unavailable)
 if (!s_gameLoadInProgress.load()) {
 EmitWakeRipple(true, wakePos, finalAmt);
 }
 }
 s_leftLastWakeMs.store(nowMs);
 // Attempt to play wake-move sound when a wake is first scheduled for this submerged session
 if (!TryPlayWakeMoveSound(true)) {
 }
 }
 }
			}
			// ---------------------------------------------------------

			// Left entry -> emit single entry ripple if downward Z speed exceeds configured threshold
			if (leftInWater && !lastLeftInWater) {

				RE::NiPoint3 impactPos = leftPos;
				if (havePrevLeft) {
					float denom = prevLeftPos.z - leftPos.z;
					if (std::abs(denom) >1e-6f) {
						float t = (prevLeftPos.z - prevLeftWaterHeight) / denom;
					 t = std::clamp(t,0.0f,1.0f);
						impactPos.x = prevLeftPos.x + (leftPos.x - prevLeftPos.x) * t;
						impactPos.y = prevLeftPos.y + (leftPos.y - prevLeftPos.y) * t;
						impactPos.z = prevLeftPos.z + (leftPos.z - prevLeftPos.z) * t;
						impactPos.z = leftWaterHeight;
					} else {
						impactPos.z = leftWaterHeight;
					}
				} else {
					impactPos.z = leftWaterHeight;
				}

				// compute downward speed (positive = downward) using central-difference + rotational contrib
				RE::NiPoint3 impactVel{0.0f,0.0f,0.0f};
				float downSpeed =0.0f;
				if (havePrevLeft) {
					// helper to compute central-difference velocity for a given sample index in leftSamples
					auto computeVelocity = [&leftSamples](size_t idx) -> RE::NiPoint3 {
						RE::NiPoint3 v{0.0f,0.0f,0.0f};
						if (leftSamples.size() >=3 && idx >0 && idx +1 < leftSamples.size()) {
							auto &sPrev = leftSamples[idx -1];
							auto &sNext = leftSamples[idx +1];
							double dt = std::chrono::duration<double>(sNext.t - sPrev.t).count();
							if (dt >1e-6) {
								v.x = (sNext.pos.x - sPrev.pos.x) / (float)dt;
								v.y = (sNext.pos.y - sPrev.pos.y) / (float)dt;
								v.z = (sNext.pos.z - sPrev.pos.z) / (float)dt;
							}
						} else if (leftSamples.size() >=2 && idx >0) {
							auto &sPrev = leftSamples[idx -1];
							auto &sCur = leftSamples[idx];
						 double dt = std::chrono::duration<double>(sCur.t - sPrev.t).count();
							if (dt >1e-6) {
								v.x = (sCur.pos.x - sPrev.pos.x) / (float)dt;
								v.y = (sCur.pos.y - sPrev.pos.y) / (float)dt;
								v.z = (sCur.pos.z - sPrev.pos.z) / (float)dt;
							}
						}
						return v;
					};

					// determine indices for prev and current in leftSamples (they should be last two entries)
					 size_t n = leftSamples.size();
					 if (n >=2) {
					 size_t idxPrev = n -2; // previous sampleound
					 size_t idxCur = n -1; // current sample
					 // compute translational velocities
					 RE::NiPoint3 vPrev = computeVelocity(idxPrev);
					 RE::NiPoint3 vCur = computeVelocity(idxCur);

					 // compute fractional impact time t in [0,1] between prev and cur (using prevLeftPos and leftPos as before)
					 float t =0.0f;
					 if (std::abs(prevLeftPos.z - leftPos.z) >1e-6f) {
						 t = (prevLeftPos.z - prevLeftWaterHeight) / (prevLeftPos.z - leftPos.z);
						 t = std::clamp(t,0.0f,1.0f);
					 }
					 // interpolate translational velocity to impact time
					 double dtPrevCur = std::chrono::duration<double>(leftSamples[idxCur].t - leftSamples[idxPrev].t).count();
					 RE::NiPoint3 vImpact{0.0f,0.0f,0.0f};
					 if (dtPrevCur >1e-9) {
						 vImpact.x = vPrev.x + (vCur.x - vPrev.x) * (t);
						 vImpact.y = vPrev.y + (vCur.y - vPrev.y) * (t);
						 vImpact.z = vPrev.z + (vCur.z - vPrev.z) * (t);
					 } else {
						 // fallback to simple delta across prev->cur
						 double dtWhole = std::chrono::duration<double>(leftSamples[idxCur].t - leftSamples[idxPrev].t).count();
						 if (dtWhole >1e-6) {
							 vImpact.x = (leftSamples[idxCur].pos.x - leftSamples[idxPrev].pos.x) / (float)dtWhole;
							 vImpact.y = (leftSamples[idxCur].pos.y - leftSamples[idxPrev].pos.y) / (float)dtWhole;
							 vImpact.z = (leftSamples[idxCur].pos.z - leftSamples[idxPrev].pos.z) / (float)dtWhole;
						 }
					 }

					 // rotational contribution was previously estimated here but caused false positives when it
					 // added vertical velocity; for impact detection we now ONLY use translational velocity
					 // (hand position delta) to determine downSpeed. This avoids rotation-triggered heavy hits.
					 RE::NiPoint3 impactVel = vImpact; // translational only

					 // derive downward speed from translational vertical velocity only (positive = downward)
					 downSpeed = std::max(0.0f, -impactVel.z);
					 // NOTE: intentionally do NOT fallback to full-magnitude to avoid counting horizontal/rotational motion
					 // AppendToPluginLog("DEBUG", "Left entry translational downSpeed=%.4f (translational-only)", downSpeed);
					}
				}
				// save current water height for exit interpolation
				prevLeftWaterHeight = leftWaterHeight;

				if (havePrevLeft && downSpeed >= cfgEntryDownZThreshold && downSpeed <= kMaxEntryDownSpeed) {
				 bool expected = false;
				 if (s_leftRippleEmitted.compare_exchange_strong(expected, true)) {
					 const float minEmitInterval =0.18f;
					 float since = std::chrono::duration<float>(now - s_leftLastRippleTime).count();
					 if (since < minEmitInterval) {
						 s_leftRippleEmitted.store(false);
					 } else {
						 s_leftLastRippleTime = now;
						 auto taskIntf = SKSE::GetTaskInterface();
						 if (taskIntf) {
							// compute splash amount based on downSpeed
						 float amt = ComputeEntrySplashAmount(downSpeed);
						 if (amt <=0.0f) {
								// nothing to emit
								s_leftRippleEmitted.store(false);
						 } else {
								// determine whether to play sound: entry implies just entered, so allow
								bool allowSound = true;
								std::function<void()> fn = [impactPos, amt, downSpeed, allowSound]() {
									if (s_gameLoadInProgress.load()) return;
									EmitSplashIfAllowed(true, impactPos, amt, true,1, "left_entry");
									// small directional ripple (25% of main amt)
									RE::NiPoint3 dir = GetControllerForward(false);
									dir.z =0.0f; dir = Normalize(dir);
									RE::NiPoint3 ripPos = impactPos + dir *0.2f; ripPos.z = impactPos.z;
									EmitSplashIfAllowed(true, ripPos, amt *0.25f, true,1, "left_entry_dir");
									// Play splash sound based on downSpeed band from SpellInteractionsVR.esp
									if (allowSound) PlaySplashSoundForDownSpeed(true, downSpeed);
								};
								taskIntf->AddTask(fn);
							}
						 } else {
							// AppendToPluginLog("WARN", "TaskInterface unavailable: skipping left entry ripple");
							s_leftRippleEmitted.store(false);
						 }
					 }
					}
				} else {
				}
			}

			// Left exit -> emit single exit ripple if upward Z speed exceeds configured threshold
			if (!leftInWater && lastLeftInWater) {
				RE::NiPoint3 impactPos = leftPos;
				if (havePrevLeft) {
					float denom = prevLeftPos.z - leftPos.z;
					if (std::abs(denom) >1e-6f) {
					 float t = (prevLeftPos.z - prevLeftWaterHeight) / denom;
					 t = std::clamp(t,0.0f,1.0f);
						impactPos.x = prevLeftPos.x + (leftPos.x - prevLeftPos.x) * t;
						impactPos.y = prevLeftPos.y + (leftPos.y - prevLeftPos.y) * t;
						impactPos.z = prevLeftPos.z + (leftPos.z - prevLeftPos.z) * t;
						impactPos.z = prevLeftWaterHeight;
					} else {
						impactPos.z = prevLeftWaterHeight;
					}
				}

				float upSpeed =0.0f;
				if (havePrevLeft && leftDt >0.0001f) {
					float vz = (leftPos.z - prevLeftPos.z) / leftDt;
					upSpeed = std::max(0.0f, vz);
				}

				if (havePrevLeft && upSpeed >= cfgExitUpZThreshold && upSpeed <= kMaxExitUpSpeed) {
				 const float minEmitInterval =0.18f;
				 float since = std::chrono::duration<float>(now - s_leftLastRippleTime).count();
				 if (since < minEmitInterval) {
				 } else {
					 s_leftLastRippleTime = now;
					 auto taskIntf = SKSE::GetTaskInterface();
					 if (taskIntf) {
						 // compute exit splash amount from upSpeed
						 float exitAmt = ComputeExitSplashAmount(upSpeed);
						 if (exitAmt <=0.0f) exitAmt = cfgSplashNormalAmt * cfgSplashScale; // fallback

						 // decide whether to play sound: if submerged start was at least1000ms, suppress sound
						 long long start = leftSubmergedStartMs.load();
						 long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
						 long long submergedMs =0;
						 if (start !=0) submergedMs = nowMs - start;
						 bool allowSound = (submergedMs <1000);

						 std::function<void()> fn = [impactPos, exitAmt, upSpeed, allowSound]() {
							 if (s_gameLoadInProgress.load()) return;
							 bool emitted = EmitSplashIfAllowed(true, impactPos, exitAmt, true,0, "left_exit");
							 if (emitted && allowSound) {
								 PlayExitSoundForUpSpeed(true, upSpeed);
							 }
						 };
						 taskIntf->AddTask(fn);
					 } else {
						 // AppendToPluginLog("WARN", "TaskInterface unavailable: skipping left exit ripple");
					 }
				 }
				} else {
				}

				// reset entry-emitted guard
			 s_leftRippleEmitted.store(false);
			}

			// Right entry -> emit single entry ripple if downward Z speed exceeds configured threshold
			if (rightInWater && !lastRightInWater) {
				RE::NiPoint3 impactPos = rightPos;
				if (havePrevRight) {
					float denom = prevRightPos.z - rightPos.z;
					if (std::abs(denom) >1e-6f) {
					 float t = (prevRightPos.z - prevRightWaterHeight) / denom;
					 t = std::clamp(t,0.0f,1.0f);
						impactPos.x = prevRightPos.x + (rightPos.x - prevRightPos.x) * t;
						impactPos.y = prevRightPos.y + (rightPos.y - prevRightPos.y) * t;
						impactPos.z = prevRightPos.z + (rightPos.z - prevRightPos.z) * t;
						impactPos.z = rightWaterHeight;
					} else {
						impactPos.z = rightWaterHeight;
					}
				} else {
					impactPos.z = rightWaterHeight;
				}

				RE::NiPoint3 impactVelR{0.0f,0.0f,0.0f};
			 float downSpeedR =0.0f;
			 if (havePrevRight && rightDt >0.0001f) {
					// previous implementation used simple translational velocity only; maintain that but ensure Z-only is used
					impactVelR.x = (rightPos.x - prevRightPos.x) / rightDt;
					impactVelR.y = (rightPos.y - prevRightPos.y) / rightDt;
					impactVelR.z = (rightPos.z - prevRightPos.z) / rightDt;
					// Use vertical component only for downSpeed
					downSpeedR = std::max(0.0f, -impactVelR.z);
					if (downSpeedR <=0.001f) {
						float magR = std::sqrt(impactVelR.x * impactVelR.x + impactVelR.y * impactVelR.y + impactVelR.z * impactVelR.z);
						// only use magnitude as absolute fallback when vertical is nearly zero but this may
						// reintroduce false positives; keep it conservative thresholded at0.001
						downSpeedR = magR;
					}
				}

				if (havePrevRight && downSpeedR >= cfgEntryDownZThreshold && downSpeedR <= kMaxEntryDownSpeed) {
				 bool expected = false;
				 if (s_rightRippleEmitted.compare_exchange_strong(expected, true)) {
					 const float minEmitInterval =0.18f;
					 float since = std::chrono::duration<float>(now - s_rightLastRippleTime).count();
					 if (since < minEmitInterval) {
						 s_rightRippleEmitted.store(false);
					 } else {
						 s_rightLastRippleTime = now;
						 auto taskIntfR = SKSE::GetTaskInterface();
						 if (taskIntfR) {
							 // compute splash amount based on downSpeedR
							 float amtR = ComputeEntrySplashAmount(downSpeedR);
							 if (amtR <=0.0f) {
								 s_rightRippleEmitted.store(false);
							 } else {
								 std::function<void()> fnR = [impactPos, amtR, downSpeedR]() {
									 if (s_gameLoadInProgress.load()) return;
									 EmitSplashIfAllowed(false, impactPos, amtR, true,1, "right_entry");
									 RE::NiPoint3 dir = GetControllerForward(true);
									 dir.z =0.0f; dir = Normalize(dir);
									 RE::NiPoint3 ripPos = impactPos + dir *0.2f; ripPos.z = impactPos.z;
									 EmitSplashIfAllowed(false, ripPos, amtR *0.25f, true,1, "right_entry_dir");
									 // Play splash sound for right hand based on downSpeedR
									 PlaySplashSoundForDownSpeed(false, downSpeedR);
								 };
								 taskIntfR->AddTask(fnR);
							}
						 } else {
							// AppendToPluginLog("WARN", "TaskInterface unavailable: skipping right entry ripple");
						 s_rightRippleEmitted.store(false);
						 }
					 }
				 }
			 } else {
			 }
			}

			// Right exit -> emit single exit ripple if upward Z speed exceeds configured threshold
			if (!rightInWater && lastRightInWater) {
				RE::NiPoint3 impactPos = rightPos;
				if (havePrevRight) {
					float denom = prevRightPos.z - rightPos.z;
					if (std::abs(denom) >1e-6f) {
					 float t = (prevRightPos.z - prevRightWaterHeight) / denom;
					 t = std::clamp(t,0.0f,1.0f);
						impactPos.x = prevRightPos.x + (rightPos.x - prevRightPos.x) * t;
						impactPos.y = prevRightPos.y + (rightPos.y - prevRightPos.y) * t;
						impactPos.z = prevRightPos.z + (rightPos.z - prevRightPos.z) * t;
						impactPos.z = prevRightWaterHeight;
					} else {
						impactPos.z = prevRightWaterHeight;
					}
				}

				float upSpeedR =0.0f;
				if (havePrevRight && rightDt >0.0001f) {
					float vz = (rightPos.z - prevRightPos.z) / rightDt;
					upSpeedR = std::max(0.0f, vz);
				}

				if (havePrevRight && upSpeedR >= cfgExitUpZThreshold && upSpeedR <= kMaxExitUpSpeed) {
				 const float minEmitInterval =0.18f;
				 float since = std::chrono::duration<float>(now - s_rightLastRippleTime).count();
				 if (since < minEmitInterval) {
				 } else {
					 s_rightLastRippleTime = now;
					 auto taskIntfR = SKSE::GetTaskInterface();
					 if (taskIntfR) {
						 float exitAmtR = ComputeExitSplashAmount(upSpeedR);
						 if (exitAmtR <=0.0f) exitAmtR = cfgSplashNormalAmt * cfgSplashScale;

						 long long start = rightSubmergedStartMs.load();
						 long long nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
						 long long submergedMs =0;
						 if (start !=0) submergedMs = nowMs - start;
						 bool allowSoundR = (submergedMs <1000);

						 std::function<void()> fnR = [impactPos, exitAmtR, upSpeedR, allowSoundR]() {
							 if (s_gameLoadInProgress.load()) return;
							 bool emitted = EmitSplashIfAllowed(false, impactPos, exitAmtR, true,0, "right_exit");
							 if (emitted && allowSoundR) {
								 PlayExitSoundForUpSpeed(false, upSpeedR);
							 }
						 };
						 taskIntfR->AddTask(fnR);
					 } else {
						 // AppendToPluginLog("WARN", "TaskInterface unavailable: skipping right exit ripple");
					 }
 }
 } else {
 }

 // reset emitted guard
 s_rightRippleEmitted.store(false);
 }

 // --- Update file-scope submerged flags AFTER emission logic ---
	 leftSubmerged.store(leftInWater);
 rightSubmerged.store(rightInWater);

			 // update previous positions/times
			 prevLeftPos = leftPos;
			 prevRightPos = rightPos;
			 prevLeftTime = now;
			 prevRightTime = now;
			 havePrevLeft = true;
			 havePrevRight = true;

			 lastLeftInWater = leftInWater;
			 lastRightInWater = rightInWater;
		} catch (const std::exception& e) {
			// on exception, avoid tight loop
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
			continue;
		} catch (...) {
			std::this_thread::sleep_for(std::chrono::milliseconds(250));
			continue;
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(kPollIntervalMs));
	}
}

// Wake ripple helper: adds a ripple at surface and rate-limited logs emission.
static void EmitWakeRipple(bool isLeft, const RE::NiPoint3& p, float amt) {
 auto ws = RE::TESWaterSystem::GetSingleton();
 if (ws) {
 ws->AddRipple(p, amt);
 }
}

void StartWaterMonitoring() {
	if (s_running.exchange(true)) return;
	// reset persistently logged moving states
	s_prevLeftMoving.store(false);
	s_prevRightMoving.store(false);
 s_leftRippleEmitted.store(false);
 s_rightRippleEmitted.store(false);
	// reset player swimming state
	s_prevPlayerSwimming.store(false);
	s_thread = std::thread(MonitoringThread);
	// monitoring start log silenced per request
	// InteractiveWaterVR::AppendToPluginLog("INFO", "StartWaterMonitoring: monitoring started");
}

void StopWaterMonitoring() {
	if (!s_running.exchange(false)) return;
	if ( s_thread.joinable()) s_thread.join();
	// reset prev-moving flags
	s_prevLeftMoving.store(false);
 s_prevRightMoving.store(false);
	// reset player swimming state
	s_prevPlayerSwimming.store(false);
}

bool IsMonitoringActive() { return s_running.load(); }

} // namespace InteractiveWaterVR
