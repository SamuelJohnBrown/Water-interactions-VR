#include "equipped_spell_interaction.h"
#include "water_coll_det.h"
#include "config.h"
#include "helper.h"
#include <thread>
#include <atomic>
#include <chrono>
#include <random>
#include <numbers>
#include <SKSE/SKSE.h>
#include <RE/Skyrim.h>
#include <functional>

namespace InteractiveWaterVR {

static std::atomic<bool> s_threadRunning{false};
static std::thread s_thread;

// Delay before unequip (ms) - requirement: flag must remain true for this duration
// Reduced to be more responsive to short noisy flag transitions
constexpr int kUnequipDelayMs =200;
constexpr int kFrostSpawnDelayMs =5000;
constexpr auto kFrostDespawnDelay = std::chrono::seconds(15);
constexpr auto kFrostChargeStaticExtraDelay = std::chrono::seconds(3);
constexpr auto kFrostChargeStaticScaleStep = std::chrono::milliseconds(50);
constexpr std::size_t kFrostChargeStaticScaleIterations = 80;
constexpr float kFrostChargeScaleMin = 0.0009f;
constexpr float kFrostChargeScaleMax = 1.0f;
constexpr auto kFrostChargeScaleUpStep = std::chrono::milliseconds(40);
constexpr std::size_t kFrostChargeScaleUpIterations = 40;
constexpr auto kFrostChargeScaleDownDelay = std::chrono::seconds(2);
constexpr auto kFrostChargeStaticRespawnInterval = std::chrono::milliseconds(100);

static RE::NiAVObject* GetPlayerHandNode(bool leftHand)
{
	auto player = RE::PlayerCharacter::GetSingleton();
	if (!player) return nullptr;
	auto root = player->Get3D();
	if (!root) return nullptr;
#if defined(ENABLE_SKYRIM_VR)
	const char* nodeName = leftHand ? "NPC L Hand [LHnd]" : "NPC R Hand [RHnd]";
	if (nodeName) {
		auto node = root->GetObjectByName(nodeName);
		if (node) return node;
	}
#endif
	return root;
}

// Shock self spell cached
static RE::SpellItem* s_shockSelfSpell = nullptr;
static std::uint32_t s_shockSelfFullId =0;
// Track whether we've cast the shock self spell (flag set when scheduled)
static std::atomic<bool> s_shockSpellActive{false};

// Frost spawn form cache
static RE::TESObjectSTAT* s_frostSpawnForm = nullptr;
static std::uint32_t s_frostSpawnFullId =0;
static RE::BGSSoundDescriptorForm* s_frostChargeSoundDesc = nullptr;
static RE::BSSoundHandle s_leftFrostChargeHandle;
static RE::BSSoundHandle s_rightFrostChargeHandle;
static bool s_leftFrostChargePlaying = false;
static bool s_rightFrostChargePlaying = false;
static RE::TESObjectSTAT* s_frostChargeStaticForm = nullptr;
static std::uint32_t s_frostChargeStaticFullId =0;
static RE::NiPointer<RE::TESObjectREFR> s_leftFrostChargeStaticRef;
static RE::NiPointer<RE::TESObjectREFR> s_rightFrostChargeStaticRef;
static std::atomic<bool> s_leftFrostChargeSpawnerRunning{false};
static std::atomic<bool> s_rightFrostChargeSpawnerRunning{false};
static std::thread s_leftFrostChargeSpawnerThread;
static std::thread s_rightFrostChargeSpawnerThread;

static void ScaleDownAndDeleteStatic(RE::NiPointer<RE::TESObjectREFR> ref);
static void RemoveFrostChargeStatic(bool leftHand, RE::NiPointer<RE::TESObjectREFR> expectedRef = nullptr);
static void SpawnFrostChargeStatic(bool leftHand);
static void StartFrostChargeStaticSpawner(bool leftHand);
static void StopFrostChargeStaticSpawner(bool leftHand);

static bool EnsureFrostChargeSound()
{
 if (s_frostChargeSoundDesc) return true;
 std::uint32_t fullId =0;
 auto form = LoadFormAndLog<RE::BGSSoundDescriptorForm>("SpellInteractionsVR.esp", fullId,0x01000817u, "FrostChargeSound");
 if (form) {
 s_frostChargeSoundDesc = form;
 IW_LOG_INFO("Loaded frost charge sound fullId=0x%08X", fullId);
 }
 return s_frostChargeSoundDesc != nullptr;
}

static bool EnsureFrostChargeStaticForm()
{
 if (s_frostChargeStaticForm) return true;
 std::uint32_t fullId =0;
 auto form = LoadFormAndLog<RE::TESObjectSTAT>("SpellInteractionsVR.esp", fullId,0x0100081Fu, "FrostChargeStatic");
 if (form) {
 s_frostChargeStaticForm = form;
 s_frostChargeStaticFullId = fullId;
 IW_LOG_INFO("Loaded frost charge static fullId=0x%08X", fullId);
 }
 return s_frostChargeStaticForm != nullptr;
}

static void StartFrostChargeSound(bool leftHand)
{
 auto& playing = leftHand ? s_leftFrostChargePlaying : s_rightFrostChargePlaying;
 if (playing) return;
 if (!EnsureFrostChargeSound()) return;
 auto audio = RE::BSAudioManager::GetSingleton();
 if (!audio) return;
 RE::BSSoundHandle handle;
 if (!audio->BuildSoundDataFromDescriptor(handle, static_cast<RE::BSISoundDescriptor*>(s_frostChargeSoundDesc),16)) {
 return;
 }
 if (auto node = GetPlayerHandNode(leftHand)) {
 handle.SetObjectToFollow(node);
 handle.SetPosition(node->world.translate);
 }
 handle.SetVolume(1.0f);
 if (handle.Play()) {
 if (leftHand) {
 s_leftFrostChargeHandle = handle;
 } else {
 s_rightFrostChargeHandle = handle;
 }
 playing = true;
 StartFrostChargeStaticSpawner(leftHand);
 } else {
 handle.Stop();
 }
}

static void StopFrostChargeSound(bool leftHand)
{
 auto& playing = leftHand ? s_leftFrostChargePlaying : s_rightFrostChargePlaying;
 auto& handle = leftHand ? s_leftFrostChargeHandle : s_rightFrostChargeHandle;
 if (!playing) return;
 handle.Stop();
 handle = RE::BSSoundHandle();
 playing = false;
 StopFrostChargeStaticSpawner(leftHand);
 RemoveFrostChargeStatic(leftHand);
}

static float GetRandomFlatYaw()
{
    static thread_local std::mt19937 rng{std::random_device{}()};
    static thread_local std::uniform_real_distribution<float> dist(0.0f, std::numbers::pi_v<float> * 2.0f);
    return dist(rng);
}

static void ApplyFlatRandomRotation(RE::TESObjectREFR* ref)
{
    if (!ref) {
        return;
    }
    constexpr float kRadToDeg = 180.0f / std::numbers::pi_v<float>;
    const float yawDegrees = GetRandomFlatYaw() * kRadToDeg;
    SetAngleFunc(ref, 0.0f, 0.0f, yawDegrees);
}

static void AnimateFrostChargeScaleUp(RE::NiPointer<RE::TESObjectREFR> ref)
{
    if (!ref) {
        return;
    }

    std::thread([ref]() {
        auto applyScale = [ref](float scale) {
            auto task = SKSE::GetTaskInterface();
            auto setScale = [ref, scale]() {
                if (ref) {
                    ref->SetScale(scale);
                }
            };
            if (task) {
                task->AddTask(setScale);
            } else {
                setScale();
            }
        };

        try {
            const float delta = (kFrostChargeScaleMax - kFrostChargeScaleMin) /
                                static_cast<float>(kFrostChargeScaleUpIterations);
            for (std::size_t i = 0; i < kFrostChargeScaleUpIterations; ++i) {
                float nextScale = kFrostChargeScaleMin + delta * static_cast<float>(i + 1);
                if (nextScale > kFrostChargeScaleMax) {
                    nextScale = kFrostChargeScaleMax;
                }
                applyScale(nextScale);
                std::this_thread::sleep_for(kFrostChargeScaleUpStep);
            }
        } catch (...) {
        }
    }).detach();
}

static void StartFrostChargeStaticSpawner(bool leftHand)
{
    auto& running = leftHand ? s_leftFrostChargeSpawnerRunning : s_rightFrostChargeSpawnerRunning;
    auto& worker = leftHand ? s_leftFrostChargeSpawnerThread : s_rightFrostChargeSpawnerThread;
    if (running.exchange(true)) {
        return;
    }

    worker = std::thread([leftHand]() {
        auto& flag = leftHand ? s_leftFrostChargeSpawnerRunning : s_rightFrostChargeSpawnerRunning;
        constexpr auto kSleepStep = std::chrono::milliseconds(50);
        while (flag.load()) {
            SpawnFrostChargeStatic(leftHand);
            auto slept = std::chrono::milliseconds(0);
            while (flag.load() && slept < kFrostChargeStaticRespawnInterval) {
                auto remaining = kFrostChargeStaticRespawnInterval - slept;
                auto slice = remaining < kSleepStep ? remaining : kSleepStep;
                if (slice.count() > 0) {
                    std::this_thread::sleep_for(slice);
                    slept += slice;
                } else {
                    break;
                }
            }
        }
    });
}

static void StopFrostChargeStaticSpawner(bool leftHand)
{
    auto& running = leftHand ? s_leftFrostChargeSpawnerRunning : s_rightFrostChargeSpawnerRunning;
    auto& worker = leftHand ? s_leftFrostChargeSpawnerThread : s_rightFrostChargeSpawnerThread;
    if (!running.exchange(false)) {
        return;
    }
    if (worker.joinable()) {
        worker.join();
    }
}

static void ScheduleDespawn(RE::NiPointer<RE::TESObjectREFR> ref, std::function<void()> onDeleted = nullptr)
{
    if (!ref) {
        return;
    }

    std::thread([ref, onDeleted = std::move(onDeleted)]() mutable {
        constexpr auto kScaleStep = std::chrono::milliseconds(50);
        constexpr float kMinScale = 0.05f;
        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(kFrostDespawnDelay).count();
        auto stepMs = kScaleStep.count();
        std::size_t steps = stepMs > 0 ? static_cast<std::size_t>(totalMs / stepMs) : 0;
        auto remainderMs = totalMs - static_cast<long long>(steps) * stepMs;

        auto applyScale = [ref](float scale) {
            auto task = SKSE::GetTaskInterface();
            auto setScale = [ref, scale]() {
                if (ref) {
                    ref->SetScale(scale);
                }
            };
            if (task) {
                task->AddTask(setScale);
            } else {
                setScale();
            }
        };

        try {
            float currentScale = ref->GetScale();
            if (currentScale <= 0.0f) {
                currentScale = 1.0f;
            }
            if (currentScale < kMinScale) {
                currentScale = kMinScale;
            }

            if (steps == 0) {
                std::this_thread::sleep_for(kFrostDespawnDelay);
            } else {
                const float scaleDelta = (currentScale - kMinScale) / static_cast<float>(steps);
                for (std::size_t i = 0; i < steps; ++i) {
                    float nextScale = currentScale - scaleDelta * static_cast<float>(i + 1);
                    if (nextScale < kMinScale) {
                        nextScale = kMinScale;
                    }
                    applyScale(nextScale);
                    std::this_thread::sleep_for(kScaleStep);
                }
                if (remainderMs > 0) {
                    std::this_thread::sleep_for(std::chrono::milliseconds(remainderMs));
                }
            }
        } catch (...) {
        }

        try {
            DeleteFunc(ref.get());
        } catch (...) {
        }

        if (onDeleted) {
            try {
                onDeleted();
            } catch (...) {
            }
        }
    }).detach();
}

void UnequipSelectedSpellsOnMainThread(RE::PlayerCharacter* player) {
 if (!player) return;
 IW_LOG_INFO("UnequipSelectedSpellsOnMainThread: executing deselect on main thread");
 auto& rt = player->GetActorRuntimeData();
 RE::MagicItem* left = rt.selectedSpells[RE::Actor::SlotTypes::kLeftHand];
 RE::MagicItem* right = rt.selectedSpells[RE::Actor::SlotTypes::kRightHand];
 if (left) {
 IW_LOG_INFO("UnequipSelectedSpellsOnMainThread: left spell present - deselecting");
 player->DeselectSpell(static_cast<RE::SpellItem*>(left));
 } else {
 IW_LOG_INFO("UnequipSelectedSpellsOnMainThread: no left selected spell");
 }
 if (right) {
 IW_LOG_INFO("UnequipSelectedSpellsOnMainThread: right spell present - deselecting");
 player->DeselectSpell(static_cast<RE::SpellItem*>(right));
 } else {
 IW_LOG_INFO("UnequipSelectedSpellsOnMainThread: no right selected spell");
 }
}

// New: unequip only the selected spell for the given hand (true == left)
void UnequipSelectedSpellOnMainThread(RE::PlayerCharacter* player, bool leftHand) {
 if (!player) return;
 auto& rt = player->GetActorRuntimeData();
 RE::MagicItem* spell = rt.selectedSpells[leftHand ? RE::Actor::SlotTypes::kLeftHand : RE::Actor::SlotTypes::kRightHand];
 if (spell) {
 IW_LOG_INFO("UnequipSelectedSpellOnMainThread: %s spell present - deselecting", leftHand ? "left" : "right");
 // Prefer clearing the specific hand's magic caster so we don't accidentally deselect the same spell from the other hand
 try {
 auto caster = player->GetMagicCaster(leftHand ? RE::MagicSystem::CastingSource::kLeftHand : RE::MagicSystem::CastingSource::kRightHand);
 if (caster) {
 IW_LOG_INFO("UnequipSelectedSpellOnMainThread: caster found for %s hand - using caster DeselectSpellImpl", leftHand ? "left" : "right");
 // interrupt current cast if it matches and clear the current spell for this caster
 try {
 if (caster->currentSpell == spell) {
 IW_LOG_INFO("UnequipSelectedSpellOnMainThread: caster currentSpell matches selected spell - interrupting");
 caster->InterruptCast(false);
 }
 // call DeselectSpellImpl to perform per-caster deselect behavior
 IW_LOG_INFO("UnequipSelectedSpellOnMainThread: calling caster->DeselectSpellImpl()");
 caster->DeselectSpellImpl();
 // ensure current spell cleared
 caster->SetCurrentSpell(nullptr);
 } catch (...) {
 IW_LOG_WARN("UnequipSelectedSpellOnMainThread: exception while clearing caster state");
 }
 return;
 }
 } catch (...) {
 // ignore and fallback
 }

 // Fallback: use the existing API which may deselect globally
 player->DeselectSpell(static_cast<RE::SpellItem*>(spell));
 } else {
 IW_LOG_INFO("UnequipSelectedSpellOnMainThread: %s selected spell none", leftHand ? "left" : "right");
 }
}

void TryLoadShockSelfSpell() {
 if (s_shockSelfSpell) return;
 std::uint32_t fullId =0;
 // base form id as provided
 auto form = LoadFormAndLog<RE::SpellItem>("SpellInteractionsVR.esp", fullId,0x01000800u, "shockself");
 if (form) {
 s_shockSelfSpell = form;
 s_shockSelfFullId = fullId;
 IW_LOG_INFO("Loaded shockself spell fullId=0x%08X", fullId);
 }
}

// Helper to cast spell on player using available magic caster (main thread)
static void CastShockSelfOnPlayer(RE::PlayerCharacter* player) {
 if (!player || !s_shockSelfSpell) return;
 // Try to get a magic caster for left or right hand
 RE::MagicCaster* caster = nullptr;
 caster = player->GetMagicCaster(RE::MagicSystem::CastingSource::kLeftHand);
 if (!caster) caster = player->GetMagicCaster(RE::MagicSystem::CastingSource::kRightHand);
 if (!caster) {
 IW_LOG_WARN("CastShockSelfOnPlayer: no magic caster available");
 return;
 }
 // Cast spell immediately on player as self-target
 IW_LOG_INFO("CastShockSelfOnPlayer: casting shockself on player");
 caster->CastSpellImmediate(s_shockSelfSpell, false, player,1.0f, false,0.0f, player);
 s_shockSpellActive.store(true);
 IW_LOG_INFO("CastShockSelfOnPlayer: cast requested");
}

// Helper to stop shock self spell on player (main thread)
static void StopShockSelfOnPlayer(RE::PlayerCharacter* player) {
 if (!player || !s_shockSelfSpell) return;
 RE::MagicCaster* caster = nullptr;
 caster = player->GetMagicCaster(RE::MagicSystem::CastingSource::kLeftHand);
 if (!caster) caster = player->GetMagicCaster(RE::MagicSystem::CastingSource::kRightHand);
 if (!caster) {
 IW_LOG_WARN("StopShockSelfOnPlayer: no magic caster available");
 return;
 }
 // Interrupt and clear current spell if it matches
 try {
 IW_LOG_INFO("StopShockSelfOnPlayer: stopping shockself on player");
 if (caster->currentSpell == s_shockSelfSpell) {
 caster->InterruptCast(false);
 }
 caster->SetCurrentSpell(nullptr);
 s_shockSpellActive.store(false);
 IW_LOG_INFO("StopShockSelfOnPlayer: stopped");
 } catch (...) {
 }
}

// Helper to spawn frost movable static in front of the player on the main thread
static void SpawnFrostMovableInFront(RE::PlayerCharacter* player, bool leftHand) {
 if (!player) return;
 // load form if needed
 if (!s_frostSpawnForm) {
 s_frostSpawnForm = LoadFormAndLog<RE::TESObjectSTAT>("SpellInteractionsVR.esp", s_frostSpawnFullId,0x01000820u, "FrostSpawn");
 if (!s_frostSpawnForm) {
 IW_LOG_WARN("SpawnFrostMovableInFront: failed to load Frost static form");
 return;
 }
 IW_LOG_INFO("SpawnFrostMovableInFront: loaded Frost form -> fullId=0x%08X", s_frostSpawnFullId);
 }
 // schedule on main thread
 auto task = SKSE::GetTaskInterface();
 float ctrlX = leftHand ? InteractiveWaterVR::s_leftControllerWorldX.load() : InteractiveWaterVR::s_rightControllerWorldX.load();
 float ctrlY = leftHand ? InteractiveWaterVR::s_leftControllerWorldY.load() : InteractiveWaterVR::s_rightControllerWorldY.load();
 if (task) {
 IW_LOG_INFO("SpawnFrostMovableInFront: scheduling spawn for %s controller at (%.3f, %.3f)", leftHand ? "left" : "right", ctrlX, ctrlY);
 
 task->AddTask([player, leftHand, ctrlX, ctrlY]() {
 if (!player) return;
 try {
 auto ref = player->PlaceObjectAtMe(s_frostSpawnForm, false);
 if (!ref) {
 IW_LOG_WARN("SpawnFrostMovableInFront: PlaceObjectAtMe failed");
 return;
 }
 auto playerPos = player->GetPosition();
 float xOffset = ctrlX - playerPos.x;
 float yOffset = ctrlY - playerPos.y;
 float waterZ = InteractiveWaterVR::s_frostSpawnWaterHeight.load();
 float zOffset = 0.0f;
 if (waterZ != 0.0f) {
 float refZ = ref->GetPosition().z;
 zOffset = waterZ - refZ;
 }
 MoveToFunc(ref.get(), player, xOffset, yOffset, zOffset, true);
 ApplyFlatRandomRotation(ref.get());
 auto chargeStaticRef = leftHand ? s_leftFrostChargeStaticRef : s_rightFrostChargeStaticRef;
 ScheduleDespawn(ref, [leftHand, chargeStaticRef]() {
 std::this_thread::sleep_for(kFrostChargeStaticExtraDelay);
 RemoveFrostChargeStatic(leftHand, chargeStaticRef);
 });
 IW_LOG_INFO("SpawnFrostMovableInFront: spawned frost movable via %s controller (pos %.3f, %.3f)", leftHand ? "left" : "right", ctrlX, ctrlY);
 } catch (...) {
 IW_LOG_WARN("SpawnFrostMovableInFront: exception during spawn task");
 }
 });
 } else {
 // Fallback: run inline (should be main thread when called) 
 try {
 IW_LOG_INFO("SpawnFrostMovableInFront: running inline spawn for %s controller at (%.3f, %.3f)", leftHand ? "left" : "right", ctrlX, ctrlY);
 auto ref = player->PlaceObjectAtMe(s_frostSpawnForm, false);
 if (ref) {
 auto playerPos = player->GetPosition();
 float xOffset = ctrlX - playerPos.x;
 float yOffset = ctrlY - playerPos.y;
 float waterZ = InteractiveWaterVR::s_frostSpawnWaterHeight.load();
 float zOffset = 0.0f;
 if (waterZ != 0.0f) {
 float refZ = ref->GetPosition().z;
 zOffset = waterZ - refZ;
 }
 MoveToFunc(ref.get(), player, xOffset, yOffset, zOffset, true);
 ApplyFlatRandomRotation(ref.get());
 auto chargeStaticRef = leftHand ? s_leftFrostChargeStaticRef : s_rightFrostChargeStaticRef;
 ScheduleDespawn(ref, [leftHand, chargeStaticRef]() {
 std::this_thread::sleep_for(kFrostChargeStaticExtraDelay);
 RemoveFrostChargeStatic(leftHand, chargeStaticRef);
 });
 IW_LOG_INFO("SpawnFrostMovableInFront: spawned frost movable via %s controller (pos %.3f, %.3f)", leftHand ? "left" : "right", ctrlX, ctrlY);
 } else {
 IW_LOG_WARN("SpawnFrostMovableInFront: PlaceObjectAtMe failed (inline)");
 }
 } catch (...) {
 IW_LOG_WARN("SpawnFrostMovableInFront: exception during inline spawn");
 }
 }
}

static void RemoveFrostChargeStatic(bool leftHand, RE::NiPointer<RE::TESObjectREFR> expectedRef)
{
 auto& stored = leftHand ? s_leftFrostChargeStaticRef : s_rightFrostChargeStaticRef;
 if (!stored) return;
 if (expectedRef && stored.get() != expectedRef.get()) {
 return;
 }
 auto ref = stored;
 stored.reset();
 IW_LOG_INFO("RemoveFrostChargeStatic: removing %s hand static", leftHand ? "left" : "right");
 ScaleDownAndDeleteStatic(ref);
}

static void SpawnFrostChargeStatic(bool leftHand)
{
 if (!EnsureFrostChargeStaticForm()) return;
 auto player = RE::PlayerCharacter::GetSingleton();
 if (!player) return;
 float ctrlX = leftHand ? InteractiveWaterVR::s_leftControllerWorldX.load() : InteractiveWaterVR::s_rightControllerWorldX.load();
 float ctrlY = leftHand ? InteractiveWaterVR::s_leftControllerWorldY.load() : InteractiveWaterVR::s_rightControllerWorldY.load();
 auto task = SKSE::GetTaskInterface();
 auto spawn = [player, leftHand, ctrlX, ctrlY]() {
 if (!player) return;
 try {
 auto ref = player->PlaceObjectAtMe(s_frostChargeStaticForm, false);
 if (!ref) {
 IW_LOG_WARN("SpawnFrostChargeStatic: PlaceObjectAtMe failed");
 return;
 }
 auto& stored = leftHand ? s_leftFrostChargeStaticRef : s_rightFrostChargeStaticRef;
 if (stored) {
 auto prev = stored;
 stored.reset();
 ScaleDownAndDeleteStatic(prev);
 }
 ref->SetScale(kFrostChargeScaleMin);
 AnimateFrostChargeScaleUp(ref);
 auto playerPos = player->GetPosition();
 float xOffset = ctrlX - playerPos.x;
 float yOffset = ctrlY - playerPos.y;
 float waterZ = InteractiveWaterVR::s_frostSpawnWaterHeight.load();
 float zOffset = 0.0f;
 if (waterZ != 0.0f) {
 float refZ = ref->GetPosition().z;
 zOffset = waterZ - refZ;
 }
 MoveToFunc(ref.get(), player, xOffset, yOffset, zOffset, true);
 ApplyFlatRandomRotation(ref.get());
 stored = ref;
 IW_LOG_INFO("SpawnFrostChargeStatic: spawned %s hand static at (%.3f, %.3f)", leftHand ? "left" : "right", ctrlX, ctrlY);
 } catch (...) {
 IW_LOG_WARN("SpawnFrostChargeStatic: exception during spawn");
 }
 };
 if (task) {
 task->AddTask(spawn);
 } else {
 spawn();
 }
}

static void ScaleDownAndDeleteStatic(RE::NiPointer<RE::TESObjectREFR> ref)
{
 if (!ref) {
 return;
 }
 std::thread([ref]() {
 std::this_thread::sleep_for(kFrostChargeScaleDownDelay);
 auto applyScale = [ref](float scale) {
 auto task = SKSE::GetTaskInterface();
 auto setScale = [ref, scale]() {
 if (ref) {
 ref->SetScale(scale);
 }
 };
 if (task) {
 task->AddTask(setScale);
 } else {
 setScale();
 }
 };
 try {
 float currentScale = ref->GetScale();
 if (currentScale <= 0.0f) {
 currentScale = 1.0f;
 }
 const float delta = currentScale / static_cast<float>(kFrostChargeStaticScaleIterations);
 for (std::size_t i = 0; i < kFrostChargeStaticScaleIterations; ++i) {
 float nextScale = currentScale - delta * static_cast<float>(i + 1);
 if (nextScale < 0.0f) {
 nextScale = 0.0f;
 }
 applyScale(nextScale);
 std::this_thread::sleep_for(kFrostChargeStaticScaleStep);
 }
 } catch (...) {
 }
 try {
 DeleteFunc(ref.get());
 } catch (...) {
 }
 }).detach();
}

void MonitorThread() {
 using clock = std::chrono::steady_clock;
 bool prevLeftFire = false;
 bool prevRightFire = false;
 bool handledLeftWhileTrue = false;
 bool handledRightWhileTrue = false;
 auto leftFireTrueSince = clock::time_point{};
 auto rightFireTrueSince = clock::time_point{};

 bool prevShock = false;
 bool prevAnyFrost = false;
 bool prevLeftFrost = false;
 bool prevRightFrost = false;
 bool leftFrostHandledWhileTrue = false;
 bool rightFrostHandledWhileTrue = false;
 auto leftFrostTrueSince = clock::time_point{};
 auto rightFrostTrueSince = clock::time_point{};

 IW_LOG_INFO("MonitorThread: starting (cfgAutoUnequipFire=%d)", cfgAutoUnequipFire ?1 :0);

 while (s_threadRunning.load()) {
 try {
 std::this_thread::sleep_for(std::chrono::milliseconds(100));

 auto now = clock::now();

 // --- Handle fire unequip per-hand (new behavior) ---
 if (cfgAutoUnequipFire) {
 bool curLeftFire = InteractiveWaterVR::s_submergedMagicDamageFireLeft.load();
 bool curRightFire = InteractiveWaterVR::s_submergedMagicDamageFireRight.load();
 // LEFT
 if (curLeftFire) {
 if (!prevLeftFire) {
 leftFireTrueSince = now;
 handledLeftWhileTrue = false;
 IW_LOG_INFO("MonitorThread: left fire flag rising edge detected");
 }
 if (!handledLeftWhileTrue) {
 auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - leftFireTrueSince).count();
 if (elapsed >= kUnequipDelayMs) {
 IW_LOG_INFO("MonitorThread: left fire flag held for %d ms -> scheduling unequip", (int)elapsed);
 auto player = RE::PlayerCharacter::GetSingleton();
 if (player) {
 auto task = SKSE::GetTaskInterface();
 if (task) {
 task->AddTask([player]() {
 if (InteractiveWaterVR::s_submergedMagicDamageFireLeft.load()) {
 UnequipSelectedSpellOnMainThread(player, true);
 }
 });
 } else {
 if (InteractiveWaterVR::s_submergedMagicDamageFireLeft.load()) {
 UnequipSelectedSpellOnMainThread(player, true);
 }
 }
 }
 handledLeftWhileTrue = true;
 }
 }
 } else {
 if (prevLeftFire) IW_LOG_INFO("MonitorThread: left fire flag cleared");
 handledLeftWhileTrue = false;
 }
 prevLeftFire = curLeftFire;

 // RIGHT
 if (curRightFire) {
 if (!prevRightFire) {
 rightFireTrueSince = now;
 handledRightWhileTrue = false;
 IW_LOG_INFO("MonitorThread: right fire flag rising edge detected");
 }
 if (!handledRightWhileTrue) {
 auto elapsedR = std::chrono::duration_cast<std::chrono::milliseconds>(now - rightFireTrueSince).count();
 if (elapsedR >= kUnequipDelayMs) {
 IW_LOG_INFO("MonitorThread: right fire flag held for %d ms -> scheduling unequip", (int)elapsedR);
 auto player = RE::PlayerCharacter::GetSingleton();
 if (player) {
 auto task = SKSE::GetTaskInterface();
 if (task) {
 task->AddTask([player]() {
 if (InteractiveWaterVR::s_submergedMagicDamageFireRight.load()) {
 UnequipSelectedSpellOnMainThread(player, false);
 }
 });
 } else {
 if (InteractiveWaterVR::s_submergedMagicDamageFireRight.load()) {
 UnequipSelectedSpellOnMainThread(player, false);
 }
 }
 }
 handledRightWhileTrue = true;
 }
 }
 } else {
 if (prevRightFire) IW_LOG_INFO("MonitorThread: right fire flag cleared");
 handledRightWhileTrue = false;
 }
 prevRightFire = curRightFire;

 } else {
 // feature disabled - reset state
 if (prevLeftFire || prevRightFire) IW_LOG_INFO("MonitorThread: fire flag ignored due to configuration");
 prevLeftFire = false;
 prevRightFire = false;
 handledLeftWhileTrue = false;
 handledRightWhileTrue = false;
 }

 // --- Handle shock cast: ensure shock self spell is active while submerged flag true ---
 bool curShock = InteractiveWaterVR::s_submergedMagicDamageShock.load();
 if (curShock && !prevShock) {
 // rising edge: ensure spell loaded and cast on player
 TryLoadShockSelfSpell();
 if (s_shockSelfSpell) {
 auto player = RE::PlayerCharacter::GetSingleton();
 if (player) {
 auto task = SKSE::GetTaskInterface();
 if (task) {
 task->AddTask([player]() { CastShockSelfOnPlayer(player); });
 } else {
 CastShockSelfOnPlayer(player);
 }
 }
 }
 }

 if (!curShock && prevShock) {
 // falling edge: stop spell on player
 if (s_shockSelfSpell) {
 auto player = RE::PlayerCharacter::GetSingleton();
 if (player) {
 auto task = SKSE::GetTaskInterface();
 if (task) {
 task->AddTask([player]() { StopShockSelfOnPlayer(player); });
 } else {
 StopShockSelfOnPlayer(player);
 }
 }
 }
 }
 prevShock = curShock;

 // --- Handle frost spawn: require frost flag held for kFrostSpawnDelayMs before spawning ---
 bool curLeftFrost = InteractiveWaterVR::s_submergedMagicDamageFrostLeft.load();
 bool curRightFrost = InteractiveWaterVR::s_submergedMagicDamageFrostRight.load();
 bool curFrost = curLeftFrost || curRightFrost;

 if (curLeftFrost) {
 if (!prevLeftFrost) {
 leftFrostTrueSince = now;
 leftFrostHandledWhileTrue = false;
 StartFrostChargeSound(true);
 float lx = InteractiveWaterVR::s_leftControllerWorldX.load();
 float ly = InteractiveWaterVR::s_leftControllerWorldY.load();
 IW_LOG_INFO("MonitorThread: left frost flag set (spell submerged) at (%.3f, %.3f)", lx, ly);
 }
 if (!leftFrostHandledWhileTrue) {
 auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - leftFrostTrueSince).count();
 if (elapsed >= kFrostSpawnDelayMs) {
 IW_LOG_INFO("MonitorThread: left frost flag held for %d ms -> spawning frost movable", static_cast<int>(elapsed));
 auto player = RE::PlayerCharacter::GetSingleton();
 if (player) SpawnFrostMovableInFront(player, true);
 leftFrostHandledWhileTrue = true;
 StopFrostChargeSound(true);
 }
 }
 } else {
 if (prevLeftFrost) {
 StopFrostChargeSound(true);
 }
 leftFrostHandledWhileTrue = false;
 }
 prevLeftFrost = curLeftFrost;

 if (curRightFrost) {
 if (!prevRightFrost) {
 rightFrostTrueSince = now;
 rightFrostHandledWhileTrue = false;
 StartFrostChargeSound(false);
 float rx = InteractiveWaterVR::s_rightControllerWorldX.load();
 float ry = InteractiveWaterVR::s_rightControllerWorldY.load();
 IW_LOG_INFO("MonitorThread: right frost flag set (spell submerged) at (%.3f, %.3f)", rx, ry);
 }
 if (!rightFrostHandledWhileTrue) {
 auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - rightFrostTrueSince).count();
 if (elapsed >= kFrostSpawnDelayMs) {
 IW_LOG_INFO("MonitorThread: right frost flag held for %d ms -> spawning frost movable", static_cast<int>(elapsed));
 auto player = RE::PlayerCharacter::GetSingleton();
 if (player) SpawnFrostMovableInFront(player, false);
 rightFrostHandledWhileTrue = true;
 StopFrostChargeSound(false);
 }
 }
 } else {
 if (prevRightFrost) {
 StopFrostChargeSound(false);
 }
 rightFrostHandledWhileTrue = false;
 }
 prevRightFrost = curRightFrost;

 if (curFrost && !prevAnyFrost) {
 IW_LOG_INFO("MonitorThread: frost flag rising edge detected");
 } else if (!curFrost && prevAnyFrost) {
 IW_LOG_INFO("MonitorThread: frost flag cleared");
 }
 prevAnyFrost = curFrost;

 } catch (...) {
 // ignore
 }
 }
}

void StartSpellUnequipMonitor() {
 if (!cfgSpellInteractionsEnabled) {
 IW_LOG_INFO("StartSpellUnequipMonitor: disabled via configuration");
 return;
 }
 if (s_threadRunning.exchange(true)) return;
 s_thread = std::thread(MonitorThread);
}

void StopSpellUnequipMonitor() {
 if (!s_threadRunning.exchange(false)) return;
 if (s_thread.joinable()) s_thread.join();
 StopFrostChargeSound(true);
 StopFrostChargeSound(false);
}

} // namespace InteractiveWaterVR
