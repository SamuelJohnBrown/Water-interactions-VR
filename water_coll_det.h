#pragma once

#include <atomic>

namespace InteractiveWaterVR
{
 // Start continuous monitoring of the player's water state. Safe to call multiple times.
 void StartWaterMonitoring();

 // Stop monitoring and join the background thread. Safe to call multiple times.
 void StopWaterMonitoring();

 // Return true if monitoring thread is active
 bool IsMonitoringActive();

 // Notify monitoring thread that game load is starting (will pause emissions)
 void NotifyGameLoadStart();

 // Notify monitoring thread that the game load finished
 void NotifyGameLoadEnd();

 // Query whether a game load is currently in progress
 bool IsGameLoadInProgress();

 // Global flags indicating whether a submerged selected spell has the given magic-damage-type keyword.
 // These are `extern` so other translation units may observe them without pulling in the full implementation.
 extern std::atomic<bool> s_submergedMagicDamageFire;
 extern std::atomic<bool> s_submergedMagicDamageShock;
 extern std::atomic<bool> s_submergedMagicDamageFrost;

 // Per-hand flags for MagicDamageFire (true when that hand is submerged with a fire spell)
 extern std::atomic<bool> s_submergedMagicDamageFireLeft;
 extern std::atomic<bool> s_submergedMagicDamageFireRight;

 // Per-hand flags for MagicDamageFrost (true when that hand is submerged with a frost spell)
 extern std::atomic<bool> s_submergedMagicDamageFrostLeft;
 extern std::atomic<bool> s_submergedMagicDamageFrostRight;

 // Water height to be used when spawning frost movable (world Z). Set by monitoring thread before scheduling spawn.
 extern std::atomic<float> s_frostSpawnWaterHeight;

 // Last-known controller world XY positions (meters). Zero when controller is missing.
 extern std::atomic<float> s_leftControllerWorldX;
 extern std::atomic<float> s_leftControllerWorldY;
 extern std::atomic<float> s_rightControllerWorldX;
 extern std::atomic<float> s_rightControllerWorldY;

 // Controller-specific water detection control. These allow enabling/disabling detection independently.
 void StartLeftWaterDetection();
 void StopLeftWaterDetection();
 void StartRightWaterDetection();
 void StopRightWaterDetection();
 bool IsLeftWaterDetectionActive();
 bool IsRightWaterDetectionActive();
}
