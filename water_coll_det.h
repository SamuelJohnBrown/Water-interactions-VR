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
}
