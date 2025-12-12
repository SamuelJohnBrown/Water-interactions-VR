#pragma once

#include <string>
#include <fstream>
#include <cstdarg>
#include "helper.h"
#include "water_coll_det.h"

namespace InteractiveWaterVR
{
 // Config values
 extern int logging; // log level threshold (0 = errors only)
 extern int leftHandedMode;

 // Movement detection configurable values
 extern float cfgMovingConfirmSeconds;
 extern float cfgJitterThresholdAdjusted;
 extern float cfgMovingThresholdAdjusted;

 // Ripple entry/exit Z thresholds (m/s). Entry requires downward Z speed >= this to emit on entry.
 // Exit requires upward Z speed >= this to emit on exit.
 extern float cfgEntryDownZThreshold;
 extern float cfgExitUpZThreshold;
 
 // Minimum absolute Z position change (meters) required to accept an in/out water state change.
 // Small Z changes due to rotation should be ignored if below this threshold.
 extern float cfgMinZDiffForEntryExit;

 // Splash amplitude band thresholds and amplitudes (tunable via INI)
 extern float cfgSplashVeryLightMax; // upper bound velocity for very light band (entry)
 extern float cfgSplashLightMax; // upper bound velocity for light band (entry)
 extern float cfgSplashNormalMax; // upper bound for normal band (entry)
 extern float cfgSplashHardMax; // upper bound for hard band (entry)
 extern float cfgSplashVeryLightAmt; // amplitude for very light (entry)
 extern float cfgSplashLightAmt; // amplitude for light (entry)
 extern float cfgSplashNormalAmt; // amplitude for normal (entry)
 extern float cfgSplashHardAmt; // amplitude for hard (entry)
 extern float cfgSplashVeryHardAmt; // amplitude for very hard (entry)

 // Per-band volume multipliers for splash sounds (0.0 = silent,1.0 = default)
 extern float cfgSplashVeryLightVol;
 extern float cfgSplashLightVol;
 extern float cfgSplashNormalVol;
 extern float cfgSplashHardVol;
 extern float cfgSplashVeryHardVol;

 // Exit-specific bands (separate tuning)
 extern float cfgSplashExitVeryLightMax;
 extern float cfgSplashExitLightMax;
 extern float cfgSplashExitNormalMax;
 extern float cfgSplashExitHardMax;
 extern float cfgSplashExitVeryLightAmt;
 extern float cfgSplashExitLightAmt;
 extern float cfgSplashExitNormalAmt;
 extern float cfgSplashExitHardAmt;
 extern float cfgSplashExitVeryHardAmt;

 // Per-band volume multipliers for exit sounds
 extern float cfgSplashExitVeryLightVol;
 extern float cfgSplashExitLightVol;
 extern float cfgSplashExitNormalVol;
 extern float cfgSplashExitHardVol;
 extern float cfgSplashExitVeryHardVol;

 extern float cfgSplashScale; // global multiplier applied to final amount

 // Wake ripple amplitude (spawned every frame while submerged & moving)
 extern float cfgWakeAmt;
 extern bool cfgWakeEnabled; // Wake ripple enabled flag
 extern int cfgWakeSpawnMs; // Minimum milliseconds between scheduling wake ripples per hand (0 = every frame)
 extern float cfgWakeScaleMultiplier; // multiplier applied per m/s of recent speed
 extern float cfgWakeMinMultiplier; // minimum multiplier applied to base cfgWakeAmt
 extern float cfgWakeMaxMultiplier; // maximum multiplier applied to base cfgWakeAmt
 extern float cfgWakeMoveSoundVol; // volume for wake movement sound

 // Tracking loss splash delay
 extern float cfgTrackingLossSplashDelaySeconds; // delay before splash when tracking is lost

 // Load configuration from Data\\SKSE\\Plugins\\Interactive_Water_VR.ini
 void loadConfig();

 // Simple logging helper (keeps compatibility with old LOG macros)
 void Log(int msgLogLevel, const char* fmt, ...);

 enum eLogLevels
 {
 LOGLEVEL_ERR =0,
 LOGLEVEL_WARN,
 LOGLEVEL_INFO,
 };
}

// Convenience macros matching original project
#define LOG(fmt, ...) InteractiveWaterVR::Log(InteractiveWaterVR::LOGLEVEL_WARN, fmt, ##__VA_ARGS__)
#define LOG_ERR(fmt, ...) InteractiveWaterVR::Log(InteractiveWaterVR::LOGLEVEL_ERR, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) InteractiveWaterVR::Log(InteractiveWaterVR::LOGLEVEL_INFO, fmt, ##__VA_ARGS__)
