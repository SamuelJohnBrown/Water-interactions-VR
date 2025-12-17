#pragma once

#include <SKSE/SKSE.h>
#include "helper.h"

// Forward-declare HIGGS API types to avoid including the full header in this public header.
namespace HiggsPluginAPI { struct IHiggsInterface001; extern IHiggsInterface001* g_higgsInterface; }

namespace InteractiveWaterVR
{
 // Trampoline provided by CommonLibSSE-NG; pointer placeholder set after creation
 extern SKSE::Trampoline* g_trampoline;

 // Entry point called once init is complete and dependent APIs are available.
 void StartMod();

 // Schedule StartMod to run after a delay (seconds). Safe to call multiple times; only one scheduled start will run.
 void ScheduleStartMod(int delaySeconds =2);

 // Cancel any pending scheduled StartMod and reset internal start state.
 void CancelScheduledStartMod();

 // Log whether SpellInteractionsVR.esp is loaded when game data is ready
 void LogSpellInteractionsVRLoaded();
}
