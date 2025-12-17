#include "engine.h"
#include <SKSE/SKSE.h>
#include "higgsinterface.h"
#include "helper.h"
#include "water_coll_det.h"
#include <thread>
#include <atomic>
#include <string>

namespace InteractiveWaterVR
{
 SKSE::Trampoline* g_trampoline = nullptr;

 static std::atomic<bool> s_startScheduled{ false };
 static std::thread s_startThread;
 // Indicates whether StartMod has already executed for the current runtime session
 static std::atomic<bool> s_modStarted{ false };
 // Generation token to invalidate previously scheduled start tasks
 static std::atomic<uint32_t> s_startGeneration{0};
 // Consecutive success tracking for readiness checks
 static std::atomic<uint32_t> s_checkSuccesses{0};
 static std::atomic<uint32_t> s_checkGen{0};

 void CancelScheduledStartMod()
 {
	 // increment generation to invalidate pending tasks
	 s_startGeneration.fetch_add(1, std::memory_order_acq_rel);
	 s_startScheduled.store(false);
	 // reset mod started flag so future loads can start fresh
	 s_modStarted.store(false);
	 try {
		 if (s_startThread.joinable()) s_startThread.join();
	 } catch (...) {
		 IW_LOG_WARN("CancelScheduledStartMod: failed to join start thread");
	 }
 }

 // Ensure SpellInteractions logging runs only once per process
 static std::atomic<bool> s_spellLogged{false};

 void LogSpellInteractionsVRLoaded()
 {
	 // Only run once
	 bool expected = false;
	 if (!s_spellLogged.compare_exchange_strong(expected, true)) {
		 return;
	 }
 
 auto handler = RE::TESDataHandler::GetSingleton();
 if (!handler) {
 IW_LOG_WARN("LogSpellInteractionsVRLoaded: TESDataHandler not available");
 return;
 }

 const auto mod = handler->LookupLoadedModByName("SpellInteractionsVR.esp");
 if (mod) {
 // mod->GetModIndex() isn't universally available; use handler->GetLoadedModIndex
 auto idx = handler->GetLoadedModIndex("SpellInteractionsVR.esp");
 if (idx && *idx !=0xFF) {
 unsigned int modIndex = static_cast<unsigned int>(*idx);
 SKSE::log::info("SpellInteractionsVR.esp is loaded. Mod index:0x{:02X}", modIndex);
 InteractiveWaterVR::AppendToPluginLog("INFO", "SpellInteractionsVR.esp is loaded. Mod index:0x%02X", modIndex);
 } else {
 IW_LOG_WARN("SpellInteractionsVR.esp is loaded but mod index invalid");
 }

 // Iterate all loaded forms and log those that belong to this mod
 auto allFormsPair = RE::TESForm::GetAllForms();
 auto& allFormsMap = *allFormsPair.first;
 auto& lock = allFormsPair.second.get();
 std::size_t found =0;
 {
 // Acquire read lock while iterating
 RE::BSReadLockGuard guard{ lock };
 for (const auto& kv : allFormsMap) {
 const auto form = kv.second;
 if (!form) {
 continue;
 }
 const auto file = form->GetFile();
 if (file == mod) {
 ++found;
 const char* editorID = form->GetFormEditorID();
 const char* name = form->GetName();

 std::string typeStr;
 auto typeView = RE::FormTypeToString(form->GetFormType());
 if (!typeView.empty()) {
 typeStr = std::string(typeView);
 } else {
 typeStr = "<none>";
 }

 std::string editorIDStr = (editorID && *editorID) ? std::string(editorID) : std::string("<none>");
 std::string nameStr = (name && *name) ? std::string(name) : std::string("<none>");

 // Use fmt-style logging for SKSE log
 SKSE::log::info(
 "SpellInteractionsVR record #{}: FormID0x{:08X} Type {} EditorID '{}' Name '{}'",
 found,
 static_cast<unsigned int>(form->GetFormID()),
 typeStr,
 editorIDStr,
 nameStr);

 // Append to plugin log (printf-style)
 InteractiveWaterVR::AppendToPluginLog(
 "INFO",
 "SpellInteractionsVR record #%zu: FormID0x%08X Type %s EditorID '%s' Name '%s'",
 found,
 static_cast<unsigned int>(form->GetFormID()),
 typeStr.c_str(),
 editorIDStr.c_str(),
 nameStr.c_str());
 }
 }
 }

 SKSE::log::info("SpellInteractionsVR.esp: logged {} records", found);
 InteractiveWaterVR::AppendToPluginLog("INFO", "SpellInteractionsVR.esp: logged %zu records", found);

 } else {
 IW_LOG_WARN("SpellInteractionsVR.esp is NOT loaded");
 }
 }

 void StartMod()
 {
 // Implement module startup here: install hooks, initialize state, use helper functions.
 IW_LOG_INFO("Interactive_WATER_VR: StartMod called");

 // Prevent duplicate starts
 bool expectedModStarted = false;
 if (!s_modStarted.compare_exchange_strong(expectedModStarted, true)) {
 IW_LOG_WARN("StartMod: module already started; ignoring duplicate call");
 return;
 }
 
 // Initialize SetAngle relocation for VM call
 try {
 SetAngle = REL::Relocation<_SetAngle>{ REL::VariantID(0,0,0x009D18F0) };
 IW_LOG_INFO("SetAngle relocation initialized");
 } catch (...) {
 IW_LOG_WARN("StartMod: failed to initialize SetAngle relocation");
 }
 
 // Initialize MoveTo relocation for VM call (VR address)
 try {
 MoveTo = REL::Relocation<_MoveTo>{ REL::VariantID(0,0,0x009CF360) };
 IW_LOG_INFO("MoveTo relocation initialized");
 } catch (...) {
 IW_LOG_WARN("StartMod: failed to initialize MoveTo relocation");
 }
 
 // Initialize Delete relocation for despawning (VR address)
 try {
 Delete = REL::Relocation<_Delete>{ REL::VariantID(0,0,0x009CE380) };
 IW_LOG_INFO("Delete relocation initialized");
 } catch (...) {
 IW_LOG_WARN("StartMod: failed to initialize Delete relocation");
 }
  
  // Defensive readiness checks: ensure player and3D root are available before starting runtime monitoring.
 auto player = RE::PlayerCharacter::GetSingleton();
 if (!player) {
 IW_LOG_WARN("StartMod: player singleton not available yet ? rescheduling StartMod");
 ScheduleStartMod(1);
 return;
 }
 auto root = player->Get3D();
 if (!root) {
 IW_LOG_WARN("StartMod: player3D root not available yet ? rescheduling StartMod");
 ScheduleStartMod(1);
 return;
 }
 
 // Clear game-load flag now that we are ready to resume monitoring
 InteractiveWaterVR::NotifyGameLoadEnd();
 
 // Start water monitoring only; HIGGS logging should be done during PostPostLoad when interface is obtained
 StartWaterMonitoring();

 // Schedule logging of SpellInteractionsVR records after a short delay to ensure TES data and editorIDs are available
 std::thread([](){
 std::this_thread::sleep_for(std::chrono::seconds(3));
 auto taskIntf = SKSE::GetTaskInterface();
 if (taskIntf) {
 taskIntf->AddTask([](){ LogSpellInteractionsVRLoaded(); });
 } else {
 LogSpellInteractionsVRLoaded();
 }
 }).detach();
 }

 void ScheduleStartMod(int delaySeconds)
 {
 if (s_startScheduled.exchange(true)) {
 // already scheduled
 return;
 }
 
 // Simple scheduler: sleep for delaySeconds, then post a single main-thread
 // task that calls StartMod. StartMod itself will reschedule if the engine
 // isn't ready yet. This avoids complicated readiness loops that give up.
 s_startThread = std::thread([delaySeconds]() {
 try {
 std::this_thread::sleep_for(std::chrono::seconds(delaySeconds));
 // If scheduling was cancelled while sleeping, bail
 if (!s_startScheduled.load()) return;

 auto taskIntf = SKSE::GetTaskInterface();
 if (taskIntf) {
 taskIntf->AddTask([]() {
 // final cancellation check
 if (!s_startScheduled.load()) return;
 StartMod();
 // clear scheduled flag - StartMod will guard duplicate starts
 s_startScheduled.store(false);
 });
 } else {
 // No TaskInterface: attempt to call StartMod directly on this thread
 if (!s_startScheduled.load()) return;
 StartMod();
 s_startScheduled.store(false);
 }
 } catch (...) {
 IW_LOG_WARN("ScheduleStartMod: exception in delay thread");
 s_startScheduled.store(false);
 }
 });
 }
} // namespace InteractiveWaterVR
