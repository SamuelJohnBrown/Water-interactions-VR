#include "pch.h"
#include <SKSE/SKSE.h>
#include <SKSE/Trampoline.h>
#include <RE/Skyrim.h>
#include "higgsinterface.h"
#include "helper.h"
#include "engine.h"
#include "water_coll_det.h"
#include <cstdint>
#include <fstream>
#include <cstdlib>
#include <cstdio>

// Minimal plugin info struct used by many SKSE loaders
struct SKSEPluginInfo {
	std::uint32_t infoVersion;
	const char* name;
	std::uint32_t version;
};

// Helper: return the plugin-specific log file path inside the SKSE folder in Documents
static std::string GetPluginLogPath()
{
	const char* user = std::getenv("USERPROFILE");
	if (!user) {
		return {};
	}
	return std::string(user) + "\\Documents\\My Games\\Skyrim VR\\SKSE\\Interactive_Water_VR.log";
}

// Messaging callback: called for SKSE messages
static void OnSKSEMessage(SKSE::MessagingInterface::Message* msg)
{
	if (!msg) {
		return;
	}

	switch (msg->type) {
	case SKSE::MessagingInterface::kPostPostLoad: {
		// Try to obtain HIGGS interface via SKSE messaging now that PostPostLoad occurred
		auto pluginHandle = SKSE::GetPluginHandle();
		auto messaging = SKSE::GetMessagingInterface();
		auto higgs = HiggsPluginAPI::GetHiggsInterface001(pluginHandle, const_cast<SKSE::MessagingInterface*>(messaging));
		if (higgs) {
			// Cache interface for later use
			HiggsPluginAPI::g_higgsInterface = higgs;
			unsigned long build = higgs->GetBuildNumber();
			// Log to SKSE log with fmt-style and to plugin log with printf-style to ensure both receive the build number
			SKSE::log::info("Interactive_Water_VR: obtained HIGGS interface, build {}", build);
			InteractiveWaterVR::AppendToPluginLog("INFO", "Interactive_Water_VR: obtained HIGGS interface, build %lu", build);
			IW_LOG_INFO("Interactive_Water_VR: obtained HIGGS interface");
		} else {
			SKSE::log::info("Interactive_Water_VR: HIGGS interface not available on PostPostLoad");
			IW_LOG_WARN("Interactive_Water_VR: HIGGS interface not available on PostPostLoad");
		}
		break;
	}
	case SKSE::MessagingInterface::kDataLoaded: {
		// Game data has been loaded; check for ESP presence (still not "in world")
		IW_LOG_INFO("Interactive_Water_VR: received kDataLoaded message");
		InteractiveWaterVR::LogSpellInteractionsVRLoaded();
		break;
	}
	case SKSE::MessagingInterface::kPreLoadGame: {
		// A save is about to be loaded - stop runtime monitoring to avoid detecting during load transition
		IW_LOG_INFO("Interactive_Water_VR: received kPreLoadGame - stopping monitoring");
		// Log whether monitoring was active before this load (helps diagnose CTDs when loading from an existing session)
		if (InteractiveWaterVR::IsMonitoringActive()) {
			InteractiveWaterVR::AppendToPluginLog("INFO", "PreLoadGame: monitoring active - will stop before loading save");
		} else {
			InteractiveWaterVR::AppendToPluginLog("INFO", "PreLoadGame: monitoring not active");
		}

		// Notify monitoring code that a game load is starting (pauses background sampling/emission)
		InteractiveWaterVR::NotifyGameLoadStart();

		// Cancel any pending scheduled starts to avoid StartMod firing mid-load
		InteractiveWaterVR::CancelScheduledStartMod();

		// Log key engine pointers (main thread) to help debug transient nulls causing CTDs
		auto player = RE::PlayerCharacter::GetSingleton();
		if (player) {
			auto root = player->Get3D();
			InteractiveWaterVR::AppendToPluginLog("INFO", "PreLoadGame: player=%p, root=%p", static_cast<void*>(player), static_cast<void*>(root));
			#if defined(ENABLE_SKYRIM_VR)
			if (root) {
				auto leftNode = root->GetObjectByName("NPC L Hand [LHnd]");
				auto rightNode = root->GetObjectByName("NPC R Hand [RHnd]");
				InteractiveWaterVR::AppendToPluginLog("INFO", "PreLoadGame: leftNode=%p rightNode=%p", static_cast<void*>(leftNode), static_cast<void*>(rightNode));
			}
			#endif
		} else {
			InteractiveWaterVR::AppendToPluginLog("INFO", "PreLoadGame: player singleton is null");
		}
		InteractiveWaterVR::StopWaterMonitoring();
		break;
	}
	case SKSE::MessagingInterface::kPostLoadGame: {
		// Save finished loading and player should be in-world -> schedule module/runtime logic after short delay
		IW_LOG_INFO("Interactive_Water_VR: received kPostLoadGame - scheduling module start");
		InteractiveWaterVR::AppendToPluginLog("INFO", "PostLoadGame: scheduling StartMod (from load event)");

		InteractiveWaterVR::ScheduleStartMod(2); //5 second delay before starting water detection
		break;
	}
	case SKSE::MessagingInterface::kNewGame: {
		// New game started (fresh world) -> do NOT start water monitoring; just log.
		IW_LOG_INFO("Interactive_Water_VR: received kNewGame - new game detected; not starting water monitoring");
		break;
	}
	default:
		break;
	}
}

// Minimal Query export used by the SKSE loader to identify the plugin
extern "C" __declspec(dllexport) bool SKSEPlugin_Query(const void* /*a_skse*/, SKSEPluginInfo* a_info)
{
	if (a_info) {
		a_info->infoVersion =1; // standard SKSE info version
		a_info->name = "Interactive_Water_VR"; // shown in loader log
		a_info->version =1; // plugin version
	}
	return true;
}

// Load export called after Query
extern "C" __declspec(dllexport) bool SKSEPlugin_Load(const SKSE::LoadInterface* a_skse)
{
	SKSE::Init(a_skse);

	SKSE::log::info("Interactive_Water_VR loaded");

	// Remove old plugin log so we replace it on each load
	const std::string path = GetPluginLogPath();
	if (!path.empty()) {
		std::remove(path.c_str());
	}

	// Defer trampoline creation to CommonLibSSE-NG API init callback for compatibility
	SKSE::RegisterForAPIInitEvent([]()
	{
		try {
			auto& trampoline = SKSE::GetTrampoline();
			if (trampoline.empty()) {
				constexpr std::size_t TrampolineSize =64; // adjust if you need more
				trampoline.create(TrampolineSize);
			}

			// Wire the global trampoline pointer so other modules can reference it
			InteractiveWaterVR::g_trampoline = &trampoline;

			// Confirmation logging
			SKSE::log::info("Interactive_Water_VR: trampoline created (capacity = {} bytes)", trampoline.capacity());
			IW_LOG_INFO("Interactive_Water_VR: trampoline created");

			// Register listener for SKSE messages so we can fetch HIGGS on PostPostLoad and detect DataLoaded / load events
			auto messaging = SKSE::GetMessagingInterface();
			if (messaging) {
				bool reg = messaging->RegisterListener("SKSE", OnSKSEMessage);
				SKSE::log::info("Interactive_Water_VR: registered SKSE messaging listener: {}", reg);
				if (reg) {
					IW_LOG_INFO("Interactive_Water_VR: registered SKSE messaging listener");
				} else {
					IW_LOG_ERROR("Interactive_Water_VR: failed to register SKSE messaging listener");
				}
			} else {
				SKSE::log::warn("Interactive_Water_VR: messaging interface not available during API init");
				IW_LOG_WARN("Interactive_WATER_VR: messaging interface not available during API init");
			}

		} catch (const std::exception& e) {
			SKSE::log::error("Interactive_Water_VR: trampoline creation failed: %s", e.what());
			IW_LOG_ERROR("Interactive_Water_VR: trampoline creation failed");
		} catch (...) {
			SKSE::log::error("Interactive_Water_VR: trampoline creation failed: unknown error");
			IW_LOG_ERROR("Interactive_Water_VR: trampoline creation failed");
		}
	});

	return true;
}


//THIS IS THE PATH WHERE THE DLL IS SENT
// C:\Users\user\Downloads\commonlibsse-ng-template-main\commonlibsse-ng-template-main\build\windows\x64\release


// TO REBUILD THE SKSE PLUGIN, RUN THIS COMMAND
// &"C:\Program Files\\xmake\\xmake.exe" build skse_plugin