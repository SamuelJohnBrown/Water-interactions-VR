#pragma once

#include <cstdint>
#include <string>
#include <optional>

#include <SKSE/SKSE.h>
#include <RE/Skyrim.h>

namespace InteractiveWaterVR
{
using VM = RE::BSScript::Internal::VirtualMachine;

// Logging macros forwarding to SKSE::log and also appending to plugin-specific log file
 void AppendToPluginLog(const char* level, const char* fmt, ...);
 std::string GetPluginLogPath();

 #define IW_LOG_INFO(fmt, ...) do { SKSE::log::info(fmt, ##__VA_ARGS__); InteractiveWaterVR::AppendToPluginLog("INFO", fmt, ##__VA_ARGS__); } while(0)
 #define IW_LOG_WARN(fmt, ...) do { SKSE::log::warn(fmt, ##__VA_ARGS__); InteractiveWaterVR::AppendToPluginLog("WARN", fmt, ##__VA_ARGS__); } while(0)
 #define IW_LOG_ERROR(fmt, ...) do { SKSE::log::error(fmt, ##__VA_ARGS__); InteractiveWaterVR::AppendToPluginLog("ERROR", fmt, ##__VA_ARGS__); } while(0)

 // Backwards-compatible helper used by existing code to write simple plugin messages to SKSE log
 inline void WritePluginLog(const char* a_msg)
 {
 if (a_msg) {
 IW_LOG_INFO("%s", a_msg);
 }
 }

 // Replace a5-byte call at a_src with a trampoline-installed call to a_dst.
 // Returns the original call target address (the address the original call would have invoked).
 std::uintptr_t Write5Call(std::uintptr_t a_src, std::uintptr_t a_dst) noexcept;

 // Show a modal error box and terminate the process (used for fatal errors).
 void ShowErrorBoxAndTerminate(const char* a_errorString) noexcept;

 // Resolve a full form id (combined with mod index) from an esp/plugin name and base form id.
 // Returns0 on failure.
 std::uint32_t GetFullFormIdMine(const char* a_espName, std::uint32_t a_baseFormId) noexcept;

 // Helper: lookup a form by plugin name and form id, cast it to T and log if missing.
 template <class T>
 T* LoadFormAndLog(const std::string& a_pluginName, std::uint32_t& a_outFullFormId, std::uint32_t a_baseFormId,
 const char* a_formName) noexcept
 {
 a_outFullFormId = GetFullFormIdMine(a_pluginName.c_str(), a_baseFormId);
 if (a_outFullFormId ==0) {
 IW_LOG_WARN("LoadFormAndLog: {} not found (formid:0x{:08X})", a_formName, a_baseFormId);
 return nullptr;
 }

 auto form = RE::TESForm::LookupByID(a_outFullFormId);
 if (!form) {
 IW_LOG_WARN("LoadFormAndLog: {} not found (full formid:0x{:08X})", a_formName, a_outFullFormId);
 return nullptr;
 }

 if (!form->Is(T::FORMTYPE)) {
 IW_LOG_WARN("LoadFormAndLog: {} has wrong type (full formid:0x{:08X})", a_formName, a_outFullFormId);
 return nullptr;
 }

 return static_cast<T*>(form);
 }

// SetAngle relocation/wrapper
 using _SetAngle = bool(*)(RE::BSScript::Internal::VirtualMachine* registry, std::uint32_t stackId, RE::TESObjectREFR* akSource, float afXAngle, float afYAngle, float afZAngle);
 extern REL::Relocation<_SetAngle> SetAngle;
 
 // MoveTo relocation/wrapper
 using _MoveTo = void(*)(RE::BSScript::Internal::VirtualMachine* registry, std::uint32_t stackId, RE::TESObjectREFR* akSource, RE::TESObjectREFR* refObj, float afXOffset, float afYOffset, float afZOffset, bool abMatchRotation);
 extern REL::Relocation<_MoveTo> MoveTo;
 
 // Delete relocation/wrapper
 using _Delete = void(*)(VM* registry, std::uint32_t stackId, RE::TESObjectREFR* obj);
 extern REL::Relocation<_Delete> Delete;

 // Helper to call SetAngle on the main thread via SKSE task interface
 void SetAngleFunc(RE::TESObjectREFR* akSource, float xAngle, float yAngle, float zAngle);

 // Helper to call MoveTo on the main thread via SKSE task interface
 void MoveToFunc(RE::TESObjectREFR* akSource, RE::TESObjectREFR* refObj, float xOffset, float yOffset, float zOffset, bool matchRotation);

 // Helper to safely delete an object via Papyrus VM on the main thread
 void DeleteFunc(RE::TESObjectREFR* obj);
 
 }
