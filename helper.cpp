#include "helper.h"
#include <windows.h>
#include <cstring>
#include <algorithm>
#include <cstdarg>
#include <fstream>

#include <SKSE/SKSE.h>
#include <RE/Skyrim.h>

#include <SKSE/Trampoline.h>

namespace InteractiveWaterVR
{
 std::string GetPluginLogPath()
 {
 const char* user = std::getenv("USERPROFILE");
 if (!user) return {};
 return std::string(user) + "\\Documents\\My Games\\Skyrim VR\\SKSE\\Interactive_Water_VR.log";
 }

 void AppendToPluginLog(const char* level, const char* fmt, ...)
 {
 auto path = GetPluginLogPath();
 if (path.empty()) return;

 std::ofstream ofs(path, std::ios::app);
 if (!ofs.is_open()) return;

 // Timestamp
 SYSTEMTIME st;
 GetLocalTime(&st);
 char timebuf[64];
 snprintf(timebuf, sizeof(timebuf), "%04d-%02d-%02d %02d:%02d:%02d", st.wYear, st.wMonth, st.wDay, st.wHour, st.wMinute, st.wSecond);

 char msgbuf[4096];
 va_list args;
 va_start(args, fmt);
 vsnprintf(msgbuf, sizeof(msgbuf), fmt, args);
 va_end(args);

 ofs << "[" << timebuf << "] [" << level << "] " << msgbuf << std::endl;
 }

 std::uintptr_t Write5Call(std::uintptr_t a_src, std::uintptr_t a_dst) noexcept
 {
 // The original code computed the absolute target from the32-bit displacement at src+1
 const auto disp = *reinterpret_cast<std::int32_t*>(a_src +1);
 const auto nextOp = a_src +5;
 const auto origTarget = nextOp + static_cast<std::uintptr_t>(disp);

 // Use CommonLibSSE-NG trampoline helper to write the call
 auto& trampoline = SKSE::GetTrampoline();
 if (trampoline.empty()) {
 // Trampoline must be created before calling this
 SKSE::log::error("Write5Call: trampoline not initialized");
 return origTarget;
 }

 // write_call will install a call from a_src to a_dst and preserve original bytes in trampoline
 trampoline.write_call<5>(a_src, reinterpret_cast<void*>(a_dst));
 return origTarget;
 }

 void ShowErrorBoxAndTerminate(const char* a_errorString) noexcept
 {
 MessageBoxA(nullptr, a_errorString, "Interactive_Water_VR Fatal Error", MB_ICONERROR | MB_OK | MB_TASKMODAL);
 // intentionally terminate to produce a crash / stop execution for debugging
 std::terminate();
 }

 std::uint32_t GetFullFormIdMine(const char* a_espName, std::uint32_t a_baseFormId) noexcept
 {
 if (!a_espName) {
 return 0;
 }

 std::string espLower(a_espName);
 std::transform(espLower.begin(), espLower.end(), espLower.begin(), ::tolower);

 // If the master plugin is Skyrim.esm, the base id is already full
 if (espLower == "skyrim.esm") {
 return a_baseFormId;
 }

 auto handler = RE::TESDataHandler::GetSingleton();
 if (!handler) {
 return 0;
 }

 auto mod = handler->LookupLoadedModByName(a_espName);
 if (!mod) {
 return 0;
 }

 auto loadedIndex = handler->GetLoadedModIndex(a_espName);
 if (!loadedIndex) {
 return 0;
 }

 // Compose full formid: (modIndex <<24) | baseFormId
 return (static_cast<std::uint32_t>(*loadedIndex) <<24) | (a_baseFormId &0x00FFFFFFu);
 }

 // Define relocation variable for SetAngle and MoveTo
 REL::Relocation<_SetAngle> SetAngle;
 REL::Relocation<_MoveTo> MoveTo;
 REL::Relocation<_Delete> Delete;

 // Implement SetAngleFunc which schedules a task to call the VM function on main thread
 void SetAngleFunc(RE::TESObjectREFR* akSource, float xAngle, float yAngle, float zAngle)
 {
 auto task = SKSE::GetTaskInterface();
 if (!task) {
 // Fallback: attempt to call directly (unsafe unless on main thread)
 auto vm = RE::BSScript::Internal::VirtualMachine::GetSingleton();
 // Retrieve the function pointer from relocation and check it
 auto setAngleFn = SetAngle.get();
 if (vm && setAngleFn) {
 setAngleFn(vm,0, akSource, xAngle, yAngle, zAngle);
 }
 return;
 }

 // Capture parameters by value and post to main thread
 task->AddTask([akSource, xAngle, yAngle, zAngle]() {
 auto vm = RE::BSScript::Internal::VirtualMachine::GetSingleton();
 auto setAngleFn = SetAngle.get();
 if (vm && setAngleFn) {
 setAngleFn(vm,0, akSource, xAngle, yAngle, zAngle);
 }
 });
 }

 // Implement MoveToFunc which schedules a task to call the VM function on main thread
 void MoveToFunc(RE::TESObjectREFR* akSource, RE::TESObjectREFR* refObj, float xOffset, float yOffset, float zOffset, bool matchRotation)
 {
 auto task = SKSE::GetTaskInterface();
 if (!task) {
 // Fallback: attempt to call directly (unsafe unless on main thread)
 auto vm = RE::BSScript::Internal::VirtualMachine::GetSingleton();
 auto moveToFn = MoveTo.get();
 if (vm && moveToFn) {
 moveToFn(vm,0, akSource, refObj, xOffset, yOffset, zOffset, matchRotation);
 }
 return;
 }

 // Capture parameters by value and post to main thread
 task->AddTask([akSource, refObj, xOffset, yOffset, zOffset, matchRotation]() {
 auto vm = RE::BSScript::Internal::VirtualMachine::GetSingleton();
 auto moveToFn = MoveTo.get();
 if (vm && moveToFn) {
 moveToFn(vm,0, akSource, refObj, xOffset, yOffset, zOffset, matchRotation);
 }
 });
 }

 void DeleteFunc(RE::TESObjectREFR* obj)
 {
 if (!obj) {
 return;
 }
 
 auto task = SKSE::GetTaskInterface();
 auto invoke = [obj]() {
 auto vm = VM::GetSingleton();
 auto deleteFn = Delete.get();
 if (vm && deleteFn) {
 deleteFn(vm,0, obj);
 }
 };
 
 if (task) {
 task->AddTask(invoke);
 } else {
 invoke();
 }
 }

 } // namespace InteractiveWaterVR
