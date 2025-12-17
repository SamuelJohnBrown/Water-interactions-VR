#include "config.h"
#include "water_coll_det.h"
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <cstdarg>
#include <windows.h>

namespace InteractiveWaterVR
{
 int logging =0;
 int leftHandedMode =0;

 // Movement detection configurable values (defaults match current code)
 float cfgMovingConfirmSeconds =1.0f;
 float cfgJitterThresholdAdjusted =0.02f;
 float cfgMovingThresholdAdjusted =0.08f;
 // Ripple entry/exit Z thresholds (m/s)
 float cfgEntryDownZThreshold =0.5f;
 float cfgExitUpZThreshold =0.5f;
 // Minimum Z change to consider for entry/exit to avoid rotation-induced small Z moves
 float cfgMinZDiffForEntryExit =0.01f; //1 cm

 // Splash amplitude band thresholds and amplitudes (defaults matching your requested values)
 float cfgSplashVeryLightMax =30.0f; // up to30 -> very light
 float cfgSplashLightMax =60.0f; // new "light" band upper bound (default60 m/s)
 float cfgSplashNormalMax =1500.0f; // up to1500 -> normal
 float cfgSplashHardMax =4500.0f; // up to4500 -> hard
 float cfgSplashVeryLightAmt =0.01f;
 float cfgSplashLightAmt =0.02f; // default amplitude for light band
 float cfgSplashNormalAmt =0.03f;
 float cfgSplashHardAmt =0.07f;
 float cfgSplashVeryHardAmt =0.10f;

 // Per-band sound volume multipliers (0.0..1.0+)
 float cfgSplashVeryLightVol =1.0f;
 float cfgSplashLightVol =1.0f;
 float cfgSplashNormalVol =1.0f;
 float cfgSplashHardVol =1.0f;
 float cfgSplashVeryHardVol =1.0f;

 // Exit-specific defaults (start same as entry but configurable separately)
 float cfgSplashExitVeryLightMax =30.0f;
 float cfgSplashExitLightMax =60.0f;
 float cfgSplashExitNormalMax =1500.0f;
 float cfgSplashExitHardMax =4500.0f;
 float cfgSplashExitVeryLightAmt =0.01f;
 float cfgSplashExitLightAmt =0.02f;
 float cfgSplashExitNormalAmt =0.03f;
 float cfgSplashExitHardAmt =0.07f;
 float cfgSplashExitVeryHardAmt =0.10f;

 // Per-band exit sound volume defaults per user request
 float cfgSplashExitVeryLightVol =0.2f;
 float cfgSplashExitLightVol =0.2f;
 float cfgSplashExitNormalVol =0.2f;
 float cfgSplashExitHardVol =0.5f;
 float cfgSplashExitVeryHardVol =0.5f;

 float cfgSplashScale =1.0f; // global multiplier

 // Wake ripple amplitude default
 float cfgWakeAmt =0.009f; // default wake amplitude; can be overridden by INI WakeAmt
 // Wake control defaults
 bool cfgWakeEnabled = true;
 int cfgWakeSpawnMs =0; //0 => schedule every frame (subject to rate limiting logs)
 // Wake size multipliers
 float cfgWakeScaleMultiplier =0.06f; // multiply recent speed (m/s) by this to affect wake size (higher default for high-speed wakes)
 float cfgWakeMinMultiplier =0.5f; // min multiplier applied to cfgWakeAmt
 float cfgWakeMaxMultiplier =2.0f; // max multiplier applied to cfgWakeAmt
 // Wake movement sound volume (0.0..1.0+)
 float cfgWakeMoveSoundVol =0.8f; // default volume for wake movement sound
 float cfgTrackingLossSplashDelaySeconds =2.0f; // default2 seconds

 // Option: automatically unequip fire spells when submerged and flagged
 bool cfgAutoUnequipFire = true; // default ON
 // Option: automatically cast shock self spell when submerged and flagged
 bool cfgAutoCastShock = true; // default ON

 float cfgFrostSpawnForwardOffset =2.5f;
 float cfgFrostSpawnOffsetX =0.0f;
 float cfgFrostSpawnOffsetY =0.0f;
 float cfgFrostSpawnOffsetZ =0.0f;
 bool cfgSpellInteractionsEnabled = true;

 static inline void trim(std::string& s)
 {
 s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) { return !std::isspace(ch); }));
 s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(), s.end());
 }

 static inline void skipComments(std::string& s)
 {
 auto pos = s.find(';');
 if (pos != std::string::npos) s.erase(pos);
 pos = s.find('#');
 if (pos != std::string::npos) s.erase(pos);
 }

 // Simple parser for "name=value" pairs returning the value and setting out the name
 static std::string GetConfigSettingsStringValue(const std::string& line, std::string& outName)
 {
 auto pos = line.find('=');
 if (pos == std::string::npos) return {};
 outName = line.substr(0, pos);
 std::string val = line.substr(pos +1);
 trim(outName);
 trim(val);
 // strip quotes
 if (!val.empty() && val.front() == '"' && val.back() == '"') val = val.substr(1, val.size() -2);
 return val;
 }

 static std::string GetDocumentsRuntimeDirectory()
 {
 const char* user = std::getenv("USERPROFILE");
 if (!user) return {};
 return std::string(user) + "\\Documents\\My Games\\Skyrim VR\\";
 }

 static std::string GetExeDirectory()
 {
 char buf[MAX_PATH];
 if (GetModuleFileNameA(nullptr, buf, MAX_PATH) ==0) return {};
 std::filesystem::path p(buf);
 return p.parent_path().string() + "\\";
 }

 static std::string GetModuleDirectory()
 {
 HMODULE h = nullptr;
 // get module handle from address in this module
 if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
 reinterpret_cast<LPCSTR>(&loadConfig), &h)) {
 return {};
 }
 char buf[MAX_PATH];
 if (GetModuleFileNameA(h, buf, MAX_PATH) ==0) return {};
 std::filesystem::path p(buf);
 return p.parent_path().string() + "\\";
 }

 void loadConfig()
 {
 static bool s_warnedOnce = false;

 std::vector<std::string> candidates;
 // Documents path (user config location)
 auto docs = GetDocumentsRuntimeDirectory();
 if (!docs.empty()) candidates.push_back(docs + "Data\\SKSE\\Plugins\\Interactive_Water_VR.ini");
 // Game exe folder Data\\SKSE\\Plugins\\Interactive_Water_VR.ini
 auto exeDir = GetExeDirectory();
 if (!exeDir.empty()) candidates.push_back(exeDir + "Data\\SKSE\\Plugins\\Interactive_Water_VR.ini");
 // Current working directory
 candidates.push_back(std::filesystem::current_path().string() + "\\Data\\SKSE\\Plugins\\Interactive_Water_VR.ini");
 // Plugin DLL directory (where the DLL sits)
 auto modDir = GetModuleDirectory();
 if (!modDir.empty()) candidates.push_back(modDir + "Interactive_Water_VR.ini");
 // also try plugin DLL dir with same filename in case
 if (!modDir.empty()) candidates.push_back(modDir + "\\Interactive_Water_VR.ini");

 std::ifstream file;
 std::string openedPath;
 for (const auto& path : candidates) {
 file.open(path);
 if (file.is_open()) {
 openedPath = path;
 break;
 }
 }

 if (!file.is_open()) {
 if (!s_warnedOnce) {
 IW_LOG_WARN("Config: failed to open any config file candidates");
 for (const auto& p : candidates) IW_LOG_WARN(" tried: %s", p.c_str());
 s_warnedOnce = true;
 }
 return;
 }

 // Successfully opened file; reset warned flag
 s_warnedOnce = false;

 std::string line;
 std::string currentSection;

 while (std::getline(file, line)) {
 trim(line);
 skipComments(line);
 if (line.empty()) continue;

 if (line[0] == '[') {
 auto endBracket = line.find(']');
 if (endBracket != std::string::npos) {
 currentSection = line.substr(1, endBracket -1);
 trim(currentSection);
 }
 } else if (currentSection == "Settings") {
 std::string varName;
 auto value = GetConfigSettingsStringValue(line, varName);
 if (varName == "Logging") {
 try { logging = std::stoi(value); } catch (...) { }
 } else if (varName == "LeftHandedMode") {
 try { leftHandedMode = std::stoi(value); } catch (...) { }
 } 
 } else if (currentSection == "Movement") {
 std::string varName;
 auto value = GetConfigSettingsStringValue(line, varName);
 try {
 if (varName == "MovingConfirmSeconds") cfgMovingConfirmSeconds = std::stof(value);
 else if (varName == "JitterThreshold") cfgJitterThresholdAdjusted = std::stof(value);
 else if (varName == "MovingThreshold") cfgMovingThresholdAdjusted = std::stof(value);
 else if (varName == "EntryDownZThreshold") cfgEntryDownZThreshold = std::stof(value);
 else if (varName == "ExitUpZThreshold") cfgExitUpZThreshold = std::stof(value);
 else if (varName == "MinZDiffForEntryExit") cfgMinZDiffForEntryExit = std::stof(value);
 } catch (...) {
 }
 } else if (currentSection == "Splash") {
 std::string varName;
 auto value = GetConfigSettingsStringValue(line, varName);
 try {
 if (varName == "VeryLightMax") cfgSplashVeryLightMax = std::stof(value);
 else if (varName == "LightMax") cfgSplashLightMax = std::stof(value);
 else if (varName == "NormalMax") cfgSplashNormalMax = std::stof(value);
 else if (varName == "HardMax") cfgSplashHardMax = std::stof(value);
 else if (varName == "VeryLightAmt") cfgSplashVeryLightAmt = std::stof(value);
 else if (varName == "LightAmt") cfgSplashLightAmt = std::stof(value);
 else if (varName == "NormalAmt") cfgSplashNormalAmt = std::stof(value);
 else if (varName == "HardAmt") cfgSplashHardAmt = std::stof(value);
 else if (varName == "VeryHardAmt") cfgSplashVeryHardAmt = std::stof(value);
 else if (varName == "Scale") cfgSplashScale = std::stof(value);
 // new volume keys
 else if (varName == "VeryLightVol") cfgSplashVeryLightVol = std::stof(value);
 else if (varName == "LightVol") cfgSplashLightVol = std::stof(value);
 else if (varName == "NormalVol") cfgSplashNormalVol = std::stof(value);
 else if (varName == "HardVol") cfgSplashHardVol = std::stof(value);
 else if (varName == "VeryHardVol") cfgSplashVeryHardVol = std::stof(value);
 // wake amount
 else if (varName == "WakeAmt") cfgWakeAmt = std::stof(value);
 } catch (...) {
 }
 } else if (currentSection == "SplashExit") {
 std::string varName;
 auto value = GetConfigSettingsStringValue(line, varName);
 try {
 if (varName == "VeryLightMax") cfgSplashExitVeryLightMax = std::stof(value);
 else if (varName == "LightMax") cfgSplashExitLightMax = std::stof(value);
 else if (varName == "NormalMax") cfgSplashExitNormalMax = std::stof(value);
 else if (varName == "HardMax") cfgSplashExitHardMax = std::stof(value);
 else if (varName == "VeryLightAmt") cfgSplashExitVeryLightAmt = std::stof(value);
 else if (varName == "LightAmt") cfgSplashExitLightAmt = std::stof(value);
 else if (varName == "NormalAmt") cfgSplashExitNormalAmt = std::stof(value);
 else if (varName == "HardAmt") cfgSplashExitHardAmt = std::stof(value);
 else if (varName == "VeryHardAmt") cfgSplashExitVeryHardAmt = std::stof(value);
 // exit volume keys
 else if (varName == "VeryLightVol") cfgSplashExitVeryLightVol = std::stof(value);
 else if (varName == "LightVol") cfgSplashExitLightVol = std::stof(value);
 else if (varName == "NormalVol") cfgSplashExitNormalVol = std::stof(value);
 else if (varName == "HardVol") cfgSplashExitHardVol = std::stof(value);
 else if (varName == "VeryHardVol") cfgSplashExitVeryHardVol = std::stof(value);
 } catch (...) {
 }
 } else if (currentSection == "Wake") {
 std::string varName;
 auto value = GetConfigSettingsStringValue(line, varName);
 try {
 if (varName == "Enabled") cfgWakeEnabled = (std::stoi(value) !=0);
 else if (varName == "SpawnMs") cfgWakeSpawnMs = std::stoi(value);
 else if (varName == "ScaleMultiplier") cfgWakeScaleMultiplier = std::stof(value);
 else if (varName == "MinMultiplier") cfgWakeMinMultiplier = std::stof(value);
 else if (varName == "MaxMultiplier") cfgWakeMaxMultiplier = std::stof(value);
 else if (varName == "WaveAmt" || varName == "WaveSize" || varName == "Amt") cfgWakeAmt = std::stof(value);
 else if (varName == "WakeMoveSoundVol") cfgWakeMoveSoundVol = std::stof(value);
 } catch (...) {
 }
 } else if (currentSection == "Loss") {
 std::string varName;
 auto value = GetConfigSettingsStringValue(line, varName);
 try {
 if (varName == "TrackingLossSplashDelaySeconds") cfgTrackingLossSplashDelaySeconds = std::stof(value);
 } catch (...) {
 }
 } else if (currentSection == "Spells") {
 std::string varName;
 auto value = GetConfigSettingsStringValue(line, varName);
 try {
 if (varName == "AutoUnequipFire") cfgAutoUnequipFire = (std::stoi(value) !=0);
 else if (varName == "AutoCastShock") cfgAutoCastShock = (std::stoi(value) !=0);
 else if (varName == "FrostSpawnForwardOffset") cfgFrostSpawnForwardOffset = std::stof(value);
 else if (varName == "FrostSpawnOffsetX") cfgFrostSpawnOffsetX = std::stof(value);
 else if (varName == "FrostSpawnOffsetY") cfgFrostSpawnOffsetY = std::stof(value);
 else if (varName == "FrostSpawnOffsetZ") cfgFrostSpawnOffsetZ = std::stof(value);
 else if (varName == "SpellInteractionsEnabled") cfgSpellInteractionsEnabled = (std::stoi(value) !=0);
 } catch (...) {
 }
 }
 }

 // Enforce maximum wake amplitude clamp so WakeAmt cannot exceed the configured maximum
 constexpr float kMaxWakeAmtClamp =0.009f;
 if (cfgWakeAmt > kMaxWakeAmtClamp) {
 IW_LOG_INFO("Config: WakeAmt %f exceeds max %f - clamping to max", cfgWakeAmt, kMaxWakeAmtClamp);
 cfgWakeAmt = kMaxWakeAmtClamp;
 }
 }

 void Log(int msgLogLevel, const char* fmt, ...)
 {
 if (msgLogLevel > logging) return;

 va_list args;
 va_start(args, fmt);
 char buffer[4096];
 vsnprintf(buffer, sizeof(buffer), fmt, args);
 va_end(args);

 IW_LOG_INFO("%s", buffer);
 }

} // namespace InteractiveWaterVR

