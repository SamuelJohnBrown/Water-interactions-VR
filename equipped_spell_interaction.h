#pragma once

namespace InteractiveWaterVR {
// Start/stop monitor that will unequip selected spells / manage shock/frost behaviors.
void StartSpellUnequipMonitor();
void StopSpellUnequipMonitor();

// Note: historical names kept for compatibility with other code that may include the old header
inline void StartSpellUnequipMonitorAlias() { StartSpellUnequipMonitor(); }
inline void StopSpellUnequipMonitorAlias() { StopSpellUnequipMonitor(); }
}
