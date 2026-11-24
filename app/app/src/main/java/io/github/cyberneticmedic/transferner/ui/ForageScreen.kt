package io.github.cyberneticmedic.transferner.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.cyberneticmedic.transferner.R
import io.github.cyberneticmedic.transferner.core.Tier

/**
 * The forage list (D-044): which genera buzz in walk mode (Alert), only show a chip (Show), or are ignored
 * (Off). Defaults come from the bundle's forage.json (valuable fruit, nuts, tubers, greens and medicinals
 * alert; other cited edibles show); changes are kept on the phone.
 */
@Composable
fun ForageScreen(vm: AppViewModel) {
    val files = (vm.bundle.collectAsState().value as? BundleState.Ready)?.files ?: return
    val forage = files.forage ?: return
    var query by remember { mutableStateOf("") }
    var version by remember { mutableIntStateOf(0) }
    val tiers = remember(version) { vm.prefs.forageTiers }
    fun tierOf(genus: String, def: String) = tiers[genus] ?: Tier.of(def)
    val rows = forage.genera
        .filter { query.isBlank() || it.genus.contains(query, true) || it.label.contains(query, true) }
        .sortedWith(compareBy({ tierOf(it.genus, it.tier).ordinal }, { it.genus }))
    Column(Modifier.fillMaxSize().background(C.paper).safeDrawingPadding()) {
        TopBar("Forage list", R.drawable.ic_back, { vm.back() })
        Column(Modifier.padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text("Alert buzzes when BioCLIP agrees; Show only raises a chip on the live screen. A genus is never called safe: many hold toxic species too.",
                fontSize = 13.sp, color = C.ink2, lineHeight = 18.sp)
            OutlinedTextField(value = query, onValueChange = { query = it }, singleLine = true, label = { Text("Search genus or name") },
                modifier = Modifier.fillMaxWidth())
        }
        LazyColumn(Modifier.weight(1f).padding(horizontal = 16.dp)) {
            items(rows, key = { it.genus }) { f ->
                val t = tierOf(f.genus, f.tier)
                Column(Modifier.fillMaxWidth().padding(vertical = 8.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(f.label.ifBlank { f.genus }, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = C.ink)
                            Text(f.genus + (if (f.note.isNotBlank()) " · ${f.note}" else ""), style = Scientific, fontSize = 13.sp, color = C.muted)
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        Tier.entries.forEach { opt ->
                            FilterChip(selected = t == opt, onClick = { vm.setForageTier(f.genus, opt); version++ },
                                label = { Text(opt.name.lowercase().replaceFirstChar { it.uppercase() }) })
                        }
                    }
                }
                androidx.compose.foundation.layout.Box(Modifier.fillMaxWidth().height(1.dp).background(C.line))
            }
        }
        OutlinedButton(onClick = { vm.resetForageTiers(); version++ }, shape = RoundedCornerShape(10.dp), modifier = Modifier.padding(16.dp)) {
            Text("Reset to the defaults", color = C.green)
        }
    }
}
