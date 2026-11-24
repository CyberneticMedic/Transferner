package io.github.cyberneticmedic.transferner.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.cyberneticmedic.transferner.R
import io.github.cyberneticmedic.transferner.ml.AccelChoice

@Composable
fun SettingsScreen(vm: AppViewModel) {
    val models by vm.models.collectAsState()
    val bundle by vm.bundle.collectAsState()
    val live by vm.live.collectAsState()
    val files = (bundle as? BundleState.Ready)?.files
    var detAcc by remember { mutableStateOf(vm.prefs.detectorAccel) }
    var spAcc by remember { mutableStateOf(vm.prefs.speciesAccel) }
    var boxes by remember { mutableStateOf(vm.prefs.showBoxes) }
    var timings by remember { mutableStateOf(vm.prefs.showTimings) }

    Column(Modifier.fillMaxSize().background(C.paper).safeDrawingPadding()) {
        TopBar("Models & settings", R.drawable.ic_back, { vm.back() })
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 4.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            SectionLabel("Installed models")
            Card {
                Row { Text("Live detector", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f)); Text(files?.let { mb(it, it.detector.model) } ?: "", fontSize = 13.sp, color = C.muted) }
                Text("MobileNetV4 + SSDLite, full INT8 · ${files?.detector?.genera?.size ?: 0} genera", fontSize = 13.sp, color = C.ink2)
                Text("Runtime: ${models.detector}", fontSize = 13.sp, color = C.muted)
                live?.let { Text("Last frame: model %.1f ms + preprocessing %.1f ms".format(it.modelMs, it.preMs), fontSize = 13.sp, color = C.muted) }
                AccelRow(detAcc) { detAcc = it; vm.prefs.detectorAccel = it }
            }
            Card {
                Row { Text("Species model", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f)); Text(files?.let { mb(it, it.species.model) } ?: "", fontSize = 13.sp, color = C.muted) }
                Text("BioCLIP 2.5 image tower (${files?.species?.variant ?: "?"}) + regional head · ${files?.head?.n ?: 0} species", fontSize = 13.sp, color = C.ink2)
                Text("Runtime: ${models.species}", fontSize = 13.sp, color = C.muted)
                vm.result.collectAsState().value?.takeIf { it.modelMs > 0 }?.let {
                    Text("Last identification: model %.2f s, total %.2f s (budget 10 s)".format(it.modelMs / 1000, it.totalMs / 1000), fontSize = 13.sp, color = C.muted)
                }
                AccelRow(spAcc) { spAcc = it; vm.prefs.speciesAccel = it }
            }
            OutlinedButton(onClick = { vm.reloadModels() }, shape = RoundedCornerShape(10.dp), modifier = Modifier.fillMaxWidth()) {
                Text("Reload models with these runtimes", color = C.green, fontWeight = FontWeight.SemiBold)
            }
            models.error?.let { Text(it, fontSize = 12.sp, color = C.caution) }

            SectionLabel("Bundle", Modifier.padding(top = 6.dp))
            Card {
                Text("Version ${files?.manifest?.version ?: "none"} · ${files?.manifest?.files?.size ?: 0} files, verified", fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                Text("Server: ${vm.prefs.server}", fontSize = 13.sp, color = C.muted)
                OutlinedButton(onClick = { vm.go(Screen.SETUP) }, shape = RoundedCornerShape(10.dp)) { Text("Check for a new bundle", color = C.green) }
            }

            SectionLabel("Prototype controls", Modifier.padding(top = 6.dp))
            Card {
                Toggle("Show live boxes", boxes) { boxes = it; vm.prefs.showBoxes = it }
                Toggle("Show timings", timings) { timings = it; vm.prefs.showTimings = it }
                Text("The live label comes from the whole frame (D-033: 56% genus top-1 against 49% for the best box). NPU runs need the phone tests first.",
                    fontSize = 12.5.sp, color = C.muted, lineHeight = 17.sp)
            }
        }
    }
}

private fun mb(files: io.github.cyberneticmedic.transferner.data.BundleFiles, name: String): String =
    files.manifest.files.firstOrNull { it.path == name }?.let { "%.0f MB".format(it.bytes / 1048576.0) } ?: ""

@Composable
private fun Card(content: @Composable ColumnScope.() -> Unit) =
    Column(Modifier.fillMaxWidth().background(C.card, RoundedCornerShape(14.dp)).border(1.dp, C.line, RoundedCornerShape(14.dp)).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(6.dp), content = content)

@Composable
private fun AccelRow(sel: AccelChoice, onPick: (AccelChoice) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
        Text("Run on", fontSize = 13.sp, color = C.ink2)
        AccelChoice.entries.forEach { a -> FilterChip(selected = sel == a, onClick = { onPick(a) }, label = { Text(a.label) }) }
    }
}

@Composable
private fun Toggle(label: String, on: Boolean, set: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, fontSize = 14.sp, modifier = Modifier.weight(1f))
        Switch(checked = on, onCheckedChange = set, colors = SwitchDefaults.colors(checkedTrackColor = C.green))
    }
}
