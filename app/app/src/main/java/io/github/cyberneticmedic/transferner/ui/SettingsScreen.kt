package io.github.cyberneticmedic.transferner.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.FlowRow
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
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
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
import io.github.cyberneticmedic.transferner.data.Servers
import io.github.cyberneticmedic.transferner.R
import io.github.cyberneticmedic.transferner.core.Runtime
import io.github.cyberneticmedic.transferner.core.Tier
import io.github.cyberneticmedic.transferner.ml.AccelChoice

@Composable
fun SettingsScreen(vm: AppViewModel) {
    val models by vm.models.collectAsState()
    val live by vm.live.collectAsState()
    val device by vm.device.collectAsState()
    val plan by vm.plan.collectAsState()
    val last by vm.result.collectAsState()
    val files = vm.inUse.collectAsState().value           // the models in use, also during a download
    var detAcc by remember { mutableStateOf(vm.prefs.detectorAccel) }
    var boxes by remember { mutableStateOf(vm.prefs.showBoxes) }
    var timings by remember { mutableStateOf(vm.prefs.showTimings) }

    Column(Modifier.fillMaxSize().background(C.paper).safeDrawingPadding()) {
        TopBar("Models & settings", R.drawable.ic_back, { vm.back() })
        Column(Modifier.verticalScroll(rememberScrollState()).padding(horizontal = 16.dp, vertical = 4.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
            UpdatesCard(vm)
            SectionLabel("This phone", Modifier.padding(top = 6.dp))
            Card {
                Text(device?.label ?: "…", fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                Text("GPU for LiteRT: " + if (device?.gpuAvailable == true) "available" else "not available", fontSize = 13.sp, color = C.muted)
                plan?.let { Text("Plan: ${variantName(it.speciesVariant)} on the ${it.speciesRuntime.name}, ${it.speciesThreads} threads, live up to ${it.liveMaxFps} fps. ${it.reason}.", fontSize = 13.sp, color = C.ink2, lineHeight = 18.sp) }
                OutlinedButton(onClick = { vm.replan() }, shape = RoundedCornerShape(10.dp)) { Text("Re-check this phone", color = C.green) }
            }

            SectionLabel("Live detector", Modifier.padding(top = 6.dp))
            Card {
                Text("MobileNetV4 + SSDLite, full INT8 · ${files?.detector?.genera?.size ?: 0} genera · ${files?.let { mb(it, it.detector.model) } ?: ""}", fontSize = 13.sp, color = C.ink2)
                Text("Runtime: ${models.detector}", fontSize = 13.sp, color = C.muted, lineHeight = 18.sp)
                live?.let { Text("Last frame: model %.1f ms + preprocessing %.1f ms".format(it.modelMs, it.preMs), fontSize = 13.sp, color = C.muted) }
                Text("Run on", fontSize = 13.sp, color = C.ink2)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    AccelChoice.entries.forEach { a ->
                        FilterChip(selected = detAcc == a, onClick = { detAcc = a; vm.prefs.detectorAccel = a; vm.reloadModels() },
                            label = { Text(if (a == AccelChoice.AUTO) "Auto (measured)" else a.label) })
                    }
                }
                OutlinedButton(onClick = { detAcc = AccelChoice.AUTO; vm.rerunDetectorBenchmark() }, shape = RoundedCornerShape(10.dp)) {
                    Text("Measure GPU vs CPU again", color = C.green)
                }
                vm.prefs.benchErrors.takeIf { it.isNotBlank() }?.let { Text("Not usable here: $it", fontSize = 12.5.sp, color = C.caution, lineHeight = 17.sp) }
            }

            SectionLabel("Walk mode", Modifier.padding(top = 6.dp))
            Card {
                var dim by remember { mutableStateOf(vm.prefs.walkDim) }
                var buzz by remember { mutableStateOf(vm.prefs.walkHaptics) }
                var rec by remember { mutableStateOf(vm.prefs.recordWalks) }
                Text("Walk mode keeps the camera scanning at ${files?.tuning?.walkFps ?: 4} frames a second with the screen on. When a plant on your forage list " +
                    "stays in view for about a second, BioCLIP checks the frame, and the phone buzzes twice if it agrees. A long buzz means a genus with a poisonous species is in view.",
                    fontSize = 13.sp, color = C.ink2, lineHeight = 18.sp)
                Toggle("Dim the screen", dim) { dim = it; vm.prefs.walkDim = it }
                Toggle("Buzz on alerts", buzz) { buzz = it; vm.prefs.walkHaptics = it }
                Toggle("Record walks for training (about 150 MB an hour)", rec) { rec = it; vm.prefs.recordWalks = it }
                val tiers = vm.prefs.forageTiers
                val n = files?.forage?.genera?.count { (tiers[it.genus] ?: Tier.of(it.tier)) == Tier.ALERT } ?: 0
                Text(if (files?.forage == null) "This model bundle has no forage list; update the models to use walk alerts."
                    else "Forage list: $n genera alert" + if (tiers.isNotEmpty()) " (${tiers.size} changed by you)" else "", fontSize = 13.sp, color = C.muted)
                if (files?.forage != null) OutlinedButton(onClick = { vm.go(Screen.FORAGE) }, shape = RoundedCornerShape(10.dp)) { Text("Edit the forage list", color = C.green) }
            }

            SectionLabel("Species model", Modifier.padding(top = 6.dp))
            Card {
                Text("BioCLIP 2.5 image tower + regional head · ${files?.head?.n ?: 0} species", fontSize = 13.sp, color = C.ink2)
                Text("Runtime: ${models.species}", fontSize = 13.sp, color = C.muted, lineHeight = 18.sp)
                last?.takeIf { it.modelMs > 0 }?.let {
                    Text("Last identification: model %.2f s, total %.2f s (budget 10 s)".format(it.modelMs / 1000, it.totalMs / 1000), fontSize = 13.sp, color = C.muted)
                }
                files?.speciesModels?.variants?.forEach { v ->
                    val installed = v.id == files.variant
                    val on = plan?.speciesVariant == v.id
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Column(Modifier.weight(1f)) {
                            Text(v.label, fontSize = 14.sp, fontWeight = if (on) FontWeight.SemiBold else FontWeight.Normal, color = C.ink)
                            Text("%.0f MB · %s".format(v.bytes / 1048576.0, if (installed) "on this phone" else "download to use"), fontSize = 12.5.sp, color = C.muted)
                        }
                        FilterChip(selected = on, onClick = { vm.chooseVariant(v.id, if (v.accelerator == "GPU") Runtime.GPU else Runtime.CPU) },
                            label = { Text(if (on) "In use" else "Use") })
                    }
                }
                Text("CPU threads", fontSize = 13.sp, color = C.ink2)
                FlowRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(2, 4, 6, 8).forEach { n ->
                        FilterChip(selected = plan?.speciesThreads == n, onClick = { vm.setSpeciesThreads(n) }, label = { Text("$n") })
                    }
                }
                Text("On phones with 2 fast and 6 slow cores (e.g. Dimensity 6300), compare 2 and 4 threads.", fontSize = 12.5.sp, color = C.muted, lineHeight = 17.sp)
            }
            models.error?.let { Text(it, fontSize = 12.sp, color = C.caution) }

            SectionLabel("Bundle", Modifier.padding(top = 6.dp))
            Card {
                Text("Version ${files?.manifest?.version ?: "none"} · variant ${files?.variant ?: "-"} · verified", fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
                val inUse by vm.serverInUse.collectAsState()
                Text(inUse?.let { "Server: $it (${Servers.label(it)})" } ?: "Server: ${vm.prefs.server}", fontSize = 13.sp, color = C.muted)
                if (Servers.tailnet.isNotEmpty()) Text("Away from home it uses the tailnet when Tailscale is on.", fontSize = 12.sp, color = C.muted)
                OutlinedButton(onClick = { vm.go(Screen.SETUP) }, shape = RoundedCornerShape(10.dp)) { Text("Check for a new bundle", color = C.green) }
            }

            SectionLabel("Prototype controls", Modifier.padding(top = 6.dp))
            Card {
                Toggle("Show live boxes", boxes) { boxes = it; vm.prefs.showBoxes = it }
                Toggle("Show timings", timings) { timings = it; vm.prefs.showTimings = it }
                val thr by vm.boxThreshold.collectAsState()
                Text("Plant-score sensitivity: boxes drawn from score %.2f (default 0.25). Higher sensitivity shows weaker detections, and more false boxes.".format(thr),
                    fontSize = 13.sp, color = C.ink2, lineHeight = 18.sp)
                Slider(value = MAX_BOX_THRESHOLD + MIN_BOX_THRESHOLD - thr, onValueChange = { vm.setBoxThreshold(MAX_BOX_THRESHOLD + MIN_BOX_THRESHOLD - it) },
                    valueRange = MIN_BOX_THRESHOLD..MAX_BOX_THRESHOLD,
                    colors = SliderDefaults.colors(thumbColor = C.green, activeTrackColor = C.green))
                Row {
                    Text("Fewer boxes", fontSize = 12.sp, color = C.muted, modifier = Modifier.weight(1f))
                    Text("More boxes", fontSize = 12.sp, color = C.muted)
                }
                OutlinedButton(onClick = { vm.setBoxThreshold(0.25f) }, shape = RoundedCornerShape(10.dp)) { Text("Reset to 0.25", color = C.green) }
                Text("The live label comes from the whole frame (D-033). If the app is stopped while loading or running a model, it steps down to the next fallback on restart (D-035).",
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
private fun Toggle(label: String, on: Boolean, set: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Text(label, fontSize = 14.sp, modifier = Modifier.weight(1f))
        Switch(checked = on, onCheckedChange = set, colors = SwitchDefaults.colors(checkedTrackColor = C.green))
    }
}

/** In-app updates (D-042): app APK from app.json, model bundle from latest.json, both on the home server. */
@Composable
private fun UpdatesCard(vm: AppViewModel) {
    val upd by vm.appUpdate.collectAsState()
    val bundleNew by vm.bundleUpdate.collectAsState()
    val ctx = androidx.compose.ui.platform.LocalContext.current
    SectionLabel("Updates")
    Card {
        Text("App ${vm.updater.currentVersionName} · models ${vm.inUse.collectAsState().value?.manifest?.version ?: "none"}",
            fontSize = 14.sp, fontWeight = FontWeight.SemiBold)
        when (val u = upd) {
            is UpdateState.Idle -> Text("Not checked yet.", fontSize = 13.sp, color = C.muted)
            is UpdateState.Checking -> Text("Checking ${vm.prefs.server}…", fontSize = 13.sp, color = C.muted)
            is UpdateState.UpToDate -> Text("The app is up to date.", fontSize = 13.sp, color = C.muted)
            is UpdateState.Failed -> Text(u.message, fontSize = 13.sp, color = C.caution)
            is UpdateState.Available -> {
                Text("App ${u.release.versionName} is available (%.0f MB).".format(u.release.bytes / 1048576.0), fontSize = 13.sp, color = C.ink2)
                if (u.release.notes.isNotBlank()) Text(u.release.notes, fontSize = 12.5.sp, color = C.muted, lineHeight = 17.sp)
                androidx.compose.material3.Button(onClick = { vm.downloadAppUpdate() }, shape = RoundedCornerShape(10.dp),
                    colors = androidx.compose.material3.ButtonDefaults.buttonColors(containerColor = C.green)) { Text("Download app update") }
            }
            is UpdateState.Downloading -> {
                Text("Downloading ${u.release.versionName}…", fontSize = 13.sp, color = C.ink2)
                androidx.compose.material3.LinearProgressIndicator(progress = { if (u.total > 0) u.done.toFloat() / u.total else 0f },
                    modifier = Modifier.fillMaxWidth(), color = C.green)
            }
            is UpdateState.Ready -> {
                androidx.compose.material3.Button(onClick = { vm.installIntent()?.let { ctx.startActivity(it) } }, shape = RoundedCornerShape(10.dp),
                    colors = androidx.compose.material3.ButtonDefaults.buttonColors(containerColor = C.green)) { Text("Install ${u.release.versionName}") }
                Text(if (vm.updater.canInstall()) "Android shows its installer; your settings, models and log are kept."
                     else "Android first asks to allow installs from Transferner. Allow it, come back, and tap Install again.",
                    fontSize = 12.5.sp, color = C.muted, lineHeight = 17.sp)
            }
        }
        bundleNew?.let { v ->
            Text("New models available (bundle $v). Only changed files are downloaded.", fontSize = 13.sp, color = C.ink2)
            androidx.compose.material3.Button(onClick = { vm.go(Screen.SETUP) }, shape = RoundedCornerShape(10.dp),
                colors = androidx.compose.material3.ButtonDefaults.buttonColors(containerColor = C.green)) { Text("Get new models") }
        }
        OutlinedButton(onClick = { vm.checkUpdates() }, shape = RoundedCornerShape(10.dp)) { Text("Check for updates", color = C.green) }
    }
}
