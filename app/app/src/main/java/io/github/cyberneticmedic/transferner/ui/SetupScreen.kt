package io.github.cyberneticmedic.transferner.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.cyberneticmedic.transferner.R

@Composable
fun SetupScreen(vm: AppViewModel) {
    val state by vm.bundle.collectAsState()
    val device by vm.device.collectAsState()
    val plan by vm.plan.collectAsState()
    val needed by vm.neededVariant.collectAsState()
    var server by remember { mutableStateOf(vm.prefs.server) }
    val downloading = state is BundleState.Downloading
    Column(
        Modifier.fillMaxSize().background(C.paper).safeDrawingPadding().verticalScroll(rememberScrollState())
            .padding(horizontal = 22.dp, vertical = 24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp),
    ) {
        Text("Transferner", fontFamily = FontFamily.Serif, fontWeight = FontWeight.SemiBold, fontSize = 36.sp, color = C.green)
        Text("Plant identification for South Carolina, North Carolina, Georgia and Tennessee.", fontSize = 16.sp, color = C.ink2, lineHeight = 22.sp)
        Fact(R.drawable.ic_photo, C.green, "Works offline once set up. Both models run on this phone; photos never leave it.")
        Fact(R.drawable.ic_list, C.green, "4,411 regional plants, including mosses and liverworts. No fungi or lichens.")
        Fact(R.drawable.ic_warning, C.caution, "It never says a plant is safe. It shows matches, cited trait notes and cautions, and the decision stays yours.")
        Column(
            Modifier.fillMaxWidth().background(C.card, RoundedCornerShape(14.dp)).border(1.dp, C.line, RoundedCornerShape(14.dp)).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("Model bundle", fontSize = 16.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                Text("0.4 to 0.7 GB", fontSize = 13.sp, color = C.muted)
            }
            OutlinedTextField(
                value = server, onValueChange = { server = it }, singleLine = true, enabled = !downloading,
                label = { Text("Bundle server (home network)") },
                keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Uri), modifier = Modifier.fillMaxWidth(),
            )
            Text("Use Wi-Fi. Every file is checked against the bundle's SHA-256 manifest before it is used.", fontSize = 12.5.sp, color = C.muted, lineHeight = 18.sp)
            device?.let { Text("This phone: ${it.label}", fontSize = 12.5.sp, color = C.ink2, lineHeight = 18.sp) }
            Text(
                plan?.let { "Species model for this phone: ${variantName(it.speciesVariant)} on the ${it.speciesRuntime.name} (${it.reason})." }
                    ?: "The app picks the species-model variant that fits this phone: GPU on flagships, CPU (INT8 or INT4) elsewhere.",
                fontSize = 12.5.sp, color = C.ink2, lineHeight = 18.sp)
            needed?.let {
                Text("This phone needs the ${variantName(it)} species model instead of the one installed. Download it to continue (the shared files are kept).",
                    fontSize = 13.sp, color = C.caution, lineHeight = 18.sp)
            }
            when (val s = state) {
                is BundleState.Downloading -> {
                    val p = s.p
                    if (p == null || p.totalBytes == 0L) LinearProgressIndicator(Modifier.fillMaxWidth(), color = C.green)
                    else LinearProgressIndicator(progress = { p.doneBytes.toFloat() / p.totalBytes }, modifier = Modifier.fillMaxWidth(), color = C.green)
                    Text(p?.let { "%.0f / %.0f MB · %s · variant %s".format(it.doneBytes / 1048576.0, it.totalBytes / 1048576.0, it.file, it.variant) } ?: "Connecting…",
                        fontSize = 12.5.sp, color = C.muted)
                }
                is BundleState.Failed -> Text("Download failed: ${s.message}", fontSize = 13.sp, color = C.caution)
                else -> {}
            }
        }
        Spacer(Modifier.height(4.dp))
        Button(
            onClick = { vm.download(server) }, enabled = !downloading && server.isNotBlank(),
            colors = ButtonDefaults.buttonColors(containerColor = C.green), shape = RoundedCornerShape(12.dp),
            modifier = Modifier.fillMaxWidth().height(52.dp),
        ) { Text(if (downloading) "Downloading…" else "Download and continue", fontSize = 15.sp, fontWeight = FontWeight.SemiBold) }
        Text("Prototype 0.3 · camera permission is asked next", fontSize = 12.sp, color = C.muted, modifier = Modifier.align(Alignment.CenterHorizontally))
    }
}

@Composable
private fun Fact(icon: Int, tint: androidx.compose.ui.graphics.Color, text: String) {
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
        Glyph(icon, null, tint)
        Text(text, fontSize = 14.5.sp, color = C.ink, lineHeight = 20.sp)
    }
}

fun variantName(id: String) = when (id) {
    "gpu" -> "GPU (INT8 weights)"
    "cpu" -> "CPU (dynamic INT8)"
    "cpu-small" -> "CPU low-memory (INT4)"
    else -> id
}
