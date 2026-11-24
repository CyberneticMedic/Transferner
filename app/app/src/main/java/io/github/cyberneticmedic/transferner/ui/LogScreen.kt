package io.github.cyberneticmedic.transferner.ui

import android.content.Intent
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.FileProvider
import io.github.cyberneticmedic.transferner.R

@Composable
fun LogScreen(vm: AppViewModel) {
    val entries by vm.entries.collectAsState()
    val ctx = LocalContext.current
    Column(Modifier.fillMaxSize().background(C.paper).safeDrawingPadding()) {
        TopBar("Observation log", R.drawable.ic_back, { vm.back() }) {
            OutlinedButton(
                onClick = {
                    if (!vm.log.csv.exists()) { vm.message.value = "Nothing logged yet."; return@OutlinedButton }
                    val uri = FileProvider.getUriForFile(ctx, ctx.packageName + ".files", vm.log.csv)
                    val send = Intent(Intent.ACTION_SEND).setType("text/csv").putExtra(Intent.EXTRA_STREAM, uri)
                        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                    ctx.startActivity(Intent.createChooser(send, "Export observations"))
                },
                shape = RoundedCornerShape(10.dp), modifier = Modifier.padding(end = 8.dp),
            ) { Text("Export CSV", color = C.green, fontWeight = FontWeight.SemiBold) }
        }
        LazyColumn(Modifier.padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
            items(entries) { e ->
                val top = e.top.firstOrNull()
                Row(
                    Modifier.fillMaxWidth().background(C.card, RoundedCornerShape(14.dp)).border(1.dp, C.line, RoundedCornerShape(14.dp))
                        .clickable { vm.openEntry(e) }.padding(12.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Box(Modifier.size(72.dp).clip(RoundedCornerShape(10.dp)).background(C.noneSoft)) {
                        rememberThumb(e.photo)?.let { Image(it, null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
                    }
                    Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(3.dp)) {
                        Row {
                            Text(top?.species?.displayName ?: "?", fontSize = 15.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                            Text(top?.let { pct(it.p) } ?: "", fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                        }
                        Text(top?.species?.name ?: "", style = Scientific, fontSize = 13.sp, color = C.muted)
                        Text(e.time.replace('T', ' ').removeSuffix("Z") + " UTC", fontSize = 12.5.sp, color = C.muted)
                        if (e.caution) TraitChip("Caution candidate in top 5", io.github.cyberneticmedic.transferner.core.ChipKind.CAUTION)
                    }
                }
            }
            item {
                Column(Modifier.fillMaxWidth().background(C.noneSoft, RoundedCornerShape(14.dp)).padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(if (entries.isEmpty()) "No saved captures yet" else "Saved on this phone", fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = C.ink2)
                    Text("Each entry keeps the photo, the top 5 with scores, the caution flag and the model versions. Location is not recorded in this build.",
                        fontSize = 13.sp, color = C.none, lineHeight = 18.sp)
                }
            }
        }
    }
}
