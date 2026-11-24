package io.github.cyberneticmedic.transferner.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.cyberneticmedic.transferner.R
import io.github.cyberneticmedic.transferner.core.Candidate
import io.github.cyberneticmedic.transferner.core.Species
import io.github.cyberneticmedic.transferner.core.TraitSummary
import io.github.cyberneticmedic.transferner.core.cautionCandidates
import io.github.cyberneticmedic.transferner.core.cautionHeadline
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

@Composable
fun rememberThumb(file: File?): ImageBitmap? {
    val v by produceState<ImageBitmap?>(null, file) {
        value = file?.takeIf { it.exists() }?.let { f -> withContext(Dispatchers.IO) { BitmapFactory.decodeFile(f.path)?.asImageBitmap() } }
    }
    return v
}

@Composable
fun Thumb(vm: AppViewModel, s: Species, size: Int) {
    val img = rememberThumb(vm.thumbFile(s))
    Box(Modifier.size(size.dp).clip(RoundedCornerShape(10.dp)).background(C.noneSoft)) {
        img?.let { Image(it, contentDescription = null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
    }
}

@Composable
fun ResultScreen(vm: AppViewModel) {
    val r by vm.result.collectAsState()
    val res = r ?: return
    val caution = cautionCandidates(res.top)
    var note by remember(res) { mutableStateOf("") }
    Column(Modifier.fillMaxSize().background(C.paper).safeDrawingPadding()) {
        TopBar(if (res.fromLog) "Saved match" else "Species match", R.drawable.ic_back, { vm.back() })
        LazyColumn(Modifier.weight(1f).padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(0.dp)) {
            item {
                Image(res.photo.asImageBitmap(), contentDescription = "Your photo", contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxWidth().height(170.dp).clip(RoundedCornerShape(14.dp)))
            }
            val top1 = res.top.firstOrNull()?.p ?: 0f
            if (top1 < 0.5f) item {
                Column(
                    Modifier.padding(top = 12.dp).fillMaxWidth().background(C.noneSoft, RoundedCornerShape(12.dp)).padding(horizontal = 14.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(4.dp),
                ) {
                    Text("Low confidence (top match ${pct(top1)})", color = C.ink2, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                    Text("Try daylight or the flash, fill the frame with one plant, and include flowers, fruit or a whole leaf. Yellowing or damaged leaves are hard to match.",
                        color = C.none, fontSize = 13.sp, lineHeight = 18.sp)
                }
            }
            if (caution.isNotEmpty()) item {
                Row(
                    Modifier.padding(top = 12.dp).fillMaxWidth().background(C.cautionSoft, RoundedCornerShape(12.dp)).padding(horizontal = 14.dp, vertical = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Glyph(R.drawable.ic_warning, null, C.caution)
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(cautionHeadline(caution), color = C.caution, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                        caution.forEach { (c, t) ->
                            Text("${c.species.displayName} (${pct(c.p)}): ${t.detail}.", color = C.caution, fontSize = 13.sp, lineHeight = 18.sp)
                        }
                        Text("Compare closely before you handle or eat anything.", color = C.caution, fontSize = 13.sp, lineHeight = 18.sp)
                    }
                }
            }
            item { SectionLabel("Top 5 matches", Modifier.padding(top = 14.dp, bottom = 4.dp)) }
            items(res.top) { c -> CandidateRow(vm, c) }
            item {
                Text(
                    (if (res.modelMs > 0) "Species model %.1f s, total %.1f s. ".format(res.modelMs / 1000, res.totalMs / 1000) else "") +
                        "Scores are calibrated over 4,411 regional species. Nothing on this screen means a plant is safe to eat.",
                    fontSize = 12.sp, color = C.muted, lineHeight = 17.sp, modifier = Modifier.padding(vertical = 12.dp),
                )
            }
        }
        if (!res.saved) {
            OutlinedTextField(value = note, onValueChange = { note = it }, singleLine = true,
                label = { Text("Note (optional): what it really is, conditions…") },
                modifier = Modifier.fillMaxWidth().padding(start = 16.dp, end = 16.dp, top = 8.dp))
        }
        Row(Modifier.padding(16.dp), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            OutlinedButton(onClick = { vm.go(Screen.LIVE) }, shape = RoundedCornerShape(12.dp), modifier = Modifier.weight(1f).height(50.dp)) {
                Text("Retake", color = C.green, fontWeight = FontWeight.SemiBold)
            }
            Button(
                onClick = { vm.saveResult(note) }, enabled = !res.saved, shape = RoundedCornerShape(12.dp),
                colors = ButtonDefaults.buttonColors(containerColor = C.green), modifier = Modifier.weight(1f).height(50.dp),
            ) { Text(if (res.saved) "Saved" else "Save to log", fontWeight = FontWeight.SemiBold) }
        }
    }
}

@Composable
private fun CandidateRow(vm: AppViewModel, c: Candidate) {
    val t = TraitSummary.of(c.species)
    Row(
        Modifier.fillMaxWidth().clickable { vm.openSpecies(c.species) }.padding(vertical = 8.dp),
        horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically,
    ) {
        Thumb(vm, c.species, 52)
        Column(Modifier.weight(1f), verticalArrangement = Arrangement.spacedBy(2.dp)) {
            Text(c.species.displayName, fontSize = 15.sp, fontWeight = FontWeight.SemiBold, color = C.ink)
            Text(c.species.name, style = Scientific, fontSize = 13.sp, color = C.muted)
            TraitChip(t.chip, t.chipKind)
        }
        Column(horizontalAlignment = Alignment.End, verticalArrangement = Arrangement.spacedBy(6.dp)) {
            Text(pct(c.p), fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = C.ink)
            Box(Modifier.width(64.dp).height(5.dp).background(C.line, RoundedCornerShape(3.dp))) {
                Box(Modifier.fillMaxHeight().width((64 * c.p).coerceAtLeast(1f).dp).background(C.green, RoundedCornerShape(3.dp)))
            }
        }
    }
    Box(Modifier.fillMaxWidth().height(1.dp).background(C.line))
}
