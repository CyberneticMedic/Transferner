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
    val idx = vm.warnIndex
    val an = res.analysis
    // reopened log entries have only the top 5: the D-038 rule on them; fresh results: the D-044 banner
    val caution = an?.caution?.inTop ?: cautionCandidates(res.top, idx)
    val extra = an?.caution?.also.orEmpty()
    val showCaution = an?.caution?.show ?: (caution.isNotEmpty() || res.storedCaution)
    var note by remember(res) { mutableStateOf("") }
    Column(Modifier.fillMaxSize().background(C.paper).safeDrawingPadding()) {
        TopBar(if (res.fromLog) "Saved match" else if ((an?.photos ?: 1) > 1) "Match from 2 photos" else "Species match", R.drawable.ic_back, { vm.back() })
        LazyColumn(Modifier.weight(1f).padding(horizontal = 16.dp), verticalArrangement = Arrangement.spacedBy(0.dp)) {
            item {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Image(res.photo.asImageBitmap(), contentDescription = "Your photo", contentScale = ContentScale.Crop,
                        modifier = Modifier.weight(1f).height(170.dp).clip(RoundedCornerShape(14.dp)))
                    res.secondPhoto?.let {
                        Image(it.asImageBitmap(), contentDescription = "Your second photo", contentScale = ContentScale.Crop,
                            modifier = Modifier.weight(1f).height(170.dp).clip(RoundedCornerShape(14.dp)))
                    }
                }
            }
            an?.verdict?.let { v -> item { VerdictCard(v) } }
            if (an == null) {
                val top1 = res.top.firstOrNull()?.p ?: 0f
                if (top1 < 0.5f) item { Hint("Low confidence (top match ${pct(top1)})", LOW_TIP) }
            }
            if (showCaution) item {
                val strong = an?.caution?.strong == true
                Row(
                    Modifier.padding(top = 12.dp).fillMaxWidth().background(C.cautionSoft, RoundedCornerShape(12.dp))
                        .then(if (strong) Modifier.border(2.dp, C.caution, RoundedCornerShape(12.dp)) else Modifier)
                        .padding(horizontal = 14.dp, vertical = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    Glyph(R.drawable.ic_warning, null, C.caution)
                    Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                        Text(if (caution.isEmpty() && extra.isEmpty()) "Caution: toxic look-alikes possible" else cautionHeadline(caution + extra),
                            color = C.caution, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                        caution.forEach { (c, t) ->
                            Text("${c.species.displayName} (${pct(c.p)}): ${t.detail}.", color = C.caution, fontSize = 13.sp, lineHeight = 18.sp)
                        }
                        if (extra.isNotEmpty()) {
                            Text("Also possible, outside the top 5:", color = C.caution, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                            extra.forEach { (c, t) ->
                                Text("${c.species.displayName} (${pct(c.p)}): ${t.detail}.", color = C.caution, fontSize = 13.sp, lineHeight = 18.sp,
                                    modifier = Modifier.clickable { vm.openSpecies(c.species) })
                            }
                        }
                        an?.caution?.let { Text("Toxic species and their relatives together: ${pct(it.sum)} of the match.", color = C.caution, fontSize = 12.sp) }
                        Text("Compare closely before you handle or eat anything.", color = C.caution, fontSize = 13.sp, lineHeight = 18.sp)
                    }
                }
            }
            // second photo of a different part (D-044): asked on edible/toxic conflicts, offered when unsure
            if (an != null && an.photos == 1 && !res.fromLog && (an.secondPhoto || an.verdict.p1 < 0.8f)) item {
                Column(
                    Modifier.padding(top = 12.dp).fillMaxWidth()
                        .background(if (an.secondPhoto) C.cautionSoft else C.greenSoft, RoundedCornerShape(12.dp)).padding(horizontal = 14.dp, vertical = 12.dp),
                    verticalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    Text(if (an.secondPhoto) "Possible look-alike: add a photo of a different part" else "Not sure: another photo helps",
                        color = if (an.secondPhoto) C.caution else C.green, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
                    Text("Photograph the same plant again, showing another part: the stem, a leaf underside, a flower or fruit. The two photos are combined.",
                        color = C.ink2, fontSize = 13.sp, lineHeight = 18.sp)
                    OutlinedButton(onClick = { vm.startSecondPhoto(res) }, shape = RoundedCornerShape(10.dp)) {
                        Text("Take a second photo", color = C.green, fontWeight = FontWeight.SemiBold)
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

private const val LOW_TIP = "Try daylight or the flash, fill the frame with one plant, and include flowers, fruit or a whole leaf. Yellowing or damaged leaves are hard to match."

@Composable
private fun Hint(title: String, body: String) {
    Column(
        Modifier.padding(top = 12.dp).fillMaxWidth().background(C.noneSoft, RoundedCornerShape(12.dp)).padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        Text(title, color = C.ink2, fontSize = 15.sp, fontWeight = FontWeight.SemiBold)
        Text(body, color = C.none, fontSize = 13.sp, lineHeight = 18.sp)
    }
}

/**
 * The headline level (D-044): species when p1 >= 0.8, "<Genus> sp." when the genus sum >= 0.8, else no
 * confident match; one level lower when the photo matches the species names poorly (text score).
 */
@Composable
private fun VerdictCard(v: io.github.cyberneticmedic.transferner.core.Verdict) {
    Column(
        Modifier.padding(top = 12.dp).fillMaxWidth().background(C.card, RoundedCornerShape(12.dp)).border(1.dp, C.line, RoundedCornerShape(12.dp))
            .padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(3.dp),
    ) {
        when (v.level) {
            io.github.cyberneticmedic.transferner.core.Level.SPECIES -> {
                Text(v.species.displayName, color = C.ink, fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
                Text("${v.species.name} · ${pct(v.p1)}", style = Scientific, color = C.muted, fontSize = 14.sp)
            }
            io.github.cyberneticmedic.transferner.core.Level.GENUS -> {
                Text("${v.genus} sp.", style = Scientific, color = C.ink, fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
                Text("Genus ${pct(v.genusP)}; the species is unsure (best ${v.species.displayName}, ${pct(v.p1)}).", color = C.muted, fontSize = 13.sp, lineHeight = 18.sp)
            }
            io.github.cyberneticmedic.transferner.core.Level.UNCERTAIN -> {
                Text(if (v.maybeOutside) "Maybe not a plant I know" else "No confident match", color = C.ink, fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
                Text(if (v.maybeOutside) "This photo matches none of the 4,411 regional species well. It may be a garden or house plant I was not trained on, so its toxicity is unknown: treat it with caution."
                    else LOW_TIP, color = if (v.maybeOutside) C.caution else C.none, fontSize = 13.sp, lineHeight = 18.sp)
            }
        }
        if (v.demoted && !v.maybeOutside) Text("Shown one level lower: the photo matches the species names less well than usual.", color = C.muted, fontSize = 12.sp, lineHeight = 16.sp)
    }
}

@Composable
private fun CandidateRow(vm: AppViewModel, c: Candidate) {
    val t = TraitSummary.of(c.species, vm.warnIndex)
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
