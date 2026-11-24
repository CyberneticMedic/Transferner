package io.github.cyberneticmedic.transferner.ui

import android.content.Intent
import android.net.Uri
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
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalUriHandler
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextDecoration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.cyberneticmedic.transferner.R
import io.github.cyberneticmedic.transferner.core.Species
import io.github.cyberneticmedic.transferner.core.Trait
import io.github.cyberneticmedic.transferner.core.TraitSummary

private val ALL_PARTS = listOf("leaf", "flower", "fruit", "seed", "root", "stem", "bark", "sap")

@Composable
fun SpeciesScreen(vm: AppViewModel) {
    val sp by vm.species.collectAsState()
    val s = sp ?: return
    val res by vm.result.collectAsState()
    val here = res?.top?.firstOrNull { it.species.i == s.i }
    val others = res?.top?.filter { it.species.i != s.i }.orEmpty()
    val t = TraitSummary.of(s)
    var tab by remember(s.i) { mutableIntStateOf(0) }
    val ctx = LocalContext.current
    val uri = LocalUriHandler.current

    Column(Modifier.fillMaxSize().background(C.paper)) {
        Box(Modifier.fillMaxWidth().height(230.dp).background(C.noneSoft)) {
            rememberThumb(vm.thumbFile(s))?.let { Image(it, null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize()) }
            IconButton(onClick = { vm.back() }, modifier = Modifier.statusBarsPadding().padding(12.dp).size(44.dp).background(Color(0xEBF6F5F0), CircleShape)) {
                Glyph(R.drawable.ic_back, "Back", C.ink)
            }
            s.thumb?.let {
                Text("${it.by} · ${it.license}", color = Color.White, fontSize = 11.sp,
                    modifier = Modifier.align(Alignment.BottomEnd).padding(8.dp).background(Color(0xB80C110E), RoundedCornerShape(6.dp)).padding(horizontal = 7.dp, vertical = 3.dp))
            }
        }
        Column(Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 18.dp).navigationBarsPadding()) {
            Row(Modifier.padding(top = 16.dp), verticalAlignment = Alignment.CenterVertically) {
                Text(s.displayName, fontSize = 24.sp, fontWeight = FontWeight.SemiBold, color = C.ink, modifier = Modifier.weight(1f))
                here?.let {
                    Text("${pct(it.p)} match", fontSize = 13.sp, fontWeight = FontWeight.SemiBold, color = C.green,
                        modifier = Modifier.background(C.greenSoft, RoundedCornerShape(50)).padding(horizontal = 9.dp, vertical = 4.dp))
                }
            }
            Text(s.name, style = Scientific, fontSize = 18.sp, color = C.ink2)
            Text("${s.genus} · ${s.family}", fontSize = 14.sp, color = C.muted)

            Row(Modifier.padding(top = 16.dp).fillMaxWidth().background(C.noneSoft, RoundedCornerShape(12.dp)).padding(4.dp),
                horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                listOf("Parts", "In your result", "Sources").forEachIndexed { i, label ->
                    val on = tab == i
                    Box(
                        Modifier.weight(1f).height(40.dp).background(if (on) Color.White else Color.Transparent, RoundedCornerShape(9.dp))
                            .clickable { tab = i },
                        contentAlignment = Alignment.Center,
                    ) { Text(label, fontSize = 13.5.sp, fontWeight = if (on) FontWeight.SemiBold else FontWeight.Medium, color = if (on) C.ink else C.none) }
                }
            }
            Column(Modifier.padding(vertical = 14.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                when (tab) {
                    0 -> {
                        if (t.ingestionToxic || t.contactToxic) {
                            Column(Modifier.fillMaxWidth().background(C.cautionSoft, RoundedCornerShape(12.dp)).padding(14.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                                    Glyph(R.drawable.ic_warning, null, C.caution)
                                    Text(t.chip, color = C.caution, fontSize = 16.sp, fontWeight = FontWeight.SemiBold)
                                }
                                Text(t.detail.replaceFirstChar { it.uppercase() } + ".", color = C.caution, fontSize = 13.sp, lineHeight = 18.sp)
                                Text(if (t.contactToxic) "Skin contact: listed as harmful." else "Skin contact: no listing.", color = C.caution, fontSize = 13.sp)
                                Text("Severity is the source's rating (NC State: low, medium, high).", color = C.caution, fontSize = 12.sp)
                            }
                        }
                        if (!t.hasData) Note("No trait data", "No cited source in this bundle covers this species. That does not mean it is safe.")
                        s.traits.groupBy { Triple(it.category, it.conditions, it.symptoms) }.forEach { (_, rows) -> TraitCard(rows) }
                        val listed = s.traits.map { it.part }.toSet()
                        val missing = ALL_PARTS.filter { it !in listed }
                        if (t.hasData && missing.isNotEmpty())
                            Note("${missing.joinToString(", ").replaceFirstChar { it.uppercase() }}: no listing",
                                "Transferner only shows what a cited source states. A part with no listing is not known to be safe.")
                        if (t.serious && t.ingestionToxic) {
                            Button(
                                onClick = { ctx.startActivity(Intent(Intent.ACTION_DIAL, Uri.parse("tel:18002221222"))) },
                                colors = ButtonDefaults.buttonColors(containerColor = C.ink), shape = RoundedCornerShape(12.dp),
                                modifier = Modifier.fillMaxWidth().height(52.dp),
                            ) {
                                Glyph(R.drawable.ic_call, null, Color.White, 20)
                                Text("  Poison Control (US) 1-800-222-1222", fontWeight = FontWeight.SemiBold)
                            }
                        }
                    }
                    1 -> {
                        if (others.isEmpty()) Text("Open this page from a match to compare the other candidates.", color = C.muted, fontSize = 14.sp)
                        others.forEach { c ->
                            val ct = TraitSummary.of(c.species)
                            Row(
                                Modifier.fillMaxWidth().background(if (ct.needsCaution) C.cautionSoft else C.card, RoundedCornerShape(12.dp))
                                    .border(1.dp, if (ct.needsCaution) C.cautionSoft else C.line, RoundedCornerShape(12.dp))
                                    .clickable { vm.openSpecies(c.species) }.padding(12.dp),
                                horizontalArrangement = Arrangement.spacedBy(12.dp), verticalAlignment = Alignment.CenterVertically,
                            ) {
                                Thumb(vm, c.species, 48)
                                Column(Modifier.weight(1f)) {
                                    Text("${c.species.displayName} · ${pct(c.p)}", fontSize = 15.sp, fontWeight = FontWeight.SemiBold,
                                        color = if (ct.needsCaution) C.caution else C.ink)
                                    Text(ct.chip, fontSize = 13.sp, color = if (ct.needsCaution) C.caution else C.muted)
                                }
                            }
                        }
                        if (s.lookalikeGroups.isEmpty())
                            Text("No cited look-alike notes are on file for this species yet.", fontSize = 13.sp, color = C.none, lineHeight = 18.sp)
                    }
                    else -> {
                        val sources = s.traits.map { Triple(it.source, it.url, it.accessed) }.distinct()
                        SectionLabel("Traits")
                        if (sources.isEmpty()) Text("No trait source covers this species.", fontSize = 14.sp, color = C.ink2)
                        sources.forEach { (name, url, acc) ->
                            Text(name, fontSize = 14.sp, color = C.green, textDecoration = TextDecoration.Underline,
                                modifier = Modifier.clickable(enabled = url.isNotBlank()) { uri.openUri(url) })
                            Text("Accessed $acc", fontSize = 13.sp, color = C.muted)
                        }
                        SectionLabel("Photo", Modifier.padding(top = 6.dp))
                        Text(s.thumb?.let { "${it.by}, ${it.license}" + if (it.ref.isNotBlank()) "\n${it.ref}" else "" } ?: "No licensed photo in this bundle.",
                            fontSize = 13.sp, color = C.ink2)
                        SectionLabel("Match", Modifier.padding(top = 6.dp))
                        Text("BioCLIP 2.5 on this phone, with a head trained on regional photos. Calibrated score, not a certainty.",
                            fontSize = 13.sp, color = C.ink2, lineHeight = 18.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun TraitCard(rows: List<Trait>) {
    val r = rows.first()
    val parts = rows.map { it.part }.distinct().joinToString(", ").replaceFirstChar { it.uppercase() }
    val (label, kind) = when (r.category) {
        "edible" -> "Edible · cited" to io.github.cyberneticmedic.transferner.core.ChipKind.EDIBLE
        "conditional" -> "Edible only with conditions" to io.github.cyberneticmedic.transferner.core.ChipKind.CAUTION
        else -> ((if (r.contactToxic == 1) "Toxic to touch and eat" else "Toxic if eaten") + (if (r.severity.isNotBlank()) " · ${r.severity}" else "")) to
            io.github.cyberneticmedic.transferner.core.ChipKind.CAUTION
    }
    Column(Modifier.fillMaxWidth().background(C.card, RoundedCornerShape(12.dp)).border(1.dp, C.line, RoundedCornerShape(12.dp)).padding(14.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(parts, fontSize = 16.sp, fontWeight = FontWeight.SemiBold, color = C.ink, modifier = Modifier.weight(1f))
            TraitChip(label, kind)
        }
        if (r.conditions.isNotBlank()) Quote(r.conditions)
        if (r.symptoms.isNotBlank()) { SectionLabel("Symptoms, as listed"); Quote(r.symptoms) }
        if (r.toxicPrinciple.isNotBlank()) Text("Toxic principle: ${r.toxicPrinciple}", fontSize = 13.5.sp, color = C.ink2)
        if (r.genusLevel) Text("From the genus page: this species is not assessed on its own.", fontSize = 13.sp, color = C.caution)
        r.remarks.forEach { Text(it.replaceFirstChar { c -> c.uppercase() } + ".", fontSize = 13.sp, color = C.muted) }
        Text(r.source, fontSize = 12.sp, color = C.muted)
    }
}

@Composable
private fun Quote(text: String) = Text("“$text”", fontSize = 14.sp, color = C.ink2, lineHeight = 21.sp)

@Composable
private fun Note(title: String, body: String) {
    Column(Modifier.fillMaxWidth().background(C.noneSoft, RoundedCornerShape(12.dp)).padding(horizontal = 14.dp, vertical = 12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)) {
        Text(title, fontSize = 14.sp, fontWeight = FontWeight.SemiBold, color = C.ink2)
        Text(body, fontSize = 13.sp, color = C.none, lineHeight = 18.sp)
    }
}
