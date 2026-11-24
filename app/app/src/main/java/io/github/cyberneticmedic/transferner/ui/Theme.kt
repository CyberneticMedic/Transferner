package io.github.cyberneticmedic.transferner.ui

import androidx.annotation.DrawableRes
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import io.github.cyberneticmedic.transferner.core.ChipKind

/** Colours from the design mock (contrast checked there: every text pair is at least 4.5:1). */
object C {
    val paper = Color(0xFFF6F5F0)
    val card = Color.White
    val ink = Color(0xFF18201B)
    val ink2 = Color(0xFF2E3832)
    val muted = Color(0xFF55615A)
    val line = Color(0xFFE3E1D9)
    val green = Color(0xFF1E5B3A)
    val greenSoft = Color(0xFFE4EFE7)
    val caution = Color(0xFF8F3F00)
    val cautionSoft = Color(0xFFFCE9D4)
    val none = Color(0xFF4E5852)
    val noneSoft = Color(0xFFECEBE5)
    val dark = Color(0xFF0E1411)
    val sheet = Color(0xEB0C110E)
    val onDark2 = Color(0xFFC3CEC7)
}

val Scientific = TextStyle(fontFamily = FontFamily.Serif, fontStyle = FontStyle.Italic)

@Composable
fun TransfernerTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = lightColorScheme(primary = C.green, onPrimary = Color.White, background = C.paper,
            surface = C.paper, onSurface = C.ink, onBackground = C.ink),
        content = content,
    )
}

@Composable
fun Glyph(@DrawableRes id: Int, desc: String?, tint: Color, size: Int = 22) =
    Icon(painterResource(id), contentDescription = desc, tint = tint, modifier = Modifier.size(size.dp))

@Composable
fun TopBar(title: String, @DrawableRes backIcon: Int, onBack: () -> Unit, action: @Composable () -> Unit = {}) {
    Row(Modifier.fillMaxWidth().height(56.dp).padding(horizontal = 4.dp), verticalAlignment = Alignment.CenterVertically) {
        IconButton(onClick = onBack) { Glyph(backIcon, "Back", C.ink) }
        Text(title, fontSize = 17.sp, fontWeight = FontWeight.SemiBold, color = C.ink, modifier = Modifier.weight(1f))
        action()
    }
}

@Composable
fun TraitChip(text: String, kind: ChipKind) {
    val (fg, bg) = when (kind) {
        ChipKind.EDIBLE -> C.green to C.greenSoft
        ChipKind.CAUTION -> C.caution to C.cautionSoft
        ChipKind.NONE -> C.none to C.noneSoft
    }
    Box(Modifier.background(bg, RoundedCornerShape(6.dp)).padding(horizontal = 7.dp, vertical = 2.dp)) {
        Text(text, color = fg, fontSize = 11.5.sp, fontWeight = FontWeight.SemiBold)
    }
}

@Composable
fun SectionLabel(text: String, modifier: Modifier = Modifier) =
    Text(text.uppercase(), color = C.muted, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.6.sp, modifier = modifier)

fun pct(p: Float): String = if (p < 0.01f) "<1%" else "${Math.round(p * 100)}%"
