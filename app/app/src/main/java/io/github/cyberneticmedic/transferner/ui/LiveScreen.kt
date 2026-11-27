package io.github.cyberneticmedic.transferner.ui

import android.Manifest
import android.annotation.SuppressLint
import android.app.Activity
import android.content.pm.PackageManager
import android.util.Size
import android.view.MotionEvent
import android.view.WindowManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.IconButton
import androidx.compose.material3.Slider
import androidx.compose.material3.SliderDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size as GSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalView
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.min
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.compose.LocalLifecycleOwner
import io.github.cyberneticmedic.transferner.R
import io.github.cyberneticmedic.transferner.core.BoxRules
import java.io.ByteArrayInputStream

private val Amber = Color(0xFFFFC56B)
private val Leaf = Color(0xFF8BE0A4)

/**
 * The live screen (D-044 layout): the whole analysed frame is visible. The 3:4 camera frame is fitted between
 * the top bar and the sheet instead of filling the screen behind them, because the live label and boxes are
 * computed on the whole frame and 0.5 hid about two thirds of it (off-screen sides, opaque sheet).
 */
@SuppressLint("ClickableViewAccessibility")
@Composable
fun LiveScreen(vm: AppViewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var granted by remember {
        mutableStateOf(ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)
    }
    val ask = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted = it }
    LaunchedEffect(Unit) { if (!granted) ask.launch(Manifest.permission.CAMERA) }
    val pick = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) {
            val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            if (bytes != null) {
                val deg = runCatching { ExifInterface(ByteArrayInputStream(bytes)).rotationDegrees }.getOrDefault(0)
                vm.identify(bytes, deg)
            }
        }
    }

    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FIT_CENTER } }
    // ~3 MP stills: the species model needs 224 px and the log keeps 1080 px. Quality over latency (D-044):
    // the still is what BioCLIP judges.
    val capture = remember {
        ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
            .setResolutionSelector(ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(2048, 1536), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .build())
            .build()
    }
    var bound by remember { mutableStateOf(false) }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var torch by remember { mutableStateOf(false) }
    DisposableEffect(granted) {
        if (!granted) return@DisposableEffect onDispose {}
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            val provider = future.get()
            val aspect = ResolutionSelector.Builder().setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY).build()
            val small = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER))
                .build()
            val preview = Preview.Builder().setResolutionSelector(aspect).build()
            preview.setSurfaceProvider(previewView.surfaceProvider)
            val analysis = ImageAnalysis.Builder().setResolutionSelector(small)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888).build()
            analysis.setAnalyzer(vm.detectorExecutor) { vm.analyze(it) }
            provider.unbindAll()
            // fallback: if the phone cannot run preview + analysis + capture together, drop the live analysis
            camera = try {
                provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture)
            } catch (e: Exception) {
                try {
                    provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture).also {
                        vm.message.value = "This camera cannot run the live scanner alongside capture; capture still works."
                    }
                } catch (e2: Exception) {
                    vm.message.value = "Camera unavailable (${e2.message}). Identify photos from the gallery instead."
                    null
                }
            }
            bound = camera != null
        }, ContextCompat.getMainExecutor(context))
        onDispose { if (future.isDone) future.get().unbindAll(); camera = null; torch = false }
    }
    // tap to focus and meter on that spot
    LaunchedEffect(camera) {
        val cam = camera ?: return@LaunchedEffect
        previewView.setOnTouchListener { v, e ->
            if (e.action == MotionEvent.ACTION_UP) {
                val point = previewView.meteringPointFactory.createPoint(e.x, e.y)
                cam.cameraControl.startFocusAndMetering(FocusMeteringAction.Builder(point).build())
                v.performClick()
            }
            true
        }
    }

    val walk by vm.walk.collectAsState()
    // Walk mode keeps the screen on (dimmed) so detection continues in the hand (D-044)
    val view = LocalView.current
    val activity = context as? Activity
    DisposableEffect(walk) {
        view.keepScreenOn = walk
        val w = activity?.window
        if (walk && vm.prefs.walkDim && w != null) w.attributes = w.attributes.apply { screenBrightness = 0.08f }
        onDispose {
            view.keepScreenOn = false
            if (w != null) w.attributes = w.attributes.apply { screenBrightness = WindowManager.LayoutParams.BRIGHTNESS_OVERRIDE_NONE }
        }
    }

    val takePicture = {
        capture.takePicture(vm.detectorExecutor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val buf = image.planes[0].buffer
                val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
                val deg = image.imageInfo.rotationDegrees
                image.close()
                vm.identify(bytes, deg, fromCamera = true)
            }
            override fun onError(e: ImageCaptureException) { vm.message.value = "Capture failed: ${e.message}" }
        })
    }

    val frame by vm.live.collectAsState()
    val boxThr by vm.boxThreshold.collectAsState()
    val measurer = rememberTextMeasurer()
    val models by vm.models.collectAsState()
    val busy by vm.busy.collectAsState()
    val alert by vm.alert.collectAsState()
    val second by vm.secondFor.collectAsState()
    val toxicNow by vm.toxicNow.collectAsState()
    val alerts by vm.alertsToday.collectAsState()
    val showBoxes = vm.prefs.showBoxes
    val showTimings = vm.prefs.showTimings
    val f = frame

    Box(Modifier.fillMaxSize().background(C.dark)) {
        Column(Modifier.fillMaxSize()) {
            // ------------------------------------------------------------------------------ top bar
            Row(
                Modifier.fillMaxWidth().statusBarsPadding().padding(start = 16.dp, end = 8.dp, top = 6.dp, bottom = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(if (walk) "Walk mode" else "Transferner", color = Color.White, fontSize = 17.sp, fontWeight = FontWeight.SemiBold,
                    modifier = Modifier.weight(1f))
                val status = when {
                    !models.detectorReady -> "Loading models…"
                    showTimings && f != null -> "%.0f ms · %d box%s · plant %.2f".format(f.modelMs + f.preMs, f.result.boxes.size,
                        if (f.result.boxes.size == 1) "" else "es", f.result.presence)
                    walk -> "Scanning"
                    else -> "Live"
                }
                Row(Modifier.background(Color(0x24FFFFFF), RoundedCornerShape(50)).padding(horizontal = 10.dp, vertical = 5.dp),
                    verticalAlignment = Alignment.CenterVertically) {
                    Box(Modifier.size(8.dp).background(if (models.detectorReady) Color(0xFF7FD69B) else Color(0xFFE0B060), CircleShape))
                    Text("  $status", color = Color.White, fontSize = 13.sp)
                }
                IconButton(onClick = { torch = !torch; camera?.cameraControl?.enableTorch(torch) }, enabled = camera?.cameraInfo?.hasFlashUnit() == true,
                    modifier = Modifier.padding(start = 4.dp).size(40.dp)) {
                    Glyph(R.drawable.ic_light, if (torch) "Turn the light off" else "Turn the light on", if (torch) Amber else Color.White)
                }
            }

            // ------------------------------------------------------------------------------ camera
            BoxWithConstraints(Modifier.fillMaxWidth().weight(1f)) {
                val pw = min(maxWidth, maxHeight * 0.75f)
                Box(Modifier.size(pw, pw / 0.75f).align(Alignment.TopCenter)) {
                    if (granted) AndroidView({ previewView }, Modifier.fillMaxSize())
                    else Text("Camera permission is needed for the live view. You can still identify a photo from the gallery.",
                        color = Color.White, modifier = Modifier.align(Alignment.Center).padding(32.dp))
                    // capture guide (audit capture_model-7): the species model sees only the centre square of the
                    // photo (shortest side, centre crop), so a plant near the top or bottom of a portrait shot is cut
                    if (granted && !walk) Canvas(Modifier.fillMaxSize()) {
                        val side = size.width
                        val top = (size.height - side) / 2
                        val c = Color.White.copy(alpha = 0.35f)
                        val L = 22.dp.toPx()
                        val sw = 1.5.dp.toPx()
                        for ((p0, p1) in listOf(
                            Offset(0f, top) to Offset(L, top), Offset(side - L, top) to Offset(side, top),
                            Offset(0f, top + side) to Offset(L, top + side), Offset(side - L, top + side) to Offset(side, top + side),
                        )) drawLine(c, p0, p1, strokeWidth = sw)
                    }

                    val boxes = f?.result?.boxes.orEmpty()
                    val win = alert?.window
                    if (f != null && ((showBoxes && !walk && boxes.isNotEmpty()) || (walk && win != null))) {
                        Canvas(Modifier.fillMaxSize()) {
                            val k = f.result.size.toFloat()
                            val stroke = 3.dp.toPx()
                            val label = { text: String, at: Offset, bg: Color ->
                                val m = measurer.measure(text, TextStyle(color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.SemiBold))
                                val pad = 6.dp.toPx()
                                val x = at.x.coerceAtMost(size.width - m.size.width - 2 * pad).coerceAtLeast(0f)
                                drawRoundRect(bg, Offset(x, at.y), GSize(m.size.width + 2 * pad, m.size.height + pad), CornerRadius(6.dp.toPx()))
                                drawText(m, topLeft = Offset(x + pad, at.y + pad / 2))
                            }
                            if (walk && win != null) {
                                val r = Rect(win[0] * size.width, win[1] * size.height, win[2] * size.width, win[3] * size.height)
                                drawRoundRect(Leaf, r.topLeft, r.size, CornerRadius(14.dp.toPx()), style = Stroke(width = stroke))
                                label("Possible ${alert?.label}", Offset(r.left + 6.dp.toPx(), r.top + 6.dp.toPx()), Color(0xC00A3A1E))
                            } else for ((i, b) in boxes.withIndex()) {
                                // model pixels (squashed size x size frame) -> the fitted 3:4 preview
                                val r = Rect(b.x0 / k * size.width, b.y0 / k * size.height, b.x1 / k * size.width, b.y1 / k * size.height)
                                val c = Color.White.copy(alpha = 0.55f + 0.45f * b.score)
                                if (i == 0 && BoxRules.fillsFrame(b, f.result.size)) {
                                    val inset = 10.dp.toPx()
                                    val v = Rect(inset, inset, size.width - inset, size.height - inset)
                                    val L = 34.dp.toPx()
                                    for ((p, q) in listOf(
                                        v.topLeft to Offset(v.left + L, v.top), v.topLeft to Offset(v.left, v.top + L),
                                        v.topRight to Offset(v.right - L, v.top), v.topRight to Offset(v.right, v.top + L),
                                        v.bottomLeft to Offset(v.left + L, v.bottom), v.bottomLeft to Offset(v.left, v.bottom - L),
                                        v.bottomRight to Offset(v.right - L, v.bottom), v.bottomRight to Offset(v.right, v.bottom - L),
                                    )) drawLine(c, p, q, strokeWidth = stroke)
                                    label("Plant fills the frame · ${pct(b.score)}", Offset(v.left + 10.dp.toPx(), v.top + 10.dp.toPx()), Color(0xB80A0E0C))
                                } else {
                                    drawRoundRect(c, r.topLeft, r.size, CornerRadius(10.dp.toPx()), style = Stroke(width = stroke))
                                    label("Plant ${pct(b.score)}", Offset(r.left + 6.dp.toPx(), r.top + 6.dp.toPx()), Color(0xB80A0E0C))
                                }
                            }
                        }
                    }

                    // update banner (D-042) and the second-photo banner
                    val upd by vm.appUpdate.collectAsState()
                    val bundleNew by vm.bundleUpdate.collectAsState()
                    val updText = when {
                        upd is UpdateState.Available || upd is UpdateState.Ready || upd is UpdateState.Downloading ->
                            "App update available" + if (bundleNew != null) " · new models" else ""
                        bundleNew != null -> "New models available"
                        else -> null
                    }
                    Column(Modifier.align(Alignment.TopCenter).padding(top = 8.dp), horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        if (updText != null && !walk) Pill("$updText · tap to update", Color.White, C.ink) { vm.go(Screen.SETTINGS) }
                        if (second != null) Pill("Second photo: a different part of the same plant · tap to cancel", Amber, C.ink) { vm.cancelSecondPhoto() }
                    }

                    // a confirmed walk alert
                    alert?.let { a ->
                        Column(
                            Modifier.align(Alignment.BottomCenter).padding(10.dp).fillMaxWidth()
                                .background(Color(0xF0102A1B), RoundedCornerShape(16.dp)).padding(14.dp),
                            verticalArrangement = Arrangement.spacedBy(4.dp),
                        ) {
                            Text("Possible ${a.label}", color = Color.White, fontSize = 19.sp, fontWeight = FontWeight.SemiBold)
                            Text("${a.genus} · live ${pct(a.score)}" + (if (a.bioclip.isNotBlank()) " · BioCLIP: ${a.bioclip}" else ""),
                                color = Color(0xFFD2E6D9), fontSize = 13.sp)
                            if (a.note.isNotBlank()) Text(a.note.replaceFirstChar { it.uppercase() }, color = Amber, fontSize = 13.sp, lineHeight = 17.sp)
                            Text("Capture a clear photo to check it. Nothing here means a plant is safe to eat.", color = Color(0xFFB7C9BD),
                                fontSize = 12.sp, lineHeight = 16.sp)
                            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.padding(top = 4.dp)) {
                                Pill("Identify", Color.White, C.ink) { if (busy == null && bound) takePicture() }
                                Pill("Dismiss", Color(0x33FFFFFF), Color.White) { vm.dismissAlert() }
                            }
                        }
                    }
                }
            }

            // ------------------------------------------------------------------------------ sheet
            Column(
                Modifier.fillMaxWidth().background(C.sheet, RoundedCornerShape(topStart = 22.dp, topEnd = 22.dp))
                    .navigationBarsPadding().padding(start = 18.dp, end = 18.dp, top = 12.dp, bottom = 10.dp),
                verticalArrangement = Arrangement.spacedBy(7.dp),
            ) {
                val focusMeta = vm.inUse.collectAsState().value?.focus
                val focus by vm.focus.collectAsState()
                if (focusMeta != null && !walk) {
                    Row(Modifier.fillMaxWidth().horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                        FocusChip("Any plant", focus == null) { vm.setFocus(null) }
                        focusMeta.categories.forEach { c -> FocusChip(c.label, focus == c.id) { vm.setFocus(if (focus == c.id) null else c.id) } }
                    }
                }
                // Live warning flags (D-038): any live guess whose genus holds a cited toxic species, from 2%;
                // false positives preferred. Held for a few frames (D-044). The check is the capture.
                val warn = f?.result?.warnings.orEmpty()
                if (warn.isNotEmpty()) {
                    Row(Modifier.fillMaxWidth().background(C.cautionSoft, RoundedCornerShape(10.dp)).padding(horizontal = 10.dp, vertical = 6.dp),
                        verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Glyph(R.drawable.ic_warning, null, C.caution, 18)
                        Text(
                            (if (warn.any { it.serious }) "Possibly poisonous: " else "Possibly mildly toxic: ") +
                                warn.take(3).joinToString(", ") { "${it.genus} ${pct(it.p)}" } + " · capture to check",
                            color = C.caution, fontSize = 13.sp, fontWeight = FontWeight.SemiBold, lineHeight = 17.sp)
                    }
                }
                val files = vm.inUse.collectAsState().value
                val top = f?.result?.genera?.firstOrNull()
                // plant-presence gate (D-044): below it the frame does not look like any known plant
                val plant = f == null || f.result.presence >= (files?.tuning?.presenceBoxes ?: 0f)
                if (walk) {
                    // forage chip from the accumulated score (no haptic below the alert threshold)
                    val fs = f?.forage
                    val chipMin = files?.tuning?.chipMin ?: 0.3f
                    val byIdx = remember(files) { files?.forage?.genera?.associateBy { it.g } }
                    val chip = when {
                        fs == null -> null
                        f.forageSmoothed >= chipMin && fs.alertGenus >= 0 -> byIdx?.get(fs.alertGenus)
                        fs.show >= chipMin && fs.showGenus >= 0 -> byIdx?.get(fs.showGenus)
                        else -> null
                    }
                    SectionLabelDark("Scanning for forage · ${alerts.count { it.confirmed }} alert${if (alerts.count { it.confirmed } == 1) "" else "s"} today" +
                        if (toxicNow) " · poisonous genus in view" else "")
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Text(chip?.let { "Maybe ${it.label.ifBlank { it.genus }}" } ?: if (!plant) "No plant in view" else "Nothing on your list",
                            color = if (chip != null) Leaf else Color.White, fontSize = 24.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                        if (top != null) Text("${top.genus} ${pct(top.p)}", style = Scientific, color = C.onDark2, fontSize = 13.sp)
                    }
                    chip?.note?.takeIf { it.isNotBlank() }?.let {
                        Text("${chip.genus}: $it", color = C.onDark2, fontSize = 12.sp, lineHeight = 16.sp)
                    }
                } else {
                    // genus when the head is fairly sure, else the family (D-036)
                    val sure = top != null && top.p >= 0.5f
                    SectionLabelDark(if (!plant) "Live guess · no plant in view?" else if (top == null || sure) "Live guess · whole frame" else "Live guess · family (genus unsure)")
                    Row(verticalAlignment = Alignment.Bottom) {
                        if (top == null || sure) {
                            Text(top?.genus ?: "—", style = Scientific, color = Color.White, fontSize = 26.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                            Text(top?.let { pct(it.p) } ?: "", color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
                        } else {
                            Text(f!!.result.topFamily, color = Color.White, fontSize = 24.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                            Text(pct(f.result.topFamilyP), color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.SemiBold)
                        }
                    }
                    Row(Modifier.horizontalScroll(rememberScrollState()), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        f?.result?.genera?.drop(if (sure) 1 else 0)?.take(3)?.forEach {
                            Text("${it.genus} ${pct(it.p)}", style = Scientific, color = Color(0xFFE4EBE6), fontSize = 13.sp,
                                modifier = Modifier.background(Color(0x1AFFFFFF), RoundedCornerShape(50)).padding(horizontal = 10.dp, vertical = 4.dp))
                        }
                    }
                    if (showTimings && showBoxes) {
                        // Plant-score sensitivity (D-039): right = more sensitive (lower score needed for a box)
                        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                            Text("Box sensitivity", color = C.onDark2, fontSize = 13.sp)
                            Slider(
                                value = MAX_BOX_THRESHOLD + MIN_BOX_THRESHOLD - boxThr,
                                onValueChange = { vm.setBoxThreshold(MAX_BOX_THRESHOLD + MIN_BOX_THRESHOLD - it) },
                                valueRange = MIN_BOX_THRESHOLD..MAX_BOX_THRESHOLD,
                                colors = SliderDefaults.colors(thumbColor = Color.White, activeTrackColor = Color.White, inactiveTrackColor = Color(0x40FFFFFF)),
                                modifier = Modifier.weight(1f).height(28.dp)
                                    .semantics { contentDescription = "Box sensitivity: boxes shown from plant score %.2f".format(boxThr) },
                            )
                            Text("≥ %.2f".format(boxThr), color = Color.White, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
                        }
                    }
                }
                Row(Modifier.fillMaxWidth().padding(top = 4.dp), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) {
                    if (walk) RoundButton(R.drawable.ic_flag, "Mark a plant the app missed") { vm.markMissed() }
                    else RoundButton(R.drawable.ic_list, "Observation log") { vm.go(Screen.LOG) }
                    RoundButton(R.drawable.ic_photo, "Identify a photo from the gallery") {
                        pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                    }
                    Box(
                        Modifier.size(70.dp).border(4.dp, Color.White, CircleShape).padding(8.dp).background(Color.White, CircleShape)
                            .semantics { contentDescription = "Capture and identify"; role = Role.Button }
                            .clickable(enabled = busy == null && granted && bound) { takePicture() },
                    )
                    RoundButton(R.drawable.ic_walk, if (walk) "Stop walk mode" else "Start walk mode", if (walk) Leaf else Color.White,
                        if (walk) Color(0x408BE0A4) else Color(0x1FFFFFFF)) { vm.setWalk(!walk) }
                    RoundButton(R.drawable.ic_tune, "Models and settings") { vm.go(Screen.SETTINGS) }
                }
            }
        }
        if (busy != null) {
            Box(Modifier.fillMaxSize().background(Color(0x99000000)), contentAlignment = Alignment.Center) {
                Column(horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    CircularProgressIndicator(color = Color.White)
                    Text(busy ?: "", color = Color.White, fontSize = 15.sp)
                }
            }
        }
    }
}

@Composable
private fun Pill(text: String, bg: Color, fg: Color, onClick: () -> Unit) =
    Text(text, color = fg, fontSize = 13.sp, fontWeight = FontWeight.SemiBold,
        modifier = Modifier.background(bg, RoundedCornerShape(50)).clickable(onClick = onClick).padding(horizontal = 14.dp, vertical = 8.dp)
            .semantics { role = Role.Button })

@Composable
private fun FocusChip(label: String, on: Boolean, onClick: () -> Unit) {
    Text(label, color = if (on) C.ink else Color.White, fontSize = 13.sp, fontWeight = if (on) FontWeight.SemiBold else FontWeight.Normal,
        modifier = Modifier.background(if (on) Color.White else Color(0x24FFFFFF), RoundedCornerShape(50))
            .clickable(onClick = onClick).padding(horizontal = 12.dp, vertical = 6.dp)
            .semantics { role = Role.Button; contentDescription = "Focus: $label" + if (on) ", selected" else "" })
}

@Composable
private fun SectionLabelDark(text: String) =
    Text(text.uppercase(), color = Color(0xFFB7C3BB), fontSize = 12.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.6.sp)

@Composable
private fun RoundButton(icon: Int, desc: String, tint: Color = Color.White, bg: Color = Color(0x1FFFFFFF), onClick: () -> Unit) {
    IconButton(onClick = onClick, modifier = Modifier.size(50.dp).background(bg, CircleShape)) {
        Glyph(icon, desc, tint)
    }
}
