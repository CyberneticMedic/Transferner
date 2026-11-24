package io.github.cyberneticmedic.transferner.ui

import android.Manifest
import android.content.pm.PackageManager
import android.util.Size
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
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
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.IconButton
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
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.role
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.compose.LocalLifecycleOwner
import io.github.cyberneticmedic.transferner.R
import java.io.ByteArrayInputStream
import kotlin.math.max

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

    val previewView = remember { PreviewView(context).apply { scaleType = PreviewView.ScaleType.FILL_CENTER } }
    // ~3 MP stills: the species model needs 224 px and the log keeps 1080 px, so full 50 MP captures would
    // only cost time and memory on budget phones
    val capture = remember {
        ImageCapture.Builder().setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setResolutionSelector(ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(2048, 1536), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .build())
            .build()
    }
    var bound by remember { mutableStateOf(false) }
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
            bound = try {
                provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture); true
            } catch (e: Exception) {
                try {
                    provider.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture)
                    vm.message.value = "This camera cannot run the live scanner alongside capture; capture still works."
                    true
                } catch (e2: Exception) {
                    vm.message.value = "Camera unavailable (${e2.message}). Identify photos from the gallery instead."
                    false
                }
            }
        }, ContextCompat.getMainExecutor(context))
        onDispose { if (future.isDone) future.get().unbindAll() }
    }

    val frame by vm.live.collectAsState()
    val models by vm.models.collectAsState()
    val busy by vm.busy.collectAsState()
    val showBoxes = vm.prefs.showBoxes
    val showTimings = vm.prefs.showTimings

    Box(Modifier.fillMaxSize().background(C.dark)) {
        if (granted) AndroidView({ previewView }, Modifier.fillMaxSize())
        else Text("Camera permission is needed for the live view. You can still identify a photo from the gallery.",
            color = Color.White, modifier = Modifier.align(Alignment.Center).padding(32.dp))

        // detector boxes, mapped from the squashed model input back onto the FILL_CENTER preview
        val f = frame
        if (showBoxes && f != null && f.result.boxes.isNotEmpty()) {
            Canvas(Modifier.fillMaxSize()) {
                val s = max(size.width / f.frameW, size.height / f.frameH)
                val ox = (size.width - f.frameW * s) / 2
                val oy = (size.height - f.frameH * s) / 2
                val k = f.result.size.toFloat()
                for (b in f.result.boxes) {
                    val x0 = ox + b.x0 / k * f.frameW * s
                    val y0 = oy + b.y0 / k * f.frameH * s
                    val x1 = ox + b.x1 / k * f.frameW * s
                    val y1 = oy + b.y1 / k * f.frameH * s
                    val c = 28.dp.toPx()
                    val st = Stroke(width = 3.dp.toPx())
                    val w = Color.White.copy(alpha = 0.4f + 0.6f * b.score)
                    for ((p, q) in listOf(
                        Offset(x0, y0) to Offset(x0 + c, y0), Offset(x0, y0) to Offset(x0, y0 + c),
                        Offset(x1, y0) to Offset(x1 - c, y0), Offset(x1, y0) to Offset(x1, y0 + c),
                        Offset(x0, y1) to Offset(x0 + c, y1), Offset(x0, y1) to Offset(x0, y1 - c),
                        Offset(x1, y1) to Offset(x1 - c, y1), Offset(x1, y1) to Offset(x1, y1 - c),
                    )) drawLine(w, p, q, strokeWidth = st.width)
                }
            }
        }

        Row(
            Modifier.fillMaxWidth().background(Color(0x9E0A0E0C)).statusBarsPadding().padding(horizontal = 16.dp, vertical = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Transferner", color = Color.White, fontSize = 17.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
            val status = when {
                !models.detectorReady -> "Loading models…"
                showTimings && f != null -> "Live · %.0f ms".format(f.modelMs + f.preMs)
                else -> "Live"
            }
            Row(Modifier.background(Color(0x24FFFFFF), RoundedCornerShape(50)).padding(horizontal = 10.dp, vertical = 5.dp),
                verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(8.dp).background(if (models.detectorReady) Color(0xFF7FD69B) else Color(0xFFE0B060), CircleShape))
                Text("  $status", color = Color.White, fontSize = 13.sp)
            }
        }

        Column(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth()
                .background(C.sheet, RoundedCornerShape(topStart = 22.dp, topEnd = 22.dp))
                .navigationBarsPadding().padding(start = 20.dp, end = 20.dp, top = 18.dp, bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            // Genus when the head is fairly sure, else the family (D-036: on regional photos the shown label
            // is right 62% of the time this way, against 56% for always showing the genus)
            val top = f?.result?.genera?.firstOrNull()
            val sure = top != null && top.p >= 0.5f
            SectionLabelDark(if (top == null || sure) "Live guess · whole frame" else "Live guess · family (genus unsure)")
            Row(verticalAlignment = Alignment.Bottom) {
                if (top == null || sure) {
                    Text(top?.genus ?: "—", style = Scientific, color = Color.White, fontSize = 30.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                    Text(top?.let { pct(it.p) } ?: "", color = Color.White, fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
                } else {
                    Text(f!!.result.topFamily, color = Color.White, fontSize = 28.sp, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                    Text(pct(f.result.topFamilyP), color = Color.White, fontSize = 22.sp, fontWeight = FontWeight.SemiBold)
                }
            }
            Text(
                when {
                    f == null || top == null -> "Point the camera at a plant"
                    sure -> "${f.result.topFamily} · family ${pct(f.result.topFamilyP)}"
                    else -> "Genus guesses:"
                },
                color = Color(0xFFD2DBD5), fontSize = 14.sp)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                f?.result?.genera?.drop(if (sure) 1 else 0)?.take(3)?.forEach {
                    Text("${it.genus} ${pct(it.p)}", style = Scientific, color = Color(0xFFE4EBE6), fontSize = 13.sp,
                        modifier = Modifier.background(Color(0x1AFFFFFF), RoundedCornerShape(50)).padding(horizontal = 10.dp, vertical = 5.dp))
                }
            }
            Text("Live guesses are genus-level and rough. Capture a still for a species match.", color = C.onDark2, fontSize = 13.sp, lineHeight = 18.sp)
            Row(Modifier.fillMaxWidth().padding(top = 6.dp), horizontalArrangement = Arrangement.SpaceEvenly, verticalAlignment = Alignment.CenterVertically) {
                RoundButton(R.drawable.ic_list, "Observation log") { vm.go(Screen.LOG) }
                RoundButton(R.drawable.ic_photo, "Identify a photo from the gallery") {
                    pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                }
                Box(
                    Modifier.size(78.dp).border(4.dp, Color.White, CircleShape).padding(9.dp).background(Color.White, CircleShape)
                        .semantics { contentDescription = "Capture and identify"; role = Role.Button }
                        .clickable(enabled = busy == null && granted && bound) {
                            capture.takePicture(vm.detectorExecutor, object : ImageCapture.OnImageCapturedCallback() {
                                override fun onCaptureSuccess(image: ImageProxy) {
                                    val buf = image.planes[0].buffer
                                    val bytes = ByteArray(buf.remaining()).also { buf.get(it) }
                                    val deg = image.imageInfo.rotationDegrees
                                    image.close()
                                    vm.identify(bytes, deg)
                                }
                                override fun onError(e: ImageCaptureException) { vm.message.value = "Capture failed: ${e.message}" }
                            })
                        },
                )
                RoundButton(R.drawable.ic_tune, "Models and settings") { vm.go(Screen.SETTINGS) }
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
private fun SectionLabelDark(text: String) =
    Text(text.uppercase(), color = Color(0xFFB7C3BB), fontSize = 12.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.6.sp)

@Composable
private fun RoundButton(icon: Int, desc: String, onClick: () -> Unit) {
    IconButton(onClick = onClick, modifier = Modifier.size(52.dp).background(Color(0x1FFFFFFF), CircleShape)) {
        Glyph(icon, desc, Color.White)
    }
}
