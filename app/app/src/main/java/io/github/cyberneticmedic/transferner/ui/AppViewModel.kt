package io.github.cyberneticmedic.transferner.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.litert.Environment
import io.github.cyberneticmedic.transferner.core.Candidate
import io.github.cyberneticmedic.transferner.core.LiveResult
import io.github.cyberneticmedic.transferner.core.Species
import io.github.cyberneticmedic.transferner.core.SpeciesHead
import io.github.cyberneticmedic.transferner.core.cautionCandidates
import io.github.cyberneticmedic.transferner.data.BundleFiles
import io.github.cyberneticmedic.transferner.data.BundleRepository
import io.github.cyberneticmedic.transferner.data.LogEntry
import io.github.cyberneticmedic.transferner.data.ObservationLog
import io.github.cyberneticmedic.transferner.data.Prefs
import io.github.cyberneticmedic.transferner.data.Progress
import io.github.cyberneticmedic.transferner.ml.LiveDetector
import io.github.cyberneticmedic.transferner.ml.SpeciesIdentifier
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

enum class Screen { SETUP, LIVE, RESULT, SPECIES, LOG, SETTINGS }

sealed interface BundleState {
    data object Missing : BundleState
    data class Downloading(val p: Progress?) : BundleState
    data class Failed(val message: String) : BundleState
    data class Ready(val files: BundleFiles) : BundleState
}

data class ModelStatus(
    val detector: String = "not loaded",
    val species: String = "not loaded",
    val detectorReady: Boolean = false,
    val speciesReady: Boolean = false,
    val error: String? = null,
)

data class LiveFrame(val result: LiveResult, val frameW: Int, val frameH: Int, val modelMs: Float, val preMs: Float)

data class IdResult(
    val photo: Bitmap,
    val jpeg: ByteArray,
    val top: List<Candidate>,
    val modelMs: Float,
    val totalMs: Float,
    val fromLog: Boolean = false,
    val saved: Boolean = false,
)

class AppViewModel(app: Application) : AndroidViewModel(app) {
    val prefs = Prefs(app)
    private val repo = BundleRepository(File(app.filesDir, "bundle"))
    val log = ObservationLog.of(app)

    /** Each model is created and run on its own thread (LiteRT GPU contexts are thread-bound). */
    val detectorExecutor = Executors.newSingleThreadExecutor { Thread(it, "detector") }
    private val speciesExecutor = Executors.newSingleThreadExecutor { Thread(it, "species") }
    private val speciesDispatcher = speciesExecutor.asCoroutineDispatcher()

    private var env: Environment? = null
    @Volatile private var detector: LiveDetector? = null
    private var identifier: SpeciesIdentifier? = null

    val screen = MutableStateFlow(Screen.SETUP)
    val bundle = MutableStateFlow<BundleState>(BundleState.Missing)
    val models = MutableStateFlow(ModelStatus())
    val live = MutableStateFlow<LiveFrame?>(null)
    val busy = MutableStateFlow<String?>(null)
    val result = MutableStateFlow<IdResult?>(null)
    val species = MutableStateFlow<Species?>(null)
    val entries = MutableStateFlow<List<LogEntry>>(emptyList())
    val message = MutableStateFlow<String?>(null)
    private val history = ArrayDeque<Screen>()

    init {
        viewModelScope.launch {
            val files = withContext(Dispatchers.IO) { runCatching { repo.load() }.getOrNull() }
            if (files != null) {
                bundle.value = BundleState.Ready(files)
                screen.value = Screen.LIVE
                loadModels(files)
            }
        }
    }

    fun go(s: Screen) {
        if (s == screen.value) return
        history.addLast(screen.value)
        screen.value = s
        if (s == Screen.LOG) refreshLog()
    }

    /** Returns false when there is nowhere to go back to (the activity then finishes). */
    fun back(): Boolean {
        val prev = history.removeLastOrNull() ?: return false
        screen.value = prev
        return true
    }

    fun download(server: String) {
        prefs.server = server
        bundle.value = BundleState.Downloading(null)
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) { repo.download(server) { bundle.value = BundleState.Downloading(it) } }
                val files = withContext(Dispatchers.IO) { repo.load() } ?: error("bundle incomplete after download")
                bundle.value = BundleState.Ready(files)
                history.clear()
                screen.value = Screen.LIVE
                loadModels(files)
            } catch (e: Exception) {
                bundle.value = BundleState.Failed(e.message ?: e.toString())
            }
        }
    }

    fun loadModels(files: BundleFiles) {
        models.value = ModelStatus(detector = "loading…", species = "loading…")
        detectorExecutor.execute {
            try {
                detector?.close(); detector = null
                val d = LiveDetector(files.dir, files.detector, prefs.detectorAccel, environment())
                detector = d
                models.value = models.value.copy(detector = "${d.accelerator} · loaded in ${d.loadMs} ms", detectorReady = true)
            } catch (e: Throwable) {
                models.value = models.value.copy(detector = "failed: ${e.message}", error = e.toString())
            }
        }
        speciesExecutor.execute {
            try {
                identifier?.close(); identifier = null
                val head = SpeciesHead(files.head, File(files.dir, files.head.file).readBytes())
                val s = SpeciesIdentifier(files.dir, files.species, head, files.catalog, prefs.speciesAccel, environment())
                identifier = s
                models.value = models.value.copy(species = "${s.accelerator} · loaded in ${s.loadMs} ms", speciesReady = true)
            } catch (e: Throwable) {
                models.value = models.value.copy(species = "failed: ${e.message}", error = e.toString())
            }
        }
    }

    @Synchronized private fun environment(): Environment = env ?: Environment.create().also { env = it }

    fun reloadModels() {
        (bundle.value as? BundleState.Ready)?.let { loadModels(it.files) }
    }

    // ---------------------------------------------------------------------------------------- live
    private var px = IntArray(0)

    /** CameraX analyzer (on detectorExecutor): one RGBA frame -> upright pixels -> detector. */
    fun analyze(proxy: ImageProxy) {
        try {
            val d = detector ?: return
            if (screen.value != Screen.LIVE) return
            val up = upright(proxy.toBitmap(), proxy.imageInfo.rotationDegrees)
            val n = up.width * up.height
            if (px.size != n) px = IntArray(n)
            up.getPixels(px, 0, up.width, 0, 0, up.width, up.height)
            val r = d.run(px, up.width, up.height)
            live.value = LiveFrame(r, up.width, up.height, d.lastMs, d.lastPreMs)
        } catch (e: Throwable) {
            message.value = "Live detector: ${e.message}"
        } finally {
            proxy.close()
        }
    }

    private fun upright(b: Bitmap, deg: Int): Bitmap =
        if (deg == 0) b else Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postRotate(deg.toFloat()) }, false)

    // ------------------------------------------------------------------------------------ identify
    /** A JPEG from the camera (with its rotation) or the photo picker (EXIF rotation) -> species top 5. */
    fun identify(jpeg: ByteArray, rotation: Int) {
        if (identifier == null) { message.value = "The species model is still loading."; return }
        busy.value = "Identifying…"
        viewModelScope.launch {
            try {
                val r = withContext(speciesDispatcher) {
                    val t0 = System.nanoTime()
                    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, bounds)
                    // keep the short side >= 448 px: enough for the 224 px antialiased resize
                    var sample = 1
                    while (min(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 448) sample *= 2
                    val raw = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, BitmapFactory.Options().apply { inSampleSize = sample })
                        ?: error("could not decode the photo")
                    val bmp = upright(raw, rotation)
                    val pixels = IntArray(bmp.width * bmp.height)
                    bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
                    val (top, t) = identifier!!.identify(pixels, bmp.width, bmp.height)
                    val shown = scaled(bmp, 1080)
                    val out = ByteArrayOutputStream()
                    shown.compress(Bitmap.CompressFormat.JPEG, 88, out)
                    IdResult(shown, out.toByteArray(), top, t.modelMs, (System.nanoTime() - t0) / 1e6f)
                }
                result.value = r
                go(Screen.RESULT)
            } catch (e: Throwable) {
                message.value = "Identification failed: ${e.message}"
            } finally {
                busy.value = null
            }
        }
    }

    private fun scaled(b: Bitmap, maxSide: Int): Bitmap {
        val s = maxSide.toFloat() / max(b.width, b.height)
        return if (s >= 1f) b else Bitmap.createScaledBitmap(b, (b.width * s).toInt(), (b.height * s).toInt(), true)
    }

    fun saveResult() {
        val r = result.value ?: return
        val files = (bundle.value as? BundleState.Ready)?.files ?: return
        viewModelScope.launch(Dispatchers.IO) {
            log.add(r.jpeg, r.top, cautionCandidates(r.top).isNotEmpty(), files.manifest.version, files.species.variant, files.detector.model)
            result.value = r.copy(saved = true)
            message.value = "Saved to the log."
        }
    }

    fun openSpecies(s: Species) {
        species.value = s
        go(Screen.SPECIES)
    }

    fun refreshLog() {
        val files = (bundle.value as? BundleState.Ready)?.files ?: return
        viewModelScope.launch(Dispatchers.IO) { entries.value = log.entries(files.catalog) }
    }

    fun openEntry(e: LogEntry) {
        viewModelScope.launch {
            val r = withContext(Dispatchers.IO) {
                val bytes = e.photo.readBytes()
                val bmp = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                IdResult(bmp, bytes, e.top, 0f, 0f, fromLog = true, saved = true)
            }
            result.value = r
            go(Screen.RESULT)
        }
    }

    fun thumbFile(s: Species): File? {
        val files = (bundle.value as? BundleState.Ready)?.files ?: return null
        return s.thumb?.let { File(files.dir, it.file) }
    }

    override fun onCleared() {
        detectorExecutor.execute { detector?.close() }
        speciesExecutor.execute { identifier?.close() }
        detectorExecutor.shutdown(); speciesExecutor.shutdown()
    }
}
