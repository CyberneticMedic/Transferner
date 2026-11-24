package io.github.cyberneticmedic.transferner.ui

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.os.SystemClock
import androidx.camera.core.ImageProxy
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.google.ai.edge.litert.Environment
import io.github.cyberneticmedic.transferner.core.Candidate
import io.github.cyberneticmedic.transferner.core.DeviceInfo
import io.github.cyberneticmedic.transferner.core.DevicePlanner
import io.github.cyberneticmedic.transferner.core.LiveResult
import io.github.cyberneticmedic.transferner.core.Plan
import io.github.cyberneticmedic.transferner.core.Runtime
import io.github.cyberneticmedic.transferner.core.Species
import io.github.cyberneticmedic.transferner.core.SpeciesHead
import io.github.cyberneticmedic.transferner.core.cautionCandidates
import io.github.cyberneticmedic.transferner.data.BundleFiles
import io.github.cyberneticmedic.transferner.data.BundleRepository
import io.github.cyberneticmedic.transferner.data.DeviceProfile
import io.github.cyberneticmedic.transferner.data.LogEntry
import io.github.cyberneticmedic.transferner.data.ObservationLog
import io.github.cyberneticmedic.transferner.data.Prefs
import io.github.cyberneticmedic.transferner.data.Progress
import io.github.cyberneticmedic.transferner.ml.AccelChoice
import io.github.cyberneticmedic.transferner.ml.LiveDetector
import io.github.cyberneticmedic.transferner.ml.SpeciesIdentifier
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.flow.MutableStateFlow
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

/**
 * App state, and the fallback chain for phones from a Pixel 9 Pro down to a Moto G Power 2025 (D-035):
 * plan the species-model variant from the device profile, benchmark the detector's runtimes on the phone,
 * guard every native load and run with a crash flag and step the plan down after a crash, pause the live
 * detector during an identification, and cap the live frame rate on slow phones.
 */
class AppViewModel(app: Application) : AndroidViewModel(app) {
    val prefs = Prefs(app)
    private val repo = BundleRepository(File(app.filesDir, "bundle"))
    private val xnnCache = File(app.filesDir, "xnnpack").apply { mkdirs() }
    val log = ObservationLog.of(app)

    /** Each model is created and run on its own thread (LiteRT GPU contexts are thread-bound). */
    val detectorExecutor = Executors.newSingleThreadExecutor { Thread(it, "detector") }
    private val speciesExecutor = Executors.newSingleThreadExecutor { Thread(it, "species") }
    private val speciesDispatcher = speciesExecutor.asCoroutineDispatcher()

    private var env: Environment? = null
    @Volatile private var detector: LiveDetector? = null
    private var identifier: SpeciesIdentifier? = null
    private var speciesLoaded = CompletableDeferred<Boolean>()

    val screen = MutableStateFlow(Screen.SETUP)
    val bundle = MutableStateFlow<BundleState>(BundleState.Missing)
    val models = MutableStateFlow(ModelStatus())
    val live = MutableStateFlow<LiveFrame?>(null)
    val busy = MutableStateFlow<String?>(null)
    val result = MutableStateFlow<IdResult?>(null)
    val species = MutableStateFlow<Species?>(null)
    val entries = MutableStateFlow<List<LogEntry>>(emptyList())
    val message = MutableStateFlow<String?>(null)
    val device = MutableStateFlow<DeviceInfo?>(null)
    val plan = MutableStateFlow(prefs.plan)
    /** Set when a crash or failure means this phone needs another species-model variant downloaded. */
    val neededVariant = MutableStateFlow<String?>(null)
    private val history = ArrayDeque<Screen>()

    init {
        viewModelScope.launch {
            device.value = withContext(Dispatchers.Default) { DeviceProfile.read(app, runCatching { environment() }.getOrNull()) }
            val files = withContext(Dispatchers.IO) { runCatching { repo.load() }.getOrNull() }
            recoverFromCrash(files)
            if (files != null) {
                bundle.value = BundleState.Ready(files)
                val p = plan.value
                if (p != null && p.speciesVariant != files.variant) {
                    neededVariant.value = p.speciesVariant              // a fallback picked a variant not on the phone yet
                    screen.value = Screen.SETUP
                } else {
                    screen.value = Screen.LIVE
                }
                loadModels(files)
            }
        }
    }

    // --------------------------------------------------------------------------------- navigation
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

    // ------------------------------------------------------------------------- planning & fallback
    /** The plan for this phone given the server's variants: the stored one (it may be a fallback) or a fresh one. */
    private fun planFor(models: io.github.cyberneticmedic.transferner.core.SpeciesModels): Plan {
        val stored = plan.value
        if (stored != null && models[stored.speciesVariant] != null) return stored
        val d = device.value ?: DeviceProfile.read(getApplication(), runCatching { environment() }.getOrNull())
        return DevicePlanner.plan(d, models.variants).also { setPlan(it) }
    }

    private fun setPlan(p: Plan) {
        prefs.plan = p
        plan.value = p
    }

    /** A native crash leaves prefs.pending set: step the plan down instead of crashing the same way again. */
    private fun recoverFromCrash(files: BundleFiles?) {
        val pending = prefs.pending ?: return
        prefs.pending = null
        when {
            pending.startsWith("detector:") -> {
                prefs.detectorAccel = AccelChoice.CPU
                prefs.detectorBenchRuntime = Runtime.CPU
                message.value = "The live detector stopped the app last time (${pending.substringAfter(':')}); it now runs on the CPU."
            }
            pending.startsWith("species:") -> {
                val p = plan.value ?: return
                val next = files?.let { DevicePlanner.fallback(p, it.speciesModels.variants) }
                if (next == null) {
                    message.value = "The species model stopped the app last time and there is no smaller fallback. Close other apps and try again."
                    return
                }
                setPlan(next)
                message.value = "The species model stopped the app last time. Switched to: ${next.reason}."
            }
        }
    }

    /** Re-profile the phone and re-plan (Settings). Clears the detector benchmark so it runs again. */
    fun replan() {
        val files = (bundle.value as? BundleState.Ready)?.files ?: return
        val d = DeviceProfile.read(getApplication(), runCatching { environment() }.getOrNull())
        device.value = d
        val p = DevicePlanner.plan(d, files.speciesModels.variants)
        setPlan(p)
        prefs.detectorBenchRuntime = null
        if (p.speciesVariant != files.variant) { neededVariant.value = p.speciesVariant; go(Screen.SETUP) } else loadModels(files)
    }

    /** Settings: use a specific variant (download it if it is not on the phone). */
    fun chooseVariant(id: String, runtime: Runtime) {
        val files = (bundle.value as? BundleState.Ready)?.files ?: return
        val p = (plan.value ?: DevicePlanner.plan(device.value ?: return, files.speciesModels.variants))
            .copy(speciesVariant = id, speciesRuntime = runtime, reason = "chosen in Settings")
        setPlan(p)
        if (id != files.variant) { neededVariant.value = id; go(Screen.SETUP) } else loadModels(files)
    }

    fun setSpeciesThreads(n: Int) {
        plan.value?.let { setPlan(it.copy(speciesThreads = n)) }
        reloadModels()
    }

    // ------------------------------------------------------------------------------------ download
    fun download(server: String) {
        prefs.server = server
        bundle.value = BundleState.Downloading(null)
        viewModelScope.launch {
            try {
                withContext(Dispatchers.IO) {
                    repo.download(server, { models -> planFor(models).speciesVariant }) { bundle.value = BundleState.Downloading(it) }
                }
                val files = withContext(Dispatchers.IO) { repo.load() } ?: error("bundle incomplete after download")
                bundle.value = BundleState.Ready(files)
                neededVariant.value = null
                history.clear()
                screen.value = Screen.LIVE
                loadModels(files)
            } catch (e: Exception) {
                bundle.value = BundleState.Failed(e.message ?: e.toString())
            }
        }
    }

    // ------------------------------------------------------------------------------------- models
    fun loadModels(files: BundleFiles) {
        val p = plan.value ?: DevicePlanner.plan(device.value ?: DeviceProfile.read(getApplication(), null), files.speciesModels.variants).also { setPlan(it) }
        models.value = ModelStatus(detector = "loading…", species = "loading…")
        if (speciesLoaded.isCompleted) speciesLoaded = CompletableDeferred()
        detectorExecutor.execute { loadDetector(files, p) }
        speciesExecutor.execute { loadSpecies(files, p) }
    }

    /** Detector runtime: the user's choice, or (AUTO) the faster of GPU and CPU measured on this phone once. */
    private fun loadDetector(files: BundleFiles, p: Plan) {
        try {
            detector?.close(); detector = null
            val choice = prefs.detectorAccel
            val gpuOk = device.value?.gpuAvailable == true
            var bench = prefs.detectorBenchRuntime
            if (choice == AccelChoice.AUTO && bench == null) {
                val times = mutableMapOf<Runtime, Float>()
                for (r in if (gpuOk) listOf(Runtime.CPU, Runtime.GPU) else listOf(Runtime.CPU)) {
                    models.value = models.value.copy(detector = "measuring ${r.name}…")
                    prefs.pending = "detector:${r.name}"
                    try {
                        LiveDetector(files.dir, files.detector, r, p.detectorThreads, environment()).use { d -> times[r] = benchmark(d) }
                    } catch (e: Exception) {
                        times.remove(r)
                    } finally {
                        prefs.pending = null
                    }
                }
                bench = times.minByOrNull { it.value }?.key ?: Runtime.CPU
                prefs.detectorBenchRuntime = bench
                prefs.detectorBench = times.entries.joinToString(", ") { "${it.key.name} %.0f ms".format(it.value) }
            }
            val runtime = when (choice) {
                AccelChoice.GPU -> Runtime.GPU
                AccelChoice.CPU -> Runtime.CPU
                AccelChoice.AUTO -> bench ?: Runtime.CPU
            }
            prefs.pending = "detector:${runtime.name}"
            val d = try {
                LiveDetector(files.dir, files.detector, runtime, p.detectorThreads, environment())
            } catch (e: Exception) {
                if (runtime == Runtime.CPU) throw e
                LiveDetector(files.dir, files.detector, Runtime.CPU, p.detectorThreads, environment())   // GPU refused: CPU
            } finally {
                prefs.pending = null
            }
            detector = d
            models.value = models.value.copy(detector = "${d.accelerator} · loaded in ${d.loadMs} ms" +
                (if (prefs.detectorBench.isNotBlank()) " · measured ${prefs.detectorBench}" else ""), detectorReady = true)
        } catch (e: Throwable) {
            models.value = models.value.copy(detector = "failed: ${e.message}", error = e.toString())
        }
    }

    /** Median wall time of a full detector step (preprocess, model, read-back) over a few synthetic frames. */
    private fun benchmark(d: LiveDetector): Float {
        val w = 640; val h = 480
        val px = IntArray(w * h) { (0xFF shl 24) or ((it * 2654435761L).toInt() and 0xFFFFFF) }
        repeat(2) { d.run(px, w, h) }
        val t = FloatArray(6) {
            val t0 = System.nanoTime(); d.run(px, w, h); (System.nanoTime() - t0) / 1e6f
        }
        d.resetSmoothing()
        return t.sorted()[t.size / 2]
    }

    private fun loadSpecies(files: BundleFiles, p: Plan) {
        try {
            identifier?.close(); identifier = null
            val meta = files.species
            val runtime = if (meta.accelerator == "GPU") p.speciesRuntime else Runtime.CPU
            val avail = DeviceProfile.availMb(getApplication())
            val need = meta.bytes / (1024 * 1024) + 400
            if (avail < need) message.value = "Low memory: $avail MB free, the species model needs about $need MB. Close other apps if identification fails."
            models.value = models.value.copy(species = "preparing ${meta.id} on ${runtime.name}…")
            val head = SpeciesHead(files.head, File(files.dir, files.head.file).readBytes())
            val cache = if (runtime == Runtime.CPU) File(xnnCache, "${files.manifest.version}-${meta.id}.xnn") else null
            prefs.pending = "species:load"
            val s = try {
                SpeciesIdentifier(files.dir, meta, head, files.catalog, runtime, p.speciesThreads, cache, environment())
            } finally {
                prefs.pending = null
            }
            identifier = s
            models.value = models.value.copy(
                species = "${meta.label} · ${s.accelerator}, ${p.speciesThreads} threads · loaded in ${s.loadMs} ms" + if (s.usedCache) " (weight cache)" else "",
                speciesReady = true)
            speciesLoaded.complete(true)
        } catch (e: Throwable) {
            val next = DevicePlanner.fallback(p, files.speciesModels.variants)
            models.value = models.value.copy(species = "failed: ${e.message}", error = e.toString())
            speciesLoaded.complete(false)
            if (next != null) {
                setPlan(next)
                if (next.speciesVariant == files.variant) speciesExecutor.execute { loadSpecies(files, next) }
                else { neededVariant.value = next.speciesVariant; message.value = "Species model failed (${e.message}). Next: ${next.reason}." }
            }
        }
    }

    @Synchronized private fun environment(): Environment = env ?: Environment.create().also { env = it }

    fun reloadModels() {
        (bundle.value as? BundleState.Ready)?.let { loadModels(it.files) }
    }

    fun rerunDetectorBenchmark() {
        prefs.detectorBenchRuntime = null
        prefs.detectorAccel = AccelChoice.AUTO
        reloadModels()
    }

    // ---------------------------------------------------------------------------------------- live
    private var px = IntArray(0)
    private var lastRun = 0L

    /** CameraX analyzer (on detectorExecutor): one RGBA frame -> upright pixels -> detector. */
    fun analyze(proxy: ImageProxy) {
        try {
            val d = detector ?: return
            if (screen.value != Screen.LIVE || busy.value != null) return       // CPU goes to the identification
            val now = SystemClock.elapsedRealtime()
            val fps = plan.value?.liveMaxFps ?: 8
            if (now - lastRun < 1000L / fps) return                               // frame-rate cap on slow phones
            lastRun = now
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
        if (busy.value != null) return
        busy.value = if (models.value.speciesReady) "Identifying…" else "Preparing the species model (first run takes longest)…"
        viewModelScope.launch {
            try {
                if (!speciesLoaded.await()) error("the species model could not be loaded: ${models.value.species}")
                busy.value = "Identifying…" + (estimate()?.let { " (about $it s on this phone)" } ?: "")
                val r = withContext(speciesDispatcher) {
                    val t0 = System.nanoTime()
                    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
                    BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, bounds)
                    // keep the short side >= 448 px: enough for the 224 px antialiased resize, little memory
                    var sample = 1
                    while (min(bounds.outWidth, bounds.outHeight) / (sample * 2) >= 448) sample *= 2
                    val raw = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size, BitmapFactory.Options().apply { inSampleSize = sample })
                        ?: error("could not decode the photo")
                    val bmp = upright(raw, rotation)
                    val pixels = IntArray(bmp.width * bmp.height)
                    bmp.getPixels(pixels, 0, bmp.width, 0, 0, bmp.width, bmp.height)
                    prefs.pending = "species:run"
                    val (top, t) = try { identifier!!.identify(pixels, bmp.width, bmp.height) } finally { prefs.pending = null }
                    lastIdMs = t.modelMs
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

    private var lastIdMs = 0f
    private fun estimate(): Int? = if (lastIdMs > 1500f) Math.round(lastIdMs / 1000f) else null

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
