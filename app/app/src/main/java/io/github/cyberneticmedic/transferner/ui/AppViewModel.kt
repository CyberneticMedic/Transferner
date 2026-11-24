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
import io.github.cyberneticmedic.transferner.core.AlertAccumulator
import io.github.cyberneticmedic.transferner.core.CaptureAnalysis
import io.github.cyberneticmedic.transferner.core.CaptureRules
import io.github.cyberneticmedic.transferner.core.ForageScorer
import io.github.cyberneticmedic.transferner.core.HoldDetector
import io.github.cyberneticmedic.transferner.core.TextIndex
import io.github.cyberneticmedic.transferner.core.Tier
import io.github.cyberneticmedic.transferner.core.cautionCandidates
import io.github.cyberneticmedic.transferner.data.WalkRecorder
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
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.util.concurrent.Executors
import kotlin.math.max
import kotlin.math.min

enum class Screen { SETUP, LIVE, RESULT, SPECIES, LOG, SETTINGS, FORAGE }

const val MIN_BOX_THRESHOLD = 0.05f
const val MAX_BOX_THRESHOLD = 0.9f

/** In-app update state (D-042). */
sealed interface UpdateState {
    data object Idle : UpdateState
    data object Checking : UpdateState
    data class UpToDate(val bundle: String?) : UpdateState
    data class Available(val release: io.github.cyberneticmedic.transferner.core.AppRelease) : UpdateState
    data class Downloading(val release: io.github.cyberneticmedic.transferner.core.AppRelease, val done: Long, val total: Long) : UpdateState
    data class Ready(val release: io.github.cyberneticmedic.transferner.core.AppRelease, val apk: File) : UpdateState
    data class Failed(val message: String) : UpdateState
}

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

data class LiveFrame(val result: LiveResult, val frameW: Int, val frameH: Int, val modelMs: Float, val preMs: Float,
                     val forage: ForageScorer.Score? = null, val forageSmoothed: Float = 0f)

data class IdResult(
    val photo: Bitmap,
    val jpeg: ByteArray,
    val top: List<Candidate>,
    val modelMs: Float,
    val totalMs: Float,
    val fromLog: Boolean = false,
    val saved: Boolean = false,
    /** Levels, caution banner and second-photo trigger (D-044); null for entries reopened from the log. */
    val analysis: CaptureAnalysis? = null,
    val text: Float? = null,
    val secondPhoto: Bitmap? = null,
    val secondJpeg: ByteArray? = null,
    /** The live model's last frames before this capture, as a JSON array (camera captures only). */
    val liveRing: String = "[]",
    /** Reopened from the log: the caution flag saved with it (its banner may have come from outside the top 5). */
    val storedCaution: Boolean = false,
)

/** A walk alert (D-044): the live model's forage score crossed its threshold and BioCLIP was asked to confirm. */
data class WalkAlert(
    val timeMs: Long,
    val clock: String,
    val genus: String,
    val label: String,
    val score: Float,
    val confirmed: Boolean,
    val bioclip: String,                         // BioCLIP's top species and its probability, or "" if not run
    val genusP: Float,                           // BioCLIP probability summed over the alerted genus
    val window: List<Float>?,                    // the region window ([x0, y0, x1, y1] fractions) or null
    val frame: String?,                          // saved walk frame, if recording
    val note: String = "",                       // forage.json note: which part, and its cautions
)

/**
 * App state, and the fallback chain for phones from a Pixel 9 Pro down to a Moto G Power 2025 (D-035):
 * plan the species-model variant from the device profile, benchmark the detector's runtimes on the phone,
 * guard every native load and run with a crash flag and step the plan down after a crash, pause the live
 * detector during an identification, and cap the live frame rate on slow phones.
 */
class AppViewModel(app: Application) : AndroidViewModel(app) {
    val prefs = Prefs(app)
    val updater = io.github.cyberneticmedic.transferner.data.AppUpdater(app)
    val appUpdate = MutableStateFlow<UpdateState>(UpdateState.Idle)
    /** A newer model bundle on the server than the one installed (version), or null. */
    val bundleUpdate = MutableStateFlow<String?>(null)
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
    /** Plant-score sensitivity (D-039): minimum objectness for a live box; lower = more boxes. */
    val boxThreshold = MutableStateFlow(prefs.boxThreshold)
    fun setBoxThreshold(v: Float) {
        val t = (Math.round(v * 100) / 100f).coerceIn(MIN_BOX_THRESHOLD, MAX_BOX_THRESHOLD)
        boxThreshold.value = t
        prefs.boxThreshold = t
    }

    /** Live focus chip (D-037): a growth-form id from focus.json, or null for no focus. */
    val focus = MutableStateFlow<String?>(null)

    // ------------------------------------------------------------------------------- walk mode (D-044)
    val walk = MutableStateFlow(false)
    val alert = MutableStateFlow<WalkAlert?>(null)          // the latest confirmed alert, shown until dismissed
    val alertsToday = MutableStateFlow<List<WalkAlert>>(emptyList())
    val toxicNow = MutableStateFlow(false)
    val recorder = WalkRecorder.of(app)
    @Volatile private var scorer: ForageScorer? = null
    @Volatile private var accumulator: AlertAccumulator? = null
    @Volatile private var toxicHold: HoldDetector? = null
    @Volatile private var verifying = false
    private var lastSavedFrame = 0L
    @Volatile private var lastUpright: Bitmap? = null
    /** Capture rules (levels, banner, second photo) for the installed bundle. */
    @Volatile var rules: CaptureRules? = null
        private set
    /** Set while the user takes a second photo of the same plant for [secondFor]. */
    val secondFor = MutableStateFlow<IdResult?>(null)

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
            checkUpdates(silent = true)
        }
    }

    /** App (app.json) and model bundle (latest.json) updates on the home server. Silent: no message when offline. */
    fun checkUpdates(silent: Boolean = false) {
        if (appUpdate.value is UpdateState.Checking || appUpdate.value is UpdateState.Downloading) return
        appUpdate.value = UpdateState.Checking
        viewModelScope.launch {
            try {
                val (rel, latest) = withContext(Dispatchers.IO) {
                    updater.check(prefs.server) to runCatching { repo.remoteLatest(prefs.server) }.getOrNull()
                }
                val installed = repo.installed()?.first
                val remote = latest?.version
                // a bundle this app cannot run is not offered; the app update is (D-045)
                bundleUpdate.value = remote?.takeIf { installed != null && it != installed && latest.minApp <= updater.currentVersionCode }
                appUpdate.value = if (rel != null) UpdateState.Available(rel) else UpdateState.UpToDate(remote)
            } catch (e: Exception) {
                appUpdate.value = if (silent) UpdateState.Idle else UpdateState.Failed(e.message ?: e.toString())
            }
        }
    }

    fun downloadAppUpdate() {
        val rel = (appUpdate.value as? UpdateState.Available)?.release
            ?: (appUpdate.value as? UpdateState.Failed)?.let { return checkUpdates() } ?: return
        appUpdate.value = UpdateState.Downloading(rel, 0, rel.bytes)
        viewModelScope.launch {
            try {
                val apk = withContext(Dispatchers.IO) {
                    updater.download(prefs.server, rel) { d, t -> appUpdate.value = UpdateState.Downloading(rel, d, t) }
                }
                appUpdate.value = UpdateState.Ready(rel, apk)
            } catch (e: Exception) {
                appUpdate.value = UpdateState.Failed("App update download failed: ${e.message}")
            }
        }
    }

    /** The intent to start: the system installer, or first the one-time "allow installs from this app" screen. */
    fun installIntent(): android.content.Intent? {
        val ready = appUpdate.value as? UpdateState.Ready ?: return null
        return if (updater.canInstall()) updater.installIntent(ready.apk) else updater.allowInstallsIntent()
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
        val legacy = prefs.pending
        val det = prefs.pendingDetector
        val spc = prefs.pendingSpecies
        prefs.pending = null; prefs.pendingDetector = null; prefs.pendingSpecies = null
        // the species model is the heavy one: if both flags were set, it is the likelier cause
        val pending = spc?.let { "species:$it" } ?: det?.let { "detector:$it" } ?: legacy ?: return
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
                    repo.download(server, { models -> planFor(models).speciesVariant }, updater.currentVersionCode) {
                        bundle.value = BundleState.Downloading(it)
                    }
                }
                val files = withContext(Dispatchers.IO) { repo.load() } ?: error("bundle incomplete after download")
                bundle.value = BundleState.Ready(files)
                neededVariant.value = null
                bundleUpdate.value = null
                history.clear()
                screen.value = Screen.LIVE
                loadModels(files)
            } catch (e: Exception) {
                bundle.value = BundleState.Failed(e.message ?: e.toString())
            }
        }
    }

    // ------------------------------------------------------------------------------------- models
    /** Cited toxic species per genus (D-038), for relatives at capture and the live warning row. */
    @Volatile var warnIndex: io.github.cyberneticmedic.transferner.core.WarningIndex? = null
        private set

    private fun genusWarn(files: BundleFiles): IntArray {
        val idx = warnIndex ?: io.github.cyberneticmedic.transferner.core.WarningIndex(files.catalog).also { warnIndex = it }
        return IntArray(files.detector.genera.size) { g ->
            idx.warnLevel(files.detector.genera[g], files.detector.families[files.detector.familyOf[g]])
        }
    }

    /** Forage tiers from the bundle and the user's overrides; rebuilt when the user edits them. */
    fun rebuildForage(files: BundleFiles? = (bundle.value as? BundleState.Ready)?.files) {
        val f = files ?: return
        scorer = f.forage?.let { ForageScorer(it, f.detector.genera.size, prefs.forageTiers, f.tuning.alertWeightFloor) }
        accumulator = AlertAccumulator(f.tuning)
        toxicHold = HoldDetector(f.tuning.toxicHaptic, f.tuning.toxicHoldS, f.tuning.alertTauS, 120f)
    }

    fun setForageTier(genus: String, tier: Tier) {
        prefs.forageTiers = prefs.forageTiers + (genus to tier)
        rebuildForage()
    }

    fun resetForageTiers() { prefs.forageTiers = emptyMap(); rebuildForage() }

    fun loadModels(files: BundleFiles) {
        val p = plan.value ?: DevicePlanner.plan(device.value ?: DeviceProfile.read(getApplication(), null), files.speciesModels.variants).also { setPlan(it) }
        // capture rules belong to this bundle and must not wait for (or depend on) the live detector
        val idx = io.github.cyberneticmedic.transferner.core.WarningIndex(files.catalog)
        warnIndex = idx
        rules = CaptureRules(files.catalog, idx, files.tuning)
        rebuildForage(files)
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
            // a new model bundle gets a fresh benchmark (448 px costs about twice the 320 px graph)
            if (prefs.benchBundle != files.manifest.version) { prefs.detectorBenchRuntime = null; prefs.detectorThreads = 0 }
            var bench = prefs.detectorBenchRuntime
            if (choice == AccelChoice.AUTO && bench == null) {
                val times = mutableMapOf<String, Float>()
                val errors = mutableListOf<String>()
                val cores = device.value?.cores ?: 4
                val trials = listOf(Runtime.CPU to 2) + (if (cores >= 6) listOf(Runtime.CPU to 4) else emptyList()) +
                    (if (gpuOk) listOf(Runtime.GPU to 2) else emptyList())
                for ((r, th) in trials) {
                    models.update { it.copy(detector = "measuring ${r.name}${if (r == Runtime.CPU) " ×$th" else ""}…") }
                    prefs.pendingDetector = r.name
                    try {
                        LiveDetector(files.dir, files.detector, r, th, environment(), files.tuning).use { d -> times["${r.name}:$th"] = benchmark(d) }
                    } catch (e: Exception) {
                        errors += "${r.name}: ${e.message ?: e.javaClass.simpleName}"
                    } finally {
                        prefs.pendingDetector = null
                    }
                }
                val bestKey = times.minByOrNull { it.value }?.key ?: "CPU:2"
                bench = Runtime.valueOf(bestKey.substringBefore(':'))
                prefs.detectorBenchRuntime = bench
                prefs.detectorThreads = if (bench == Runtime.CPU) bestKey.substringAfter(':').toInt() else 2
                prefs.detectorBench = times.entries.joinToString(", ") { (k, v) ->
                    (if (k.startsWith("CPU")) "CPU ×${k.substringAfter(':')}" else k.substringBefore(':')) + " %.0f ms".format(v) }
                prefs.benchErrors = errors.joinToString("; ")
                prefs.benchBundle = files.manifest.version
            }
            val threads = prefs.detectorThreads.takeIf { it > 0 } ?: p.detectorThreads
            val runtime = when (choice) {
                AccelChoice.GPU -> Runtime.GPU
                AccelChoice.CPU -> Runtime.CPU
                AccelChoice.AUTO -> bench ?: Runtime.CPU
            }
            prefs.pendingDetector = runtime.name
            val d = try {
                LiveDetector(files.dir, files.detector, runtime, threads, environment(), files.tuning)
            } catch (e: Exception) {
                if (runtime == Runtime.CPU) throw e
                val err = "GPU load: ${e.message}"
                if (err !in prefs.benchErrors) prefs.benchErrors = listOf(prefs.benchErrors, err).filter { it.isNotBlank() }.joinToString("; ")
                if (choice == AccelChoice.AUTO) prefs.detectorBenchRuntime = Runtime.CPU   // do not retry the GPU every launch
                LiveDetector(files.dir, files.detector, Runtime.CPU, threads, environment(), files.tuning)   // GPU refused: CPU
            } finally {
                prefs.pendingDetector = null
            }
            focus.value?.let { id -> d.focusBias = files.focus?.bias(id, files.detector.genera.size) }
            d.genusWarn = genusWarn(files)
            detector = d
            models.update { it.copy(detector = "${d.accelerator}${if (d.accelerator == "CPU") " ×${d.threads}" else ""} · " +
                "${files.detector.size} px · loaded in ${d.loadMs} ms" +
                (if (prefs.detectorBench.isNotBlank()) " · measured ${prefs.detectorBench}" else ""), detectorReady = true) }
        } catch (e: Throwable) {
            models.update { it.copy(detector = "failed: ${e.message}", error = e.toString()) }
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
            models.update { it.copy(species = "preparing ${meta.id} on ${runtime.name}…") }
            val head = SpeciesHead(files.head, File(files.dir, files.head.file).readBytes())
            val cache = if (runtime == Runtime.CPU) File(xnnCache, "${files.manifest.version}-${meta.id}.xnn") else null
            prefs.pendingSpecies = "load"
            val text = files.head.textFile?.let { f -> File(files.dir, f).takeIf { it.exists() } }?.let {
                TextIndex(it.readBytes(), files.catalog.species.size, files.head.textDim)
            }
            val s = try {
                SpeciesIdentifier(files.dir, meta, head, files.catalog, runtime, p.speciesThreads, cache, environment(), text)
            } finally {
                prefs.pendingSpecies = null
            }
            identifier = s
            models.update { it.copy(
                species = "${meta.label} · ${s.accelerator}, ${p.speciesThreads} threads · loaded in ${s.loadMs} ms" + if (s.usedCache) " (weight cache)" else "",
                speciesReady = true) }
            speciesLoaded.complete(true)
        } catch (e: Throwable) {
            val next = DevicePlanner.fallback(p, files.speciesModels.variants)
            models.update { it.copy(species = "failed: ${e.message}", error = e.toString()) }
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

    fun setFocus(id: String?) {
        focus.value = id
        val files = (bundle.value as? BundleState.Ready)?.files
        val bias = if (id == null) null else files?.focus?.bias(id, files.detector.genera.size)
        detectorExecutor.execute { detector?.let { it.focusBias = bias; it.resetSmoothing() } }
    }

    fun rerunDetectorBenchmark() {
        prefs.detectorBenchRuntime = null
        prefs.detectorAccel = AccelChoice.AUTO
        reloadModels()
    }

    // ---------------------------------------------------------------------------------------- live
    private var px = IntArray(0)
    private var lastRun = 0L
    private val liveMs = ArrayDeque<Float>()
    private val liveBest = ArrayDeque<Float>()
    /** Per-stage times of recent frames (D-044 device report): camera bitmap + rotation, preprocess, model, post. */
    private val stageMs = Array(4) { ArrayDeque<Float>() }
    private val frameTimes = ArrayDeque<Long>()
    private val perMinute = ArrayDeque<Pair<Long, Float>>()
    private val ring = ArrayDeque<String>()
    private val startBattery = batteryPct()
    private val startMs = SystemClock.elapsedRealtime()
    private var alertsConfirmed = 0
    private var alertsUnconfirmed = 0

    private fun push(q: ArrayDeque<Float>, v: Float, n: Int = 300) = synchronized(q) { q.addLast(v); if (q.size > n) q.removeFirst() }
    private fun q(v: List<Float>, f: Double) = if (v.isEmpty()) null else v[((v.size - 1) * f).toInt()]

    private fun batteryPct(): Int = runCatching {
        (getApplication<Application>().getSystemService(android.content.Context.BATTERY_SERVICE) as android.os.BatteryManager)
            .getIntProperty(android.os.BatteryManager.BATTERY_PROPERTY_CAPACITY)
    }.getOrDefault(-1)

    private fun thermal(): Int = runCatching {
        (getApplication<Application>().getSystemService(android.content.Context.POWER_SERVICE) as android.os.PowerManager).currentThermalStatus
    }.getOrDefault(-1)

    /** This phone's profile, plan, runtimes and recent timings, uploaded with the log (D-036/D-038/D-044). */
    private fun deviceReport(): String {
        val files = (bundle.value as? BundleState.Ready)?.files
        val ms = synchronized(liveMs) { liveMs.sorted() }
        val bs = synchronized(liveBest) { liveBest.sorted() }
        val now = SystemClock.elapsedRealtime()
        val fps = synchronized(frameTimes) { frameTimes.count { now - it < 60_000 } / 60f }
        val o = kotlinx.serialization.json.buildJsonObject {
            fun put(k: String, v: Any?) {
                when (v) {
                    null -> {}
                    is Number -> put(k, kotlinx.serialization.json.JsonPrimitive(v))
                    is Boolean -> put(k, kotlinx.serialization.json.JsonPrimitive(v))
                    else -> put(k, kotlinx.serialization.json.JsonPrimitive(v.toString()))
                }
            }
            put("app", getApplication<Application>().packageManager.getPackageInfo(getApplication<Application>().packageName, 0).versionName)
            put("device", device.value?.label ?: "")
            put("gpu_available", device.value?.gpuAvailable)
            put("plan", plan.value?.toString() ?: "")
            put("bundle", "${files?.manifest?.version} ${files?.variant}")
            put("detector", models.value.detector)
            put("species", models.value.species)
            put("detector_bench", prefs.detectorBench)
            put("detector_bench_errors", prefs.benchErrors)
            put("detector_threads", prefs.detectorThreads)
            put("live_frames", ms.size)
            put("live_ms_p50", q(ms, 0.5)); put("live_ms_p90", q(ms, 0.9))
            listOf("camera", "pre", "model", "post").forEachIndexed { i, k ->
                val v = synchronized(stageMs[i]) { stageMs[i].sorted() }
                put("stage_${k}_ms_p50", q(v, 0.5)); put("stage_${k}_ms_p90", q(v, 0.9))
            }
            put("fps_last_minute", fps)
            put("live_ms_p50_per_minute", synchronized(perMinute) { perMinute.joinToString(" ") { "%.0f".format(it.second) } })
            put("live_best_objectness_p50", q(bs, 0.5)); put("live_best_objectness_p90", q(bs, 0.9))
            put("last_id_ms", lastIdMs)
            put("thermal_status", thermal())
            put("battery_pct", batteryPct())
            val hours = (now - startMs) / 3_600_000f
            if (hours > 0.05f && startBattery >= 0) put("battery_pct_per_hour", (startBattery - batteryPct()) / hours)
            put("walk_alerts_confirmed", alertsConfirmed); put("walk_alerts_unconfirmed", alertsUnconfirmed)
        }
        return o.toString()
    }

    /** CameraX analyzer (on detectorExecutor): one RGBA frame -> upright pixels -> detector -> walk logic. */
    fun analyze(proxy: ImageProxy) {
        try {
            val d = detector ?: return
            if (screen.value != Screen.LIVE || busy.value != null) return       // CPU goes to the identification
            val now = SystemClock.elapsedRealtime()
            val files = (bundle.value as? BundleState.Ready)?.files
            val tuning = files?.tuning ?: io.github.cyberneticmedic.transferner.core.Tuning()
            var fps = if (walk.value) tuning.walkFps else plan.value?.liveMaxFps ?: 8
            if (thermal() >= android.os.PowerManager.THERMAL_STATUS_SEVERE) fps = minOf(fps, 2)    // hot phone: slow down
            if (now - lastRun < 1000L / fps) return                               // frame-rate cap
            lastRun = now
            val t0 = System.nanoTime()
            val up = upright(proxy.toBitmap(), proxy.imageInfo.rotationDegrees)
            val n = up.width * up.height
            if (px.size != n) px = IntArray(n)
            up.getPixels(px, 0, up.width, 0, 0, up.width, up.height)
            val camMs = (System.nanoTime() - t0) / 1e6f
            val r = d.run(px, up.width, up.height, now, boxThreshold.value)
            val sc = scorer
            val fs = r.rawPs?.let { ps -> sc?.score(ps) }
            val acc = accumulator
            var startAlert = -1
            // alert candidates only when BioCLIP can check them now (0.6 review: unchecked ones were locked out)
            if (walk.value && fs != null && acc != null && !verifying && identifier != null && speciesLoaded.isCompleted)
                startAlert = acc.update(fs.alert, fs.alertGenus, now)
            live.value = LiveFrame(r, up.width, up.height, d.lastMs, d.lastPreMs, fs, acc?.smoothed ?: 0f)
            lastUpright = up
            push(liveMs, d.lastMs + d.lastPreMs, 200); push(liveBest, r.bestScore, 200)
            push(stageMs[0], camMs); push(stageMs[1], d.lastPreMs); push(stageMs[2], d.lastMs); push(stageMs[3], d.lastPostMs)
            synchronized(frameTimes) { frameTimes.addLast(now); while (frameTimes.isNotEmpty() && now - frameTimes.first() > 60_000) frameTimes.removeFirst() }
            synchronized(perMinute) {
                val minute = now / 60_000
                if (perMinute.isEmpty() || perMinute.last().first != minute) {
                    val ms = synchronized(liveMs) { liveMs.sorted() }
                    q(ms, 0.5)?.let { perMinute.addLast(minute to it) }
                    if (perMinute.size > 60) perMinute.removeFirst()
                }
            }
            val line = frameJson(r, fs, acc?.smoothed ?: 0f, d.lastMs + d.lastPreMs + camMs)
            synchronized(ring) { ring.addLast(line); if (ring.size > 16) ring.removeFirst() }
            if (walk.value) {
                if (prefs.recordWalks) {
                    recorder.line(line)
                    if (now - lastSavedFrame >= 1000) { lastSavedFrame = now; recorder.frame(jpeg(up, 70), now) }
                }
                // serious-toxic haptic: smoothed mass of genera holding a serious cited toxic species (D-038 genera)
                val gw = d.genusWarn
                val ps = r.rawPs
                if (gw != null && ps != null && r.presence >= tuning.presenceWarn) {
                    var m = 0f
                    for (g in ps.indices) if (gw[g] == 2) m += ps[g]
                    val hit = toxicHold?.update(m, now) == true
                    toxicNow.value = m >= tuning.toxicHaptic
                    if (hit && prefs.walkHaptics) buzz(longArrayOf(0, 450))
                }
                if (startAlert >= 0 && files != null && fs != null) onAlert(startAlert, fs.alert, r, files, now)
            }
        } catch (e: Throwable) {
            message.value = "Live detector: ${e.message}"
        } finally {
            proxy.close()
        }
    }

    private fun frameJson(r: LiveResult, fs: ForageScorer.Score?, smoothed: Float, totalMs: Float): String {
        val names = (bundle.value as? BundleState.Ready)?.files?.detector?.genera
        val top = r.rawPs?.let { ps -> io.github.cyberneticmedic.transferner.core.Detections.topK(ps, 3).joinToString(",") {
            "[\"${names?.get(it) ?: it}\",%.3f]".format(java.util.Locale.US, ps[it]) } } ?: ""
        val warn = r.warnings.joinToString(",") { "\"${it.genus}\"" }
        val boxes = r.boxes.joinToString(",") { "[%.0f,%.0f,%.0f,%.0f,%.2f]".format(java.util.Locale.US, it.x0, it.y0, it.x1, it.y1, it.score) }
        return ("{\"t\":${System.currentTimeMillis()},\"ms\":%.0f,\"presence\":%.3f,\"top\":[$top],\"forage\":%.3f,\"forage_smoothed\":%.3f," +
            "\"show\":%.3f,\"warn\":[$warn],\"best\":%.2f,\"boxes\":[$boxes]}")
            .format(java.util.Locale.US, totalMs, r.presence, fs?.alert ?: 0f, smoothed, fs?.show ?: 0f, r.bestScore)
    }

    private fun jpeg(b: Bitmap, quality: Int): ByteArray = ByteArrayOutputStream().also { b.compress(Bitmap.CompressFormat.JPEG, quality, it) }.toByteArray()

    /** A forage alert started: confirm it with BioCLIP on this frame before buzzing (audit: precision 0.39 -> 0.94). */
    private fun onAlert(g: Int, score: Float, r: LiveResult, files: BundleFiles, now: Long) {
        if (verifying) { accumulator?.released(g); return }
        val up = lastUpright ?: run { accumulator?.released(g); return }
        val pixels = px.copyOf()
        val w = up.width; val h = up.height
        val genus = files.detector.genera[g]
        val info = scorer?.byIndex?.get(g)
        val window = r.regionLogits?.let { rl -> files.detector.regions?.let { reg ->
            val nG = files.detector.genera.size
            val best = (0 until rl.size / nG).maxByOrNull { wi ->       // the window where this genus scores highest
                io.github.cyberneticmedic.transferner.core.Detections.softmax(rl.copyOfRange(wi * nG, (wi + 1) * nG))[g]
            }
            best?.let { reg.windows.getOrNull(it) }
        } }
        verifying = true
        val frameName = if (prefs.recordWalks) recorder.frame(jpeg(up, 85), now, "-alert") else null
        viewModelScope.launch {
            try {
                val id = if (speciesLoaded.isCompleted && identifier != null) withContext(speciesDispatcher) {
                    prefs.pendingSpecies = "run"
                    try { identifier?.identify(pixels, w, h) } finally { prefs.pendingSpecies = null }
                } else null
                var gp = 0f
                var top = ""
                var confirmed = false
                if (id != null) {
                    // confirm on the genus' cited edible or conditional species (D-045), not the whole genus: a
                    // Bradford pear must not buzz as "pear"
                    val ru = rules
                    val edible = { i: Int -> ru?.summary(i)?.let { it.edibleParts.isNotEmpty() || it.conditionalParts.isNotEmpty() } ?: true }
                    for (s in files.catalog.species) if (s.genus == genus && edible(s.i)) gp += id.probs[s.i]
                    val i1 = io.github.cyberneticmedic.transferner.core.Detections.topK(id.probs, 1)[0]
                    val s1 = files.catalog.species[i1]
                    top = "${s1.name} ${pct(id.probs[i1])}"
                    confirmed = gp >= files.tuning.verifyGenusP || (s1.genus == genus && edible(i1))
                }
                val clock = java.text.SimpleDateFormat("HH:mm", java.util.Locale.US).format(java.util.Date())
                val a = WalkAlert(now, clock, genus, info?.label?.ifBlank { genus } ?: genus, score, confirmed, top, gp, window, frameName, info?.note.orEmpty())
                when {
                    id == null -> accumulator?.released(g)
                    confirmed -> accumulator?.confirmed(g, now)
                    else -> accumulator?.rejected(g, now)
                }
                if (prefs.recordWalks) recorder.line(("{\"t\":${System.currentTimeMillis()},\"event\":\"alert\",\"genus\":\"$genus\",\"score\":%.3f," +
                    "\"confirmed\":$confirmed,\"bioclip\":\"%s\",\"genus_p\":%.3f,\"frame\":\"%s\"}")
                    .format(java.util.Locale.US, score, top.replace("\"", ""), gp, frameName ?: ""))
                if (confirmed) {
                    alertsConfirmed++
                    alert.value = a
                    alertsToday.value = alertsToday.value + a
                    if (prefs.walkHaptics) buzz(longArrayOf(0, 70, 90, 70))
                } else alertsUnconfirmed++
            } catch (e: Throwable) {
                message.value = "Walk alert check failed: ${e.message}"
            } finally {
                verifying = false
            }
        }
    }

    private fun buzz(pattern: LongArray) {
        val ctx = getApplication<Application>()
        val v = if (android.os.Build.VERSION.SDK_INT >= 31)
            (ctx.getSystemService(android.content.Context.VIBRATOR_MANAGER_SERVICE) as android.os.VibratorManager).defaultVibrator
        else @Suppress("DEPRECATION") (ctx.getSystemService(android.content.Context.VIBRATOR_SERVICE) as android.os.Vibrator)
        runCatching { v.vibrate(android.os.VibrationEffect.createWaveform(pattern, -1)) }
    }

    fun setWalk(on: Boolean) {
        if (walk.value == on) return
        walk.value = on
        accumulator?.reset(); toxicHold?.reset()
        toxicNow.value = false
        alert.value = null
        val record = on && prefs.recordWalks
        detectorExecutor.execute {
            runCatching { if (record) recorder.start() else recorder.stop() }
                .onFailure { message.value = "Walk recording is off: ${it.message}" }
            detector?.resetSmoothing()
        }
    }

    fun dismissAlert() { alert.value = null }

    /** Walk mode: the user passed an edible the app did not flag. Keeps the current frame for training. */
    fun markMissed() {
        val up = lastUpright ?: return
        val now = SystemClock.elapsedRealtime()
        detectorExecutor.execute {
            val name = runCatching {
                if (recorder.session == null) recorder.start()
                recorder.frame(jpeg(up, 90), now, "-missed")
            }.getOrNull()
            recorder.line("{\"t\":${System.currentTimeMillis()},\"event\":\"missed\",\"frame\":\"${name ?: ""}\"}")
            message.value = if (name != null) "Marked as a missed plant; the frame is kept for training." else "Marked (storage cap reached, no frame kept)."
        }
    }

    private fun upright(b: Bitmap, deg: Int): Bitmap =
        if (deg == 0) b else Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postRotate(deg.toFloat()) }, false)

    // ------------------------------------------------------------------------------------ identify
    /** Start a second photo of the same plant (a different part); the next capture is fused with [r]. */
    fun startSecondPhoto(r: IdResult) { secondFor.value = r; go(Screen.LIVE) }
    fun cancelSecondPhoto() { secondFor.value = null }

    /**
     * A JPEG from the camera (with its rotation) or the photo picker (EXIF rotation) -> species distribution,
     * levels and banner (D-044). While a second photo is pending, the two are fused (mean log-probability) and
     * the caution banner is the union of the fused and per-photo banners (false positives over negatives).
     */
    fun identify(jpeg: ByteArray, rotation: Int, fromCamera: Boolean = false) {
        if (busy.value != null) return
        busy.value = if (models.value.speciesReady) "Identifying…" else "Preparing the species model (first run takes longest)…"
        val ringNow = if (fromCamera) synchronized(ring) { ring.joinToString(",", "[", "]") } else "[]"
        viewModelScope.launch {
            try {
                if (!speciesLoaded.await()) error("the species model could not be loaded: ${models.value.species}")
                busy.value = "Identifying…" + (estimate()?.let { " (about $it s on this phone)" } ?: "")
                val first = secondFor.value
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
                    prefs.pendingSpecies = "run"
                    val id = try { identifier!!.identify(pixels, bmp.width, bmp.height) } finally { prefs.pendingSpecies = null }
                    lastIdMs = id.timing.modelMs
                    val shown = scaled(bmp, 1080)
                    val out = ByteArrayOutputStream()
                    shown.compress(Bitmap.CompressFormat.JPEG, 88, out)
                    val ru = rules ?: error("capture rules not ready")
                    val a1 = ru.analyze(id.probs, id.text)
                    if (first?.analysis != null) {
                        val fused = CaptureRules.fuse(first.analysis.probs, id.probs)
                        val text = listOfNotNull(first.text, id.text).maxOrNull()
                        var a = ru.analyze(fused, text, photos = 2)
                        a = a.copy(caution = io.github.cyberneticmedic.transferner.core.Caution.union(listOf(a.caution, first.analysis.caution, a1.caution)))
                        first.copy(top = a.top, modelMs = first.modelMs + id.timing.modelMs, totalMs = first.totalMs + (System.nanoTime() - t0) / 1e6f,
                            analysis = a, text = text, secondPhoto = shown, secondJpeg = out.toByteArray(), saved = false)
                    } else {
                        IdResult(shown, out.toByteArray(), a1.top, id.timing.modelMs, (System.nanoTime() - t0) / 1e6f,
                            analysis = a1, text = id.text, liveRing = ringNow)
                    }
                }
                secondFor.value = null
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

    fun saveResult(note: String = "") {
        val r = result.value ?: return
        val files = (bundle.value as? BundleState.Ready)?.files ?: return
        viewModelScope.launch(Dispatchers.IO) {
            val an = r.analysis
            val caution = an?.caution?.show ?: cautionCandidates(r.top, warnIndex).isNotEmpty()
            log.add(r.jpeg, r.top, caution, files.manifest.version, files.species.variant, files.detector.model, note,
                idMs = r.modelMs, text = r.text, level = an?.verdict?.level?.name?.lowercase().orEmpty(), second = r.secondJpeg) { photo ->
                val v = an?.verdict
                val json = kotlinx.serialization.json.Json
                kotlinx.serialization.json.buildJsonObject {
                    fun s(k: String, x: String) = put(k, kotlinx.serialization.json.JsonPrimitive(x))
                    fun n(k: String, x: Number?) = put(k, kotlinx.serialization.json.JsonPrimitive(x))
                    fun b(k: String, x: Boolean) = put(k, kotlinx.serialization.json.JsonPrimitive(x))
                    s("photo", photo); s("bundle", files.manifest.version); s("species_model", files.species.variant)
                    s("detector", files.detector.model); n("id_ms", r.modelMs); n("total_ms", r.totalMs); n("photos", an?.photos ?: 1)
                    put("top", kotlinx.serialization.json.JsonArray(r.top.map { c ->
                        kotlinx.serialization.json.JsonArray(listOf(kotlinx.serialization.json.JsonPrimitive(c.species.name), kotlinx.serialization.json.JsonPrimitive(c.p))) }))
                    s("level", v?.level?.name?.lowercase().orEmpty()); s("genus", v?.genus.orEmpty()); n("genus_p", v?.genusP)
                    n("text_score", r.text); b("maybe_outside", v?.maybeOutside ?: false)
                    b("caution", caution); b("caution_strong", an?.caution?.strong ?: false); n("caution_sum", an?.caution?.sum)
                    b("second_photo_asked", an?.secondPhoto ?: false); s("note", note)
                    put("live", runCatching { json.parseToJsonElement(r.liveRing) }.getOrElse { kotlinx.serialization.json.JsonArray(emptyList()) })
                }.toString()
            }
            result.value = r.copy(saved = true)
            message.value = "Saved to the log."
        }
    }

    /** Upload the log (photos + CSV) to the home server so the field captures can be analysed (D-036). */
    fun uploadLog() {
        if (busy.value != null) return
        busy.value = "Sending the log…"
        viewModelScope.launch {
            try {
                val report = deviceReport()
                val n = withContext(Dispatchers.IO) {
                    log.upload(prefs.server, deviceId(), report, recorder.uploadable()) { i, total -> busy.value = "Sending files $i / $total…" }
                }
                message.value = "Sent $n new files and the log to ${prefs.server}"
            } catch (e: Exception) {
                message.value = "Could not send the log: ${e.message}"
            } finally {
                busy.value = null
            }
        }
    }

    private fun deviceId(): String {
        val model = android.os.Build.MODEL.lowercase().replace(Regex("[^a-z0-9]+"), "-").trim('-').ifBlank { "phone" }
        return "$model-${prefs.installId}"
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
                IdResult(bmp, bytes, e.top, 0f, 0f, fromLog = true, saved = true, storedCaution = e.caution)
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
        runCatching { recorder.stop() }
        detectorExecutor.execute { detector?.close() }
        speciesExecutor.execute { identifier?.close() }
        detectorExecutor.shutdown(); speciesExecutor.shutdown()
    }
}
