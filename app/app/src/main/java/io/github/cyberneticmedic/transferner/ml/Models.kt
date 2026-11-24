package io.github.cyberneticmedic.transferner.ml

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import io.github.cyberneticmedic.transferner.core.NameAliases
import io.github.cyberneticmedic.transferner.core.TextIndex
import io.github.cyberneticmedic.transferner.core.Catalog
import io.github.cyberneticmedic.transferner.core.DetectorMeta
import io.github.cyberneticmedic.transferner.core.Detections
import io.github.cyberneticmedic.transferner.core.LiveResult
import io.github.cyberneticmedic.transferner.core.Preprocess
import io.github.cyberneticmedic.transferner.core.Runtime
import io.github.cyberneticmedic.transferner.core.SpeciesHead
import io.github.cyberneticmedic.transferner.core.SpeciesModelMeta
import io.github.cyberneticmedic.transferner.core.TensorInfo
import io.github.cyberneticmedic.transferner.core.BoxRules
import io.github.cyberneticmedic.transferner.core.BoxTracker
import io.github.cyberneticmedic.transferner.core.LiveWarning
import io.github.cyberneticmedic.transferner.core.TimeEma
import io.github.cyberneticmedic.transferner.core.Tuning
import io.github.cyberneticmedic.transferner.core.WarningIndex
import io.github.cyberneticmedic.transferner.core.topMass
import java.io.File

/** Detector runtime setting: AUTO benchmarks GPU against CPU on the phone at first launch (D-035). */
enum class AccelChoice(val label: String) { AUTO("Auto"), GPU("GPU"), CPU("CPU") }

/**
 * One LiteRT CompiledModel on one runtime. Creation and every run must happen on the same thread (GPU
 * contexts are thread-bound), so each model lives on its own single-thread executor in the ViewModel.
 * On the CPU, XNNPACK's weight cache ([cache]) keeps the packed weights in a file: the first load packs
 * them once, later loads map the file (faster, and file-backed rather than anonymous memory).
 */
class ModelRunner(
    path: String, private val signature: String, runtime: Runtime, threads: Int, cache: File?, env: Environment,
) : AutoCloseable {
    val model: CompiledModel
    val accelerator: String = runtime.name
    val loadMs: Long
    val usedCache: Boolean

    init {
        val t0 = System.nanoTime()
        fun options(withCache: Boolean) = CompiledModel.Options(if (runtime == Runtime.GPU) Accelerator.GPU else Accelerator.CPU).apply {
            cpuOptions = CompiledModel.CpuOptions(threads, null, if (withCache) cache?.path else null)
        }
        var cached = cache != null
        model = try {
            CompiledModel.create(path, options(cached), env)
        } catch (e: Exception) {
            if (!cached) throw e
            cache?.delete()                                  // a stale or unsupported cache: load without it
            cached = false
            CompiledModel.create(path, options(false), env)
        }
        usedCache = cached
        loadMs = (System.nanoTime() - t0) / 1_000_000
    }

    fun input(name: String): TensorBuffer = model.createInputBuffer(name, signature)
    fun output(name: String): TensorBuffer = model.createOutputBuffer(name, signature)
    fun run(inputs: Map<String, TensorBuffer>, outputs: Map<String, TensorBuffer>) = model.run(inputs, outputs, signature)
    val lastMs: Float get() = model.lastNativeInferenceDurationNanoseconds / 1e6f

    override fun close() = model.close()
}

private fun TensorBuffer.readAsFloat(t: TensorInfo): FloatArray =
    if (t.isInt8) Detections.dequant(readInt8(), t) else readFloat()

/**
 * The live detector's app graph (export/detector_app.py): whole-frame genus logits, objectness, boxes, and on
 * bundle 0.6+ the window logits. Per frame (D-044): the genus distribution is smoothed with a time-based EMA
 * (label_tau_s), boxes are organised (secondary threshold, containment) and tracked, and the plant-presence
 * gate (top-5 mass of the unfocused distribution) can hide boxes and the warning row on frames that do not
 * look like any known plant.
 */
class LiveDetector(dir: File, val meta: DetectorMeta, runtime: Runtime, threads: Int, env: Environment,
                   private val tuning: Tuning = Tuning()) : AutoCloseable {
    private val runner = ModelRunner(File(dir, meta.model).path, meta.signature, runtime, threads, null, env)
    private val inBuf = runner.input(meta.input.name)
    private val outs = meta.outputs
    private val outBufs = (listOf(outs.imageLogits, outs.objectness, outs.boxes) + listOfNotNull(outs.regionLogits))
        .associate { it.name to runner.output(it.name) }
    private val labelEma = TimeEma(tuning.labelTauS)        // the shown label (focused or not)
    private val warnEma = TimeEma(tuning.warnTauS)          // warnings and walk scoring: unfocused, faster
    private val tracker = BoxTracker()
    private var heldWarn: List<LiveWarning> = emptyList()
    private var warnAge = Int.MAX_VALUE
    val accelerator get() = runner.accelerator
    val loadMs get() = runner.loadMs
    val threads = threads
    var lastMs = 0f
        private set
    var lastPreMs = 0f
        private set
    var lastPostMs = 0f
        private set

    /** Live warning flags (D-038): per detector genus, 2 = has a cited serious toxic species, 1 = only mild, 0 = none. */
    @Volatile var genusWarn: IntArray? = null

    /** Focus chip (D-037): per-genus logit bias, or null for no focus. Set from the detector thread. */
    @Volatile var focusBias: FloatArray? = null

    /** One upright frame at [nowMs] (elapsed realtime). [boxThr] is the user's plant-score threshold (D-039). */
    fun run(argb: IntArray, w: Int, h: Int, nowMs: Long = System.nanoTime() / 1_000_000, boxThr: Float = 0.25f): LiveResult {
        val t0 = System.nanoTime()
        val x = Preprocess.detectorInput(argb, w, h, meta)
        lastPreMs = (System.nanoTime() - t0) / 1e6f
        if (meta.input.isInt8) inBuf.writeInt8(x) else error("detector input must be int8")
        runner.run(mapOf(meta.input.name to inBuf), outBufs)
        lastMs = runner.lastMs
        val t1 = System.nanoTime()
        val logits = outBufs.getValue(outs.imageLogits.name).readAsFloat(outs.imageLogits)
        // Warnings and walk alerts use the distribution WITHOUT the focus prior (D-038/D-037): a wrong focus chip
        // must never silence a toxic genus. The label uses the focused distribution.
        val frame = Detections.softmax(logits)
        val rawPs = warnEma.update(frame, nowMs)
        val bias = focusBias
        val ps = if (bias == null) labelEma.update(frame, nowMs) else {
            for (i in logits.indices) logits[i] += bias[i]
            labelEma.update(Detections.softmax(logits), nowMs)
        }
        // presence on this frame alone, as calibrated (eval/presence_gate.py): an average of frames that disagree
        // has a lower top-5 mass than any of them, which would hide real plants while panning
        val presence = topMass(frame)
        val obj = outBufs.getValue(outs.objectness.name).readAsFloat(outs.objectness)
        val reg = outBufs.getValue(outs.boxes.name).readAsFloat(outs.boxes)
        val cands = Detections.boxes(obj, Detections.decode(reg, meta), meta, maxBoxes = 6)
        val best = cands.firstOrNull()?.score ?: 0f
        val organised = if (presence >= tuning.presenceBoxes) BoxRules.organize(cands, boxThr, tuning, meta.size) else emptyList()
        val boxes = tracker.update(organised).take(3)
        val (genera, fam, famP) = Detections.genera(ps, meta)
        val now = genusWarn?.takeIf { presence >= tuning.presenceWarn }?.let { gw ->
            Detections.topK(rawPs, 8).filter { gw[it] > 0 && rawPs[it] >= WarningIndex.LIVE_MIN_P }
                .map { LiveWarning(meta.genera[it], rawPs[it], gw[it] == 2) }
        }.orEmpty()
        // hold the row for a few frames after it last fired, so it does not flicker on and off while walking
        if (now.isNotEmpty()) { heldWarn = now; warnAge = 0 } else warnAge++
        val warn = if (now.isNotEmpty()) now else if (warnAge <= tuning.warnHoldFrames) heldWarn else emptyList()
        val regions = outs.regionLogits?.let { outBufs.getValue(it.name).readAsFloat(it) }
        lastPostMs = (System.nanoTime() - t1) / 1e6f
        return LiveResult(genera, fam, famP, boxes, meta.size, best, warn, rawPs, presence, regions, nowMs)
    }

    fun resetSmoothing() { labelEma.reset(); warnEma.reset(); tracker.reset(); heldWarn = emptyList(); warnAge = Int.MAX_VALUE }

    override fun close() {
        inBuf.close(); outBufs.values.forEach { it.close() }; runner.close()
    }
}

/**
 * BioCLIP image tower + the regional head: one still -> the 4,411-species distribution (D-002 stage 2), merged
 * over duplicated names, plus the text score when the bundle ships the name embeddings (D-044).
 */
class SpeciesIdentifier(
    dir: File, val meta: SpeciesModelMeta, private val head: SpeciesHead, catalog: Catalog,
    runtime: Runtime, threads: Int, cache: File?, env: Environment, private val text: TextIndex? = null,
) : AutoCloseable {
    private val runner = ModelRunner(File(dir, meta.model).path, meta.signature, runtime, threads, cache, env)
    val usedCache get() = runner.usedCache
    private val inBuf = runner.input(meta.input.name)
    private val outBuf = runner.output(meta.output.name)
    private val aliases = NameAliases(catalog)
    val accelerator get() = runner.accelerator
    val loadMs get() = runner.loadMs

    data class Timing(val preMs: Float, val modelMs: Float, val headMs: Float)
    class Identification(val probs: FloatArray, val text: Float?, val timing: Timing)

    fun identify(argb: IntArray, w: Int, h: Int): Identification {
        val t0 = System.nanoTime()
        val x = Preprocess.speciesInput(argb, w, h, meta)
        val t1 = System.nanoTime()
        inBuf.writeFloat(x)
        runner.run(mapOf(meta.input.name to inBuf), mapOf(meta.output.name to outBuf))
        val e = outBuf.readFloat()
        val t2 = System.nanoTime()
        val p = aliases.merge(head.probs(e))
        val ts = text?.score(e)
        val t3 = System.nanoTime()
        return Identification(p, ts, Timing((t1 - t0) / 1e6f, (t2 - t1) / 1e6f, (t3 - t2) / 1e6f))
    }

    override fun close() {
        inBuf.close(); outBuf.close(); runner.close()
    }
}
