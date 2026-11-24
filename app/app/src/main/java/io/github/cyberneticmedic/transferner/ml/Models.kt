package io.github.cyberneticmedic.transferner.ml

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import io.github.cyberneticmedic.transferner.core.Candidate
import io.github.cyberneticmedic.transferner.core.Catalog
import io.github.cyberneticmedic.transferner.core.DetectorMeta
import io.github.cyberneticmedic.transferner.core.Detections
import io.github.cyberneticmedic.transferner.core.LiveResult
import io.github.cyberneticmedic.transferner.core.Preprocess
import io.github.cyberneticmedic.transferner.core.Runtime
import io.github.cyberneticmedic.transferner.core.SpeciesHead
import io.github.cyberneticmedic.transferner.core.SpeciesModelMeta
import io.github.cyberneticmedic.transferner.core.TensorInfo
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

/** The live detector's app graph (export/detector_app.py): whole-frame genus logits, objectness, boxes. */
class LiveDetector(dir: File, val meta: DetectorMeta, runtime: Runtime, threads: Int, env: Environment) : AutoCloseable {
    private val runner = ModelRunner(File(dir, meta.model).path, meta.signature, runtime, threads, null, env)
    private val inBuf = runner.input(meta.input.name)
    private val outs = meta.outputs
    private val outBufs = listOf(outs.imageLogits, outs.objectness, outs.boxes).associate { it.name to runner.output(it.name) }
    private var smooth: FloatArray? = null
    val accelerator get() = runner.accelerator
    val loadMs get() = runner.loadMs
    var lastMs = 0f
        private set
    var lastPreMs = 0f
        private set

    /** One upright frame. [alpha] blends this frame's genus probabilities into a running average, so the
     *  live label does not flicker between frames. */
    fun run(argb: IntArray, w: Int, h: Int, alpha: Float = 0.4f): LiveResult {
        val t0 = System.nanoTime()
        val x = Preprocess.detectorInput(argb, w, h, meta)
        lastPreMs = (System.nanoTime() - t0) / 1e6f
        if (meta.input.isInt8) inBuf.writeInt8(x) else error("detector input must be int8")
        runner.run(mapOf(meta.input.name to inBuf), outBufs)
        lastMs = runner.lastMs
        val p = Detections.softmax(outBufs.getValue(outs.imageLogits.name).readAsFloat(outs.imageLogits))
        val s = smooth
        val ps = if (s == null) p else FloatArray(p.size) { alpha * p[it] + (1 - alpha) * s[it] }
        smooth = ps
        val obj = outBufs.getValue(outs.objectness.name).readAsFloat(outs.objectness)
        val reg = outBufs.getValue(outs.boxes.name).readAsFloat(outs.boxes)
        val boxes = Detections.boxes(obj, Detections.decode(reg, meta), meta, maxBoxes = 5).filter { it.score >= 0.5f }
        val (genera, fam, famP) = Detections.genera(ps, meta)
        return LiveResult(genera, fam, famP, boxes, meta.size)
    }

    fun resetSmoothing() { smooth = null }

    override fun close() {
        inBuf.close(); outBufs.values.forEach { it.close() }; runner.close()
    }
}

/** BioCLIP image tower + the regional head: one still -> top-5 species (D-002 stage 2). */
class SpeciesIdentifier(
    dir: File, val meta: SpeciesModelMeta, private val head: SpeciesHead, private val catalog: Catalog,
    runtime: Runtime, threads: Int, cache: File?, env: Environment,
) : AutoCloseable {
    private val runner = ModelRunner(File(dir, meta.model).path, meta.signature, runtime, threads, cache, env)
    val usedCache get() = runner.usedCache
    private val inBuf = runner.input(meta.input.name)
    private val outBuf = runner.output(meta.output.name)
    val accelerator get() = runner.accelerator
    val loadMs get() = runner.loadMs

    data class Timing(val preMs: Float, val modelMs: Float, val headMs: Float)

    fun identify(argb: IntArray, w: Int, h: Int): Pair<List<Candidate>, Timing> {
        val t0 = System.nanoTime()
        val x = Preprocess.speciesInput(argb, w, h, meta)
        val t1 = System.nanoTime()
        inBuf.writeFloat(x)
        runner.run(mapOf(meta.input.name to inBuf), mapOf(meta.output.name to outBuf))
        val e = outBuf.readFloat()
        val t2 = System.nanoTime()
        val top = head.top(e, catalog)
        val t3 = System.nanoTime()
        return top to Timing((t1 - t0) / 1e6f, (t2 - t1) / 1e6f, (t3 - t2) / 1e6f)
    }

    override fun close() {
        inBuf.close(); outBuf.close(); runner.close()
    }
}
