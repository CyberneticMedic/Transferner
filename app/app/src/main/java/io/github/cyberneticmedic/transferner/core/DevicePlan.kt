package io.github.cyberneticmedic.transferner.core

/** What the app knows about the phone at first launch (android.os.Build, ActivityManager, LiteRT). */
data class DeviceInfo(
    val manufacturer: String,
    val model: String,
    val socManufacturer: String,                 // Build.SOC_MANUFACTURER (API 31+), "" if unknown
    val socModel: String,                        // Build.SOC_MODEL
    val ramGb: Float,                            // ActivityManager.MemoryInfo.totalMem
    val cores: Int,
    val sdk: Int,
    val gpuAvailable: Boolean,                   // LiteRT Environment.getAvailableAccelerators() has GPU
) {
    val label get() = "$manufacturer $model · ${if (socModel.isBlank()) "SoC unknown" else "$socManufacturer $socModel"} · %.0f GB RAM · API $sdk".format(ramGb)
}

enum class Runtime { GPU, CPU }

/**
 * Which species-model variant and runtime this phone gets (D-032's per-runtime table, D-035's fallback
 * chain). The species model is the heavy one (ViT-H, 0.3 to 0.6 GB), so it only goes to the GPU on SoCs
 * whose GPUs are known to hold it; everything else runs the dynamic-INT8 CPU variant, or the INT4 one when
 * memory is short. The detector is small enough to benchmark both runtimes on the phone itself.
 */
data class Plan(
    val speciesVariant: String,                  // a variant id from species.json
    val speciesRuntime: Runtime,
    val speciesThreads: Int,
    val detectorThreads: Int,
    val liveMaxFps: Int,
    val reason: String,
)

object DevicePlanner {
    /** Flagship GPUs that can take the ViT-H graph: Google Tensor, Snapdragon 8-series (SM8xxx), Dimensity 9xxx (MT698x/MT699x). */
    fun strongGpu(d: DeviceInfo): Boolean {
        val m = d.socManufacturer.lowercase()
        val s = d.socModel.uppercase()
        return when {
            m.contains("google") -> true
            m.contains("qualcomm") || m.contains("qti") -> s.startsWith("SM8")
            m.contains("mediatek") -> s.startsWith("MT698") || s.startsWith("MT699")
            else -> false
        }
    }

    fun plan(d: DeviceInfo, variants: List<SpeciesModelMeta>): Plan {
        val byId = variants.associateBy { it.id }
        val fits = { id: String -> byId[id]?.let { d.ramGb >= it.minRamGb } ?: false }
        val big = d.ramGb >= 10f && d.cores >= 8
        val threads = if (d.cores >= 8) 4 else maxOf(1, d.cores / 2)
        return when {
            d.gpuAvailable && strongGpu(d) && fits("gpu") ->
                Plan("gpu", Runtime.GPU, threads, 2, 15, "flagship GPU (${d.socModel.ifBlank { d.model }}) with %.0f GB RAM".format(d.ramGb))
            fits("cpu") ->
                Plan("cpu", Runtime.CPU, threads, 2, if (big) 15 else 8,
                    if (d.gpuAvailable && !strongGpu(d)) "GPU too small for the species model; dynamic INT8 on the CPU"
                    else "dynamic INT8 on the CPU")
            byId.containsKey("cpu-small") ->
                Plan("cpu-small", Runtime.CPU, threads, 2, 6, "%.0f GB RAM: INT4 weights to save memory".format(d.ramGb))
            else -> Plan(variants.first().id, Runtime.CPU, threads, 2, 6, "no variant fits %.0f GB RAM; trying the smallest available".format(d.ramGb))
        }
    }

    /** The next step down after [failed] crashed or would not load (D-035): GPU -> CPU -> INT4 CPU. */
    fun fallback(failed: Plan, variants: List<SpeciesModelMeta>): Plan? {
        val ids = variants.map { it.id }.toSet()
        return when {
            failed.speciesRuntime == Runtime.GPU && "cpu" in ids ->
                failed.copy(speciesVariant = "cpu", speciesRuntime = Runtime.CPU, reason = "GPU run failed; dynamic INT8 on the CPU")
            failed.speciesVariant == "cpu" && "cpu-small" in ids ->
                failed.copy(speciesVariant = "cpu-small", reason = "CPU INT8 run failed; INT4 weights to save memory")
            failed.speciesThreads > 2 -> failed.copy(speciesThreads = 2, reason = "reduced to 2 threads after a failed run")
            else -> null
        }
    }
}
