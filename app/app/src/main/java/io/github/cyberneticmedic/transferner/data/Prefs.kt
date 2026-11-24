package io.github.cyberneticmedic.transferner.data

import android.content.Context
import io.github.cyberneticmedic.transferner.core.Plan
import io.github.cyberneticmedic.transferner.core.Runtime
import io.github.cyberneticmedic.transferner.ml.AccelChoice

/** Prototype settings and this phone's model plan. The default server is the training box on the home network. */
class Prefs(ctx: Context) {
    private val sp = ctx.getSharedPreferences("transferner", Context.MODE_PRIVATE)

    var server: String
        get() = sp.getString("server", DEFAULT_SERVER) ?: DEFAULT_SERVER
        set(v) = sp.edit().putString("server", v.trim()).apply()
    var showBoxes: Boolean
        get() = sp.getBoolean("show_boxes", true)
        set(v) = sp.edit().putBoolean("show_boxes", v).apply()
    var showTimings: Boolean
        get() = sp.getBoolean("show_timings", true)
        set(v) = sp.edit().putBoolean("show_timings", v).apply()

    /** Live boxes are drawn when their plant score (objectness) is at least this (D-039; 0.25 by default). */
    var boxThreshold: Float
        get() = sp.getFloat("box_threshold", 0.25f)
        set(v) = sp.edit().putFloat("box_threshold", v).apply()

    /** The species-model plan (DevicePlanner, then any fallbacks); null until first planned. */
    var plan: Plan?
        get() {
            val v = sp.getString("plan_variant", null) ?: return null
            return Plan(v, Runtime.valueOf(sp.getString("plan_runtime", "CPU")!!), sp.getInt("plan_threads", 4),
                sp.getInt("plan_det_threads", 2), sp.getInt("plan_fps", 8), sp.getString("plan_reason", "")!!)
        }
        set(p) {
            val e = sp.edit()
            if (p == null) e.remove("plan_variant") else e.putString("plan_variant", p.speciesVariant)
                .putString("plan_runtime", p.speciesRuntime.name).putInt("plan_threads", p.speciesThreads)
                .putInt("plan_det_threads", p.detectorThreads).putInt("plan_fps", p.liveMaxFps).putString("plan_reason", p.reason)
            e.commit()
        }

    var detectorAccel: AccelChoice
        get() = AccelChoice.valueOf(sp.getString("detector_accel", AccelChoice.AUTO.name)!!)
        set(v) = sp.edit().putString("detector_accel", v.name).apply()

    /** Result of the first-launch detector benchmark: chosen runtime and the per-runtime times. */
    var detectorBench: String
        get() = sp.getString("detector_bench", "") ?: ""
        set(v) = sp.edit().putString("detector_bench", v).apply()
    var detectorBenchRuntime: Runtime?
        get() = sp.getString("detector_bench_runtime", null)?.let { Runtime.valueOf(it) }
        set(v) = sp.edit().putString("detector_bench_runtime", v?.name).apply()

    /**
     * Crash guard (D-035): set synchronously before a native load or run, cleared after. If the app starts
     * with it still set, that operation killed the process (typically out of memory) and the app steps down
     * its plan instead of crashing again.
     */
    var pending: String?
        get() = sp.getString("pending", null)
        set(v) { sp.edit().putString("pending", v).commit() }

    /** One crash flag per model thread (0.6 review): the detector benchmark and the species load run at the same
     *  time, and one shared key was overwritten by the other thread. Values: the runtime or step in progress. */
    var pendingDetector: String?
        get() = sp.getString("pending_detector", null)
        set(v) { sp.edit().putString("pending_detector", v).commit() }
    var pendingSpecies: String?
        get() = sp.getString("pending_species", null)
        set(v) { sp.edit().putString("pending_species", v).commit() }

    /** Detector benchmark bookkeeping (D-044): the bundle it ran on (re-run after a model update), the CPU thread
     *  count it chose (2 or 4), and why a runtime failed (shown and uploaded instead of silently dropped). */
    var benchBundle: String
        get() = sp.getString("bench_bundle", "") ?: ""
        set(v) = sp.edit().putString("bench_bundle", v).apply()
    var detectorThreads: Int
        get() = sp.getInt("detector_threads", 0)
        set(v) = sp.edit().putInt("detector_threads", v).apply()
    var benchErrors: String
        get() = sp.getString("bench_errors", "") ?: ""
        set(v) = sp.edit().putString("bench_errors", v).apply()

    /** Walk mode (D-044): dim the screen while it stays on, buzz on alerts, record frames for training. */
    var walkDim: Boolean
        get() = sp.getBoolean("walk_dim", true)
        set(v) = sp.edit().putBoolean("walk_dim", v).apply()
    var walkHaptics: Boolean
        get() = sp.getBoolean("walk_haptics", true)
        set(v) = sp.edit().putBoolean("walk_haptics", v).apply()
    var recordWalks: Boolean
        get() = sp.getBoolean("record_walks", true)
        set(v) = sp.edit().putBoolean("record_walks", v).apply()

    /** The user's forage tiers, overriding forage.json: genus -> ALERT | SHOW | OFF. */
    var forageTiers: Map<String, io.github.cyberneticmedic.transferner.core.Tier>
        get() = (sp.getStringSet("forage_tiers", emptySet()) ?: emptySet()).mapNotNull { e ->
            val (g, t) = e.split("=").takeIf { it.size == 2 } ?: return@mapNotNull null
            runCatching { g to io.github.cyberneticmedic.transferner.core.Tier.valueOf(t) }.getOrNull()
        }.toMap()
        set(v) = sp.edit().putStringSet("forage_tiers", v.map { "${it.key}=${it.value.name}" }.toSet()).apply()

    /** Random per-install suffix for log uploads (tells two phones of the same model apart). */
    val installId: String
        get() = sp.getString("install_id", null) ?: java.util.UUID.randomUUID().toString().take(6).also {
            sp.edit().putString("install_id", it).apply()
        }

    companion object {
        const val DEFAULT_SERVER = "http://192.168.1.69:8765/"
    }
}
