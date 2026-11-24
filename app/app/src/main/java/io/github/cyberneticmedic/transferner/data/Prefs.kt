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

    /** Random per-install suffix for log uploads (tells two phones of the same model apart). */
    val installId: String
        get() = sp.getString("install_id", null) ?: java.util.UUID.randomUUID().toString().take(6).also {
            sp.edit().putString("install_id", it).apply()
        }

    companion object {
        const val DEFAULT_SERVER = "http://192.168.1.69:8765/"
    }
}
