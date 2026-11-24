package io.github.cyberneticmedic.transferner.data

import android.content.Context
import io.github.cyberneticmedic.transferner.ml.AccelChoice

/** Prototype settings. The default server is the training box on the home network (scripts/serve_bundle.sh). */
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
    var detectorAccel: AccelChoice
        get() = AccelChoice.valueOf(sp.getString("detector_accel", AccelChoice.AUTO.name)!!)
        set(v) = sp.edit().putString("detector_accel", v.name).apply()
    var speciesAccel: AccelChoice
        get() = AccelChoice.valueOf(sp.getString("species_accel", AccelChoice.AUTO.name)!!)
        set(v) = sp.edit().putString("species_accel", v.name).apply()

    companion object {
        const val DEFAULT_SERVER = "http://192.168.1.69:8765/"
    }
}
