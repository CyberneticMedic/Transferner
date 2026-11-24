package io.github.cyberneticmedic.transferner.data

import android.app.ActivityManager
import android.content.Context
import android.os.Build
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.Environment
import io.github.cyberneticmedic.transferner.core.DeviceInfo

object DeviceProfile {
    fun read(ctx: Context, env: Environment?): DeviceInfo {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val mi = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
        val gpu = runCatching { env?.getAvailableAccelerators()?.contains(Accelerator.GPU) ?: false }.getOrDefault(false)
        return DeviceInfo(
            manufacturer = Build.MANUFACTURER.orEmpty(),
            model = Build.MODEL.orEmpty(),
            socManufacturer = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MANUFACTURER.orEmpty() else "",
            socModel = if (Build.VERSION.SDK_INT >= 31) Build.SOC_MODEL.orEmpty() else Build.HARDWARE.orEmpty(),
            // "8 GB" phones report about 7.5 GB of totalMem (firmware carve-outs); rounded up to 0.5 GB. The planner's
            // thresholds (6 GB for the CPU INT8 variant, 10 GB for the GPU one) sit well clear of the carve-out
            ramGb = (mi.totalMem / (1024.0 * 1024 * 1024)).let { kotlin.math.ceil(it * 2) / 2 }.toFloat(),
            cores = Runtime.getRuntime().availableProcessors(),
            sdk = Build.VERSION.SDK_INT,
            gpuAvailable = gpu,
        )
    }

    /** Memory the system would give us right now, in MB. */
    fun availMb(ctx: Context): Long {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        return ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }.availMem / (1024 * 1024)
    }
}
