package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.DeviceInfo
import io.github.cyberneticmedic.transferner.core.DevicePlanner
import io.github.cyberneticmedic.transferner.core.Runtime
import io.github.cyberneticmedic.transferner.core.SpeciesModelMeta
import io.github.cyberneticmedic.transferner.core.SpeciesPreprocess
import io.github.cyberneticmedic.transferner.core.TensorInfo
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/** The species-model plan and its fallback chain (D-035) on the phones the prototype targets. */
class DevicePlanTest {
    private fun v(id: String, acc: String, ram: Float) = SpeciesModelMeta(
        id, id, "species_$id.tflite", id, acc, ram, 0, "serving_default",
        TensorInfo("args_0", "float32", listOf(1, 3, 224, 224)), TensorInfo("output_0", "float32", listOf(1, 1024)),
        SpeciesPreprocess(listOf(0f, 0f, 0f), listOf(1f, 1f, 1f)))
    private val all = listOf(v("gpu", "GPU", 10f), v("cpu", "CPU", 6f), v("cpu-small", "CPU", 0f))

    private val pixel9pro = DeviceInfo("Google", "Pixel 9 Pro", "Google", "Tensor G4", 16f, 8, 36, true)
    private val motoGPower2025 = DeviceInfo("motorola", "moto g power - 2025", "Mediatek", "MT6835", 8f, 8, 35, true)
    private val motoG2025 = DeviceInfo("motorola", "moto g - 2025", "Mediatek", "MT6835", 4f, 8, 35, true)

    @Test fun pixelGetsGpu() {
        val p = DevicePlanner.plan(pixel9pro, all)
        assertEquals("gpu", p.speciesVariant); assertEquals(Runtime.GPU, p.speciesRuntime)
    }

    @Test fun motoGPower2025GetsCpuInt8() {
        val p = DevicePlanner.plan(motoGPower2025, all)
        assertEquals("cpu", p.speciesVariant); assertEquals(Runtime.CPU, p.speciesRuntime)
        assertEquals(4, p.speciesThreads); assertEquals(8, p.liveMaxFps)
        println("moto g power 2025: $p")
    }

    @Test fun fourGbPhoneGetsInt4() = assertEquals("cpu-small", DevicePlanner.plan(motoG2025, all).speciesVariant)

    @Test fun noGpuVariantOnFlagshipWithoutLiteRtGpu() =
        assertEquals("cpu", DevicePlanner.plan(pixel9pro.copy(gpuAvailable = false), all).speciesVariant)

    @Test fun fallbackChain() {
        var p = DevicePlanner.plan(pixel9pro, all)
        val steps = mutableListOf(p.speciesVariant + "/" + p.speciesRuntime + "/" + p.speciesThreads)
        while (true) {
            p = DevicePlanner.fallback(p, all) ?: break
            steps += p.speciesVariant + "/" + p.speciesRuntime + "/" + p.speciesThreads
        }
        assertEquals(listOf("gpu/GPU/4", "cpu/CPU/4", "cpu-small/CPU/4", "cpu-small/CPU/2"), steps)
        assertNull(DevicePlanner.fallback(p, all))
    }
}
