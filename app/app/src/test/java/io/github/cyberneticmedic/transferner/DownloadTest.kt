package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.DeviceInfo
import io.github.cyberneticmedic.transferner.core.DevicePlanner
import io.github.cyberneticmedic.transferner.data.BundleRepository
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The app's bundle download (latest.json -> manifest -> every file, SHA-256 checked) against a running
 * scripts/serve_bundle.sh. Opt-in: ./gradlew testDebugUnitTest -Ptransferner.server=http://127.0.0.1:8765/
 */
class DownloadTest {
    @Test fun downloadsAndVerifiesBundle() {
        val server = System.getProperty("transferner.server").orEmpty()
        assumeTrue("set -Ptransferner.server to run", server.isNotBlank())
        val root = File(System.getProperty("java.io.tmpdir"), "transferner-bundle-test").apply { deleteRecursively(); mkdirs() }
        try {
            val repo = BundleRepository(root)
            var calls = 0
            val t0 = System.nanoTime()
            // a Moto G Power 2025 profile: the planner gives the CPU variant, and only its model file comes down
            val moto = DeviceInfo("motorola", "moto g power - 2025", "Mediatek", "MT6835", 8f, 8, 35, true)
            val v = repo.download(server, { DevicePlanner.plan(moto, it.variants).speciesVariant }) { calls++ }
            println("downloaded bundle $v in %.1f s (%d progress callbacks)".format((System.nanoTime() - t0) / 1e9, calls))
            val files = repo.load()
            assertNotNull(files)
            assertEquals("cpu", files!!.variant)
            // app 0.7 takes the regions detector (group app:11) when the bundle has one
            if (files.manifest.files.any { it.path == BundleRepository.REGIONS_META })
                assertEquals("detector_rg.tflite", files.detector.model)
            assertEquals(files.head.n, files.catalog.species.size)
            val models = files.dir.listFiles()!!.filter { it.name.startsWith("species_") && it.name.endsWith(".tflite") }.map { it.name }
            assertEquals(listOf("species_cpu.tflite"), models)
            // switching to the INT4 variant keeps the shared files and swaps the model file
            val t1 = System.nanoTime()
            repo.download(server, { "cpu-small" }) {}
            println("switch to cpu-small %.1f s".format((System.nanoTime() - t1) / 1e9))
            val small = repo.load()!!
            assertEquals("cpu-small", small.variant)
            assertEquals(listOf("species_cpu-small.tflite"),
                small.dir.listFiles()!!.filter { it.name.startsWith("species_") && it.name.endsWith(".tflite") }.map { it.name })
        } finally {
            root.deleteRecursively()
        }
    }
}
