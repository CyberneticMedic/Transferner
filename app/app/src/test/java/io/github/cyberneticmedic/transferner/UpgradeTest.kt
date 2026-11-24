package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.data.BundleRepository
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.file.Files

/**
 * Bundle version upgrade (app 0.4): files unchanged between versions are moved over, not re-downloaded.
 * Opt-in against a mirror whose latest.json the test flips:
 *   -Ptransferner.mirror=<dir with 0.2/, 0.3/, latest.json> -Ptransferner.mirrorServer=http://127.0.0.1:8766/
 */
class UpgradeTest {
    @Test fun upgradeKeepsUnchangedFiles() {
        val mirror = System.getProperty("transferner.mirror").orEmpty()
        val server = System.getProperty("transferner.mirrorServer").orEmpty()
        assumeTrue(mirror.isNotBlank() && server.isNotBlank())
        val latest = File(mirror, "latest.json")
        val root = File(System.getProperty("java.io.tmpdir"), "transferner-upgrade-test").apply { deleteRecursively(); mkdirs() }
        try {
            latest.writeText("""{"version": "0.2", "manifest": "0.2/manifest.json"}""")
            val repo = BundleRepository(root)
            repo.download(server, { "cpu" }) {}
            val ino = { f: File -> Files.getAttribute(f.toPath(), "unix:ino") }
            val before = ino(File(root, "0.2/species_cpu.tflite"))
            latest.writeText("""{"version": "0.3", "manifest": "0.3/manifest.json"}""")
            var fetched = 0L
            var last = 0L
            val t0 = System.nanoTime()
            repo.download(server, { "cpu" }) { p -> last = p.doneBytes }
            println("upgrade 0.2 -> 0.3 took %.1f s".format((System.nanoTime() - t0) / 1e9))
            assertEquals("0.3" to "cpu", repo.installed())
            assertEquals("species model moved, not re-downloaded", before, ino(File(root, "0.3/species_cpu.tflite")))
            assertEquals(false, File(root, "0.2").exists())
            assertEquals(true, repo.load()?.focus != null)
            println("focus categories: " + repo.load()!!.focus!!.categories.joinToString { "${it.label} ${it.genera.size}" })
        } finally {
            root.deleteRecursively()
            latest.writeText("""{"version": "0.2", "manifest": "0.2/manifest.json"}""")
        }
    }
}
