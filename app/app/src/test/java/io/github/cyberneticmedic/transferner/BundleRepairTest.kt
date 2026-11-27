package io.github.cyberneticmedic.transferner

import com.sun.net.httpserver.HttpServer
import io.github.cyberneticmedic.transferner.core.BundleJson
import io.github.cyberneticmedic.transferner.core.Latest
import io.github.cyberneticmedic.transferner.core.Manifest
import io.github.cyberneticmedic.transferner.core.ManifestFile
import io.github.cyberneticmedic.transferner.core.SpeciesModelMeta
import io.github.cyberneticmedic.transferner.core.SpeciesModels
import io.github.cyberneticmedic.transferner.core.SpeciesPreprocess
import io.github.cyberneticmedic.transferner.core.TensorInfo
import io.github.cyberneticmedic.transferner.data.BundleRepository
import kotlinx.serialization.encodeToString
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File
import java.io.IOException
import java.net.InetSocketAddress
import java.nio.file.Files
import java.security.MessageDigest

/**
 * An update to a new bundle version moves the unchanged files out of the installed one before fetching the rest
 * (BundleRepository.download). 0.7.1 never moved them back, so an interrupted update left the installed bundle
 * incomplete and the phone had to download every file again. Now a failed download moves them back, and load()
 * repairs a bundle whose update was killed.
 */
class BundleRepairTest {
    private fun sha(b: ByteArray) = MessageDigest.getInstance("SHA-256").digest(b).joinToString("") { "%02x".format(it) }
    private fun mf(path: String, b: ByteArray, group: String? = null) = ManifestFile(path, b.size.toLong(), sha(b), group)

    private val a = "unchanged catalog".toByteArray()
    private val model = ByteArray(4096) { (it * 31).toByte() }
    private val bOld = "old tuning v1".toByteArray()
    private val bNew = "new tuning v2".toByteArray()          // same length as bOld, different content
    private val c = "a file only the new version has".toByteArray()

    private fun installOld(root: File): Manifest {
        val m = Manifest("0.8", files = listOf(mf("catalog.json", a), mf("thumbs/1.jpg", a), mf("tuning.json", bOld),
            mf("species_cpu.tflite", model, "species:cpu")))
        val dir = File(root, "0.8").apply { mkdirs() }
        File(dir, "catalog.json").writeBytes(a)
        File(dir, "thumbs").mkdirs(); File(dir, "thumbs/1.jpg").writeBytes(a)
        File(dir, "tuning.json").writeBytes(bOld)
        File(dir, "species_cpu.tflite").writeBytes(model)
        File(dir, "manifest.json").writeText(BundleJson.encodeToString(m))
        File(root, "current.txt").writeText("0.8 cpu")
        return m
    }

    @Test
    fun repairMovesBackOnlyFilesThatMatchTheInstalledManifest() {
        val root = Files.createTempDirectory("repair").toFile()
        try {
            val m = installOld(root)
            val old = File(root, "0.8"); val new = File(root, "0.9").apply { mkdirs() }
            // what a killed update leaves: unchanged files moved over, a changed file already fetched
            File(new, "thumbs").mkdirs()
            File(old, "catalog.json").renameTo(File(new, "catalog.json"))
            File(old, "thumbs/1.jpg").renameTo(File(new, "thumbs/1.jpg"))
            File(old, "species_cpu.tflite").renameTo(File(new, "species_cpu.tflite"))
            File(old, "tuning.json").delete(); File(new, "tuning.json").writeBytes(bNew)
            val repo = BundleRepository(root)
            assertEquals(3, repo.repairInstalled(old, m, "cpu"))
            assertTrue(File(old, "catalog.json").readBytes().contentEquals(a))
            assertTrue(File(old, "thumbs/1.jpg").exists() && File(old, "species_cpu.tflite").length() == model.size.toLong())
            assertFalse(File(new, "species_cpu.tflite").exists())
            assertFalse("a same-size file with other content must not be taken", File(old, "tuning.json").exists())
            assertTrue(File(new, "tuning.json").readBytes().contentEquals(bNew))
            assertEquals(0, repo.repairInstalled(old, m.copy(files = m.files.filter { it.path != "tuning.json" }), "cpu"))
        } finally {
            root.deleteRecursively()
        }
    }

    @Test
    fun aFailedUpdateGivesTheInstalledBundleItsFilesBack() {
        val root = Files.createTempDirectory("update").toFile()
        val newManifest = Manifest("0.9", files = listOf(mf("catalog.json", a), mf("thumbs/1.jpg", a), mf("tuning.json", bNew),
            mf("species_cpu.tflite", model, "species:cpu"), mf("extra.json", c)))
        val ti = TensorInfo("x", "float32", listOf(1, 3, 224, 224))
        val species = SpeciesModels(listOf(SpeciesModelMeta("cpu", "CPU", "species_cpu.tflite", "cpu", "CPU", signature = "s",
            input = ti, output = ti, preprocess = SpeciesPreprocess(listOf(0f, 0f, 0f), listOf(1f, 1f, 1f)))))
        val served = mapOf(
            "/latest.json" to BundleJson.encodeToString(Latest("0.9", "0.9/manifest.json")).toByteArray(),
            "/0.9/manifest.json" to BundleJson.encodeToString(newManifest).toByteArray(),
            "/0.9/species.json" to BundleJson.encodeToString(species).toByteArray(),
            "/0.9/tuning.json" to bNew,
        )                                                        // extra.json is missing: the update fails on it
        val server = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0).apply {
            createContext("/") { ex ->
                val body = served[ex.requestURI.path]
                if (body == null) { ex.sendResponseHeaders(404, -1) } else { ex.sendResponseHeaders(200, body.size.toLong()); ex.responseBody.use { it.write(body) } }
                ex.close()
            }
            start()
        }
        try {
            val m = installOld(root)
            val repo = BundleRepository(root)
            try {
                repo.download("http://127.0.0.1:${server.address.port}/", { "cpu" }) {}
                fail("the update should fail on the missing file")
            } catch (e: IOException) {
                assertTrue(e.message!!, e.message!!.contains("404"))
            }
            val old = File(root, "0.8")
            assertEquals("0.8" to "cpu", repo.installed())
            for (f in m.files) assertEquals(f.path, f.bytes, File(old, f.path).length())
            assertFalse(File(root, "0.9/catalog.json").exists())
            assertFalse(File(root, "0.9/species_cpu.tflite").exists())
            assertEquals(0, repo.repairInstalled(old, m, "cpu"))
        } finally {
            server.stop(0)
            root.deleteRecursively()
        }
    }
}
