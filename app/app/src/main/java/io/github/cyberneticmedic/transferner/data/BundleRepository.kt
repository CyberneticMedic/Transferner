package io.github.cyberneticmedic.transferner.data

import io.github.cyberneticmedic.transferner.core.BundleJson
import io.github.cyberneticmedic.transferner.core.Catalog
import io.github.cyberneticmedic.transferner.core.DetectorMeta
import io.github.cyberneticmedic.transferner.core.HeadMeta
import io.github.cyberneticmedic.transferner.core.Latest
import io.github.cyberneticmedic.transferner.core.Manifest
import io.github.cyberneticmedic.transferner.core.SpeciesModelMeta
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

data class Progress(val doneBytes: Long, val totalBytes: Long, val file: String)

class BundleFiles(
    val dir: File,
    val manifest: Manifest,
    val detector: DetectorMeta,
    val species: SpeciesModelMeta,
    val head: HeadMeta,
    val catalog: Catalog,
)

/**
 * The model bundle (src/transferner/export/bundle.py) under filesDir/bundle/<version>/. Downloads
 * latest.json -> manifest.json -> each file, verifying every file's SHA-256 against the manifest before it
 * is renamed into place; files already present with the right size and hash are kept, so an interrupted
 * download resumes file by file.
 */
class BundleRepository(private val root: File) {
    private val current = File(root, "current.txt")

    fun installedVersion(): String? = current.takeIf { it.exists() }?.readText()?.trim()?.ifBlank { null }

    fun load(): BundleFiles? {
        val v = installedVersion() ?: return null
        val dir = File(root, v)
        val mf = File(dir, "manifest.json").takeIf { it.exists() } ?: return null
        val manifest = BundleJson.decodeFromString<Manifest>(mf.readText())
        if (manifest.files.any { File(dir, it.path).length() != it.bytes }) return null
        return BundleFiles(
            dir, manifest,
            BundleJson.decodeFromString(File(dir, "detector.json").readText()),
            BundleJson.decodeFromString(File(dir, "species.json").readText()),
            BundleJson.decodeFromString(File(dir, "head.json").readText()),
            BundleJson.decodeFromString(File(dir, "catalog.json").readText()),
        )
    }

    private fun get(url: String): HttpURLConnection {
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 10_000
        c.readTimeout = 60_000
        if (c.responseCode != 200) throw IOException("HTTP ${c.responseCode} for $url")
        return c
    }

    /** Blocking; call off the main thread. [onProgress] is called often. */
    fun download(server: String, onProgress: (Progress) -> Unit): String {
        val base = if (server.endsWith("/")) server else "$server/"
        val latest = BundleJson.decodeFromString<Latest>(get(base + "latest.json").inputStream.use { it.readBytes().decodeToString() })
        val manifestText = get(base + latest.manifest).inputStream.use { it.readBytes().decodeToString() }
        val manifest = BundleJson.decodeFromString<Manifest>(manifestText)
        val dir = File(root, manifest.version).apply { mkdirs() }
        val total = manifest.files.sumOf { it.bytes }
        var done = 0L
        val buf = ByteArray(1 shl 16)
        for (f in manifest.files) {
            val dst = File(dir, f.path)
            if (dst.length() == f.bytes && sha256(dst) == f.sha256) { done += f.bytes; onProgress(Progress(done, total, f.path)); continue }
            dst.parentFile?.mkdirs()
            val part = File(dst.path + ".part")
            val md = MessageDigest.getInstance("SHA-256")
            get(base + manifest.version + "/" + f.path).inputStream.use { inp ->
                part.outputStream().use { out ->
                    var last = 0L
                    while (true) {
                        val n = inp.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n); md.update(buf, 0, n); done += n
                        if (done - last > (1 shl 20)) { onProgress(Progress(done, total, f.path)); last = done }
                    }
                }
            }
            val got = md.digest().joinToString("") { "%02x".format(it) }
            if (got != f.sha256) { part.delete(); throw IOException("checksum mismatch for ${f.path}") }
            if (!part.renameTo(dst)) throw IOException("could not move ${f.path} into place")
            onProgress(Progress(done, total, f.path))
        }
        File(dir, "manifest.json").writeText(manifestText)
        current.writeText(manifest.version)
        root.listFiles()?.filter { it.isDirectory && it.name != manifest.version }?.forEach { it.deleteRecursively() }
        return manifest.version
    }

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { inp ->
            val b = ByteArray(1 shl 16)
            while (true) { val n = inp.read(b); if (n < 0) break; md.update(b, 0, n) }
        }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
