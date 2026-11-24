package io.github.cyberneticmedic.transferner.data

import io.github.cyberneticmedic.transferner.core.BundleJson
import io.github.cyberneticmedic.transferner.core.Catalog
import io.github.cyberneticmedic.transferner.core.DetectorMeta
import io.github.cyberneticmedic.transferner.core.FocusMeta
import io.github.cyberneticmedic.transferner.core.ForageMeta
import io.github.cyberneticmedic.transferner.core.Tuning
import io.github.cyberneticmedic.transferner.core.HeadMeta
import io.github.cyberneticmedic.transferner.core.Latest
import io.github.cyberneticmedic.transferner.core.Manifest
import io.github.cyberneticmedic.transferner.core.SpeciesModelMeta
import io.github.cyberneticmedic.transferner.core.SpeciesModels
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

data class Progress(val doneBytes: Long, val totalBytes: Long, val file: String, val variant: String = "")

class BundleFiles(
    val dir: File,
    val manifest: Manifest,
    val variant: String,                         // the species-model variant installed on this phone
    val detector: DetectorMeta,
    val speciesModels: SpeciesModels,
    val head: HeadMeta,
    val catalog: Catalog,
    val focus: FocusMeta? = null,               // optional (bundle 0.3+): the live focus chip's categories
    val tuning: Tuning = Tuning(),              // optional (bundle 0.6+): thresholds; defaults otherwise
    val forage: ForageMeta? = null,             // optional (bundle 0.6+): walk-mode target genera
) {
    val species: SpeciesModelMeta get() = speciesModels[variant] ?: error("variant $variant not in species.json")
}

/**
 * The model bundle (src/transferner/export/bundle.py) under filesDir/bundle/<version>/. Downloads
 * latest.json -> manifest.json -> species.json, lets the caller pick a species-model variant for this
 * phone, then fetches the common files plus that variant's files only. Every file's SHA-256 is checked
 * against the manifest before it is renamed into place; files already present with the right size and
 * hash are kept, so an interrupted download, or a switch to another variant, resumes file by file.
 */
class BundleRepository(private val root: File) {
    private val current = File(root, "current.txt")          // "<version> <variant>"

    fun installed(): Pair<String, String>? {
        val t = current.takeIf { it.exists() }?.readText()?.trim()?.split(" ") ?: return null
        return if (t.size == 2) t[0] to t[1] else null
    }

    fun load(): BundleFiles? {
        val (v, variant) = installed() ?: return null
        val dir = File(root, v)
        val mf = File(dir, "manifest.json").takeIf { it.exists() } ?: return null
        val manifest = BundleJson.decodeFromString<Manifest>(mf.readText())
        if (manifest.files.filter { it.wantedFor(variant) }.any { File(dir, it.path).length() != it.bytes }) return null
        return BundleFiles(
            dir, manifest, variant,
            BundleJson.decodeFromString(File(dir, "detector.json").readText()),
            BundleJson.decodeFromString(File(dir, "species.json").readText()),
            BundleJson.decodeFromString(File(dir, "head.json").readText()),
            BundleJson.decodeFromString(File(dir, "catalog.json").readText()),
            File(dir, "focus.json").takeIf { it.exists() }?.let { BundleJson.decodeFromString<FocusMeta>(it.readText()) },
            File(dir, "tuning.json").takeIf { it.exists() }?.let { BundleJson.decodeFromString<Tuning>(it.readText()) } ?: Tuning(),
            File(dir, "forage.json").takeIf { it.exists() }?.let { BundleJson.decodeFromString<ForageMeta>(it.readText()) },
        )
    }

    private fun get(url: String): HttpURLConnection {
        val c = URL(url).openConnection() as HttpURLConnection
        c.connectTimeout = 10_000
        c.readTimeout = 60_000
        if (c.responseCode != 200) throw IOException("HTTP ${c.responseCode} for $url")
        return c
    }

    private fun text(url: String) = get(url).inputStream.use { it.readBytes().decodeToString() }

    /** The server's current bundle (latest.json). Blocking. */
    fun remoteLatest(server: String): Latest {
        val base = if (server.endsWith("/")) server else "$server/"
        return BundleJson.decodeFromString<Latest>(text(base + "latest.json"))
    }

    fun remoteVersion(server: String): String = remoteLatest(server).version

    /**
     * Blocking; call off the main thread. [choose] picks the variant from the server's species.json (or
     * returns a fixed one). Fails early, before any large file, when the phone lacks the storage.
     */
    fun download(server: String, choose: (SpeciesModels) -> String, appVersion: Long = Long.MAX_VALUE, onProgress: (Progress) -> Unit): Pair<String, String> {
        val base = if (server.endsWith("/")) server else "$server/"
        val latest = BundleJson.decodeFromString<Latest>(text(base + "latest.json"))
        if (latest.minApp > appVersion)
            throw IOException("model bundle ${latest.version} needs a newer app: update the app first (Settings → Updates)")
        val manifestText = text(base + latest.manifest)
        val manifest = BundleJson.decodeFromString<Manifest>(manifestText)
        val vbase = base + manifest.version + "/"
        val models = BundleJson.decodeFromString<SpeciesModels>(text(vbase + "species.json"))
        val variant = choose(models)
        require(models[variant] != null) { "variant $variant is not in this bundle" }
        val dir = File(root, manifest.version).apply { mkdirs() }
        val wanted = manifest.files.filter { it.wantedFor(variant) }
        val total = wanted.sumOf { it.bytes }
        val missing = wanted.filter { File(dir, it.path).length() != it.bytes }.sumOf { it.bytes }
        if (root.usableSpace < missing + 200L * 1024 * 1024)
            throw IOException("not enough storage: need %.0f MB free, have %.0f MB".format(missing / 1048576.0 + 200, root.usableSpace / 1048576.0))
        var done = 0L
        val buf = ByteArray(1 shl 16)
        // a new bundle version: move unchanged files (same path, size and hash) over from the installed version
        // instead of downloading them again (the species model alone is 0.3 to 0.6 GB)
        installed()?.first?.takeIf { it != manifest.version }?.let { prev ->
            val old = BundleJson.decodeFromString<Manifest>(File(root, "$prev/manifest.json").takeIf { it.exists() }?.readText() ?: return@let)
            val oldHash = old.files.associate { it.path to it.sha256 }
            for (f in wanted) {
                val src = File(root, "$prev/${f.path}")
                val dst = File(dir, f.path)
                if (!dst.exists() && oldHash[f.path] == f.sha256 && src.length() == f.bytes) {
                    dst.parentFile?.mkdirs()
                    src.renameTo(dst)
                }
            }
        }
        for (f in wanted) {
            val dst = File(dir, f.path)
            if (dst.length() == f.bytes && sha256(dst) == f.sha256) { done += f.bytes; onProgress(Progress(done, total, f.path, variant)); continue }
            dst.parentFile?.mkdirs()
            val part = File(dst.path + ".part")
            val md = MessageDigest.getInstance("SHA-256")
            get(vbase + f.path).inputStream.use { inp ->
                part.outputStream().use { out ->
                    var last = 0L
                    while (true) {
                        val n = inp.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n); md.update(buf, 0, n); done += n
                        if (done - last > (1 shl 20)) { onProgress(Progress(done, total, f.path, variant)); last = done }
                    }
                }
            }
            val got = md.digest().joinToString("") { "%02x".format(it) }
            if (got != f.sha256) { part.delete(); throw IOException("checksum mismatch for ${f.path}") }
            if (!part.renameTo(dst)) throw IOException("could not move ${f.path} into place")
            onProgress(Progress(done, total, f.path, variant))
        }
        File(dir, "manifest.json").writeText(manifestText)
        // other variants' model files are not needed on this phone
        manifest.files.filter { !it.wantedFor(variant) }.forEach { File(dir, it.path).delete() }
        current.writeText("${manifest.version} $variant")
        root.listFiles()?.filter { it.isDirectory && it.name != manifest.version }?.forEach { it.deleteRecursively() }
        return manifest.version to variant
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
