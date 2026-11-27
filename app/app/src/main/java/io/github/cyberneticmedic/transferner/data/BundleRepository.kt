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
/** [appVersion]: this app's versionCode; files grouped "app:<N>" are fetched and used only when it is >= N. */
class BundleRepository(private val root: File, private val appVersion: Long = Long.MAX_VALUE) {
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
        // not while a download (from this or an earlier screen of the app) is moving files between the directories
        if (LOCK.tryLock()) try { repairInstalled(dir, manifest, variant) } finally { LOCK.unlock() }
        // files every app needs must be complete; "app:<N>" files are extras (an app updated after the bundle
        // was downloaded lacks them until missingExtras() offers the same bundle again)
        if (manifest.files.filter { it.wantedFor(variant, 0) }.any { File(dir, it.path).length() != it.bytes }) return null
        val have = { path: String -> manifest.files.any { it.path == path && it.wantedFor(variant, appVersion) && File(dir, path).length() == it.bytes } }
        // the regions detector (bundle 0.7+, group "app:11") when this app has both its files; else the 3-output one
        val det = if (have(REGIONS_META) && have(REGIONS_MODEL)) REGIONS_META else "detector.json"
        return BundleFiles(
            dir, manifest, variant,
            BundleJson.decodeFromString(File(dir, det).readText()),
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

    /** True when the installed bundle has "app:<N>" files this app can use but has not fetched (the app was
     *  updated after the bundle was downloaded). Downloading the same version again fetches only those. */
    fun missingExtras(): Boolean {
        val (v, variant) = installed() ?: return false
        val dir = File(root, v)
        val manifest = File(dir, "manifest.json").takeIf { it.exists() }?.let { BundleJson.decodeFromString<Manifest>(it.readText()) } ?: return false
        return manifest.files.any { it.wantedFor(variant, appVersion) && !it.wantedFor(variant, 0) && File(dir, it.path).length() != it.bytes }
    }

    /**
     * Any file the installed manifest lists but [dir] lacks is taken from another version directory under the root
     * when its size and SHA-256 match the installed manifest. That is where an interrupted update leaves them:
     * download() moves unchanged files into the new version's directory before fetching the rest, and a killed app
     * never moves them back. In 0.7.1 and earlier the installed bundle then stopped loading (Setup, no models on the
     * phone) until the update was retried and finished. Returns how many files were moved back.
     */
    fun repairInstalled(dir: File, manifest: Manifest, variant: String): Int {
        val lacking = manifest.files.filter { it.wantedFor(variant, appVersion) && File(dir, it.path).length() != it.bytes }
        if (lacking.isEmpty()) return 0
        val others = root.listFiles()?.filter { it.isDirectory && it != dir }.orEmpty()
        var n = 0
        for (f in lacking) {
            val src = others.map { File(it, f.path) }.firstOrNull { it.length() == f.bytes && sha256(it) == f.sha256 } ?: continue
            val dst = File(dir, f.path)
            dst.parentFile?.mkdirs()
            if (dst.exists()) dst.delete()
            if (src.renameTo(dst)) n++
        }
        return n
    }

    /**
     * Blocking; call off the main thread. [choose] picks the variant from the server's species.json (or
     * returns a fixed one). Fails early, before any large file, when the phone lacks the storage.
     */
    fun download(server: String, choose: (SpeciesModels) -> String, appVersion: Long = this.appVersion, onProgress: (Progress) -> Unit): Pair<String, String> {
        // one download at a time in the whole process (an earlier screen's download can outlive its ViewModel),
        // and never beside load()'s repair
        if (!LOCK.tryLock(30, java.util.concurrent.TimeUnit.SECONDS)) throw IOException("a model download is already running; wait for it to finish")
        try {
            return downloadLocked(server, choose, appVersion, onProgress)
        } finally {
            LOCK.unlock()
        }
    }

    private fun downloadLocked(server: String, choose: (SpeciesModels) -> String, appVersion: Long, onProgress: (Progress) -> Unit): Pair<String, String> {
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
        val wanted = manifest.files.filter { it.wantedFor(variant, appVersion) }
        val total = wanted.sumOf { it.bytes }
        val missing = wanted.filter { File(dir, it.path).length() != it.bytes }.sumOf { it.bytes }
        if (root.usableSpace < missing + 200L * 1024 * 1024)
            throw IOException("not enough storage: need %.0f MB free, have %.0f MB".format(missing / 1048576.0 + 200, root.usableSpace / 1048576.0))
        var done = 0L
        val buf = ByteArray(1 shl 16)
        // a new bundle version: move unchanged files (same path, size and hash) over from the installed version
        // instead of downloading them again (the species model alone is 0.3 to 0.6 GB). If this download fails
        // they are moved back (below); if the app is killed, load() moves them back (repairInstalled).
        val moved = mutableListOf<Pair<File, File>>()
        val prevVersion = installed()?.first
        prevVersion?.takeIf { it != manifest.version }?.let { prev ->
            val old = BundleJson.decodeFromString<Manifest>(File(root, "$prev/manifest.json").takeIf { it.exists() }?.readText() ?: return@let)
            val oldHash = old.files.associate { it.path to it.sha256 }
            for (f in wanted) {
                val src = File(root, "$prev/${f.path}")
                val dst = File(dir, f.path)
                if (!dst.exists() && oldHash[f.path] == f.sha256 && src.length() == f.bytes) {
                    dst.parentFile?.mkdirs()
                    if (src.renameTo(dst)) moved += src to dst
                }
            }
        }
        try {
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
        } catch (e: Throwable) {
            // the installed bundle (still named by current.txt) gets its files back, so it keeps working
            if (installed()?.first == prevVersion)
                for ((src, dst) in moved) if (!src.exists() && dst.exists()) { src.parentFile?.mkdirs(); dst.renameTo(src) }
            throw e
        }
        // other variants' model files are not needed on this phone
        manifest.files.filter { !it.wantedFor(variant, appVersion) }.forEach { File(dir, it.path).delete() }
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

    companion object {
        /** Serialises download() and load()'s repair across every repository instance in the process. */
        private val LOCK = java.util.concurrent.locks.ReentrantLock()

        /** Sidecar of the detector with region outputs (bundle.py --detector-regions). */
        const val REGIONS_META = "detector_rg.json"
        const val REGIONS_MODEL = "detector_rg.tflite"
    }
}
