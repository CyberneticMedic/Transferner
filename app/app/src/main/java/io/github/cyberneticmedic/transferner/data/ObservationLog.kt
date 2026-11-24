package io.github.cyberneticmedic.transferner.data

import android.content.Context
import io.github.cyberneticmedic.transferner.core.Candidate
import io.github.cyberneticmedic.transferner.core.Catalog
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

data class LogEntry(val time: String, val photo: File, val top: List<Candidate>, val caution: Boolean, val note: String = "")

/**
 * On-phone observation log (D-014/D-016): one JPEG per capture (two when a second photo was fused) and one CSV
 * row with the top 5, the caution flag, the model versions and (0.6+) the identification time, text score, result
 * level and photo count. Each capture also gets obs-<stamp>.json with everything the screen showed and the last
 * live frames before the capture (D-044). Location columns exist but stay empty until location is added.
 */
class ObservationLog(private val dir: File) {
    val csv = File(dir, "observations.csv")
    private val header = listOf("time_utc", "latitude", "longitude", "photo") +
        (1..5).flatMap { listOf("species_$it", "class_$it", "p_$it") } +
        listOf("caution", "bundle", "species_model", "detector_model", "user_note", "id_ms", "text_score", "level", "photos")

    private fun q(s: String) = "\"" + s.replace("\"", "\"\"") + "\""

    fun add(jpeg: ByteArray, top: List<Candidate>, caution: Boolean, bundle: String, speciesModel: String, detectorModel: String,
            note: String = "", idMs: Float = 0f, text: Float? = null, level: String = "", second: ByteArray? = null,
            sidecar: ((String) -> String)? = null) {
        dir.mkdirs()
        val now = Date()
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(now)
        val photo = File(dir, "obs-$stamp.jpg")
        photo.writeBytes(jpeg)
        second?.let { File(dir, "obs-$stamp-2.jpg").writeBytes(it) }
        sidecar?.let { File(dir, "obs-$stamp.json").writeText(it(photo.name)) }
        val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }.format(now)
        val cells = mutableListOf(iso, "", "", photo.name)
        for (k in 0 until 5) {
            val c = top.getOrNull(k)
            cells += listOf(c?.species?.name.orEmpty(), c?.species?.i?.toString().orEmpty(), c?.let { "%.4f".format(Locale.US, it.p) }.orEmpty())
        }
        cells += listOf(if (caution) "1" else "0", bundle, speciesModel, detectorModel, note.replace('\n', ' ').trim(),
            "%.0f".format(Locale.US, idMs), text?.let { "%.4f".format(Locale.US, it) }.orEmpty(), level, if (second != null) "2" else "1")
        if (!csv.exists()) csv.writeText(header.joinToString(",") + "\n")
        else csv.readLines().let { l ->                               // an older log: write the current header
            if (l.isNotEmpty() && l[0] != header.joinToString(",")) csv.writeText((listOf(header.joinToString(",")) + l.drop(1)).joinToString("\n") + "\n")
        }
        csv.appendText(cells.joinToString(",") { q(it) } + "\n")
    }

    private fun parse(line: String): List<String> {
        val out = mutableListOf<String>()
        val sb = StringBuilder()
        var inQ = false
        var i = 0
        while (i < line.length) {
            val ch = line[i]
            when {
                inQ && ch == '"' && i + 1 < line.length && line[i + 1] == '"' -> { sb.append('"'); i++ }
                ch == '"' -> inQ = !inQ
                ch == ',' && !inQ -> { out += sb.toString(); sb.clear() }
                else -> sb.append(ch)
            }
            i++
        }
        out += sb.toString()
        return out
    }

    fun entries(catalog: Catalog): List<LogEntry> {
        if (!csv.exists()) return emptyList()
        val lines = csv.readLines().drop(1).filter { it.isNotBlank() }
        return lines.mapNotNull { line ->
            val c = parse(line)
            if (c.size < 23) return@mapNotNull null                       // 0.2 rows end at detector_model
            val top = (0 until 5).mapNotNull { k ->
                val idx = c[5 + 3 * k].toIntOrNull() ?: return@mapNotNull null
                val p = c[6 + 3 * k].toFloatOrNull() ?: return@mapNotNull null
                catalog.species.getOrNull(idx)?.let { Candidate(it, p) }
            }
            LogEntry(c[0], File(dir, c[3]), top, c[19] == "1", c.getOrElse(23) { "" })
        }.reversed()
    }

    /**
     * Send the log to the home server (scripts/bundle_server.py). The CSV goes first (0.5 sent it last, and an
     * interrupted upload on 2026-11-24 left five photos without their rows); then each capture's photos and
     * sidecar together, each file once; then finished walk recordings ([walkFiles]); then the device report.
     * Blocking; returns how many files went up.
     */
    fun upload(server: String, device: String, report: String, walkFiles: List<File> = emptyList(), onProgress: (Int, Int) -> Unit): Int {
        val base = (if (server.endsWith("/")) server else "$server/") + "upload/$device/"
        val sent = File(dir, "uploaded.txt")
        val done = if (sent.exists()) sent.readLines().toMutableSet() else mutableSetOf()
        if (csv.exists()) post(base + csv.name, csv.readBytes(), "text/csv")
        val obs = dir.listFiles { f -> f.name.startsWith("obs-") && (f.name.endsWith(".jpg") || f.name.endsWith(".json")) && f.name !in done }
            .orEmpty().sortedBy { it.name }
        val files = obs + walkFiles.filter { it.name !in done }
        files.forEachIndexed { i, f ->
            // a file the server refuses for its size (empty, or over its 25 MB limit) is skipped, not retried
            // forever: one bad file must not block every later upload
            if (f.length() > 0) {
                try {
                    post(base + f.name, f.readBytes(), if (f.name.endsWith(".jpg")) "image/jpeg" else "application/json")
                } catch (e: java.io.IOException) {
                    if (e.message?.startsWith("HTTP 413") != true) throw e
                }
            }
            sent.appendText(f.name + "\n")
            onProgress(i + 1, files.size)
        }
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        post(base + "device-$stamp.json", report.toByteArray(), "application/json")
        return files.size
    }

    private fun post(url: String, body: ByteArray, type: String) {
        val c = java.net.URL(url).openConnection() as java.net.HttpURLConnection
        c.requestMethod = "POST"
        c.doOutput = true
        c.connectTimeout = 10_000
        c.readTimeout = 60_000
        c.setRequestProperty("Content-Type", type)
        c.setFixedLengthStreamingMode(body.size)
        c.outputStream.use { it.write(body) }
        if (c.responseCode !in 200..299) throw java.io.IOException("HTTP ${c.responseCode} for $url")
        c.disconnect()
    }

    companion object {
        fun of(ctx: Context) = ObservationLog(File(ctx.filesDir, "log"))
    }
}
