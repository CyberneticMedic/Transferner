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
 * On-phone observation log (D-014/D-016): one JPEG per capture and one CSV row with the top 5, the caution
 * flag and the model versions. Location columns exist but stay empty until location is added.
 */
class ObservationLog(private val dir: File) {
    val csv = File(dir, "observations.csv")
    private val header = listOf("time_utc", "latitude", "longitude", "photo") +
        (1..5).flatMap { listOf("species_$it", "class_$it", "p_$it") } +
        listOf("caution", "bundle", "species_model", "detector_model", "user_note")

    private fun q(s: String) = "\"" + s.replace("\"", "\"\"") + "\""

    fun add(jpeg: ByteArray, top: List<Candidate>, caution: Boolean, bundle: String, speciesModel: String, detectorModel: String,
            note: String = "") {
        dir.mkdirs()
        val now = Date()
        val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(now)
        val photo = File(dir, "obs-$stamp.jpg")
        photo.writeBytes(jpeg)
        val iso = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }.format(now)
        val cells = mutableListOf(iso, "", "", photo.name)
        for (k in 0 until 5) {
            val c = top.getOrNull(k)
            cells += listOf(c?.species?.name.orEmpty(), c?.species?.i?.toString().orEmpty(), c?.let { "%.4f".format(Locale.US, it.p) }.orEmpty())
        }
        cells += listOf(if (caution) "1" else "0", bundle, speciesModel, detectorModel, note.replace('\n', ' ').trim())
        if (!csv.exists()) csv.writeText(header.joinToString(",") + "\n")
        else csv.readLines().let { l ->                               // a 0.2 log: add the user_note column name
            if (l.isNotEmpty() && !l[0].contains("user_note")) csv.writeText((listOf(header.joinToString(",")) + l.drop(1)).joinToString("\n") + "\n")
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
            if (c.size < header.size - 1) return@mapNotNull null          // 0.2 rows have no user_note
            val top = (0 until 5).mapNotNull { k ->
                val idx = c[5 + 3 * k].toIntOrNull() ?: return@mapNotNull null
                val p = c[6 + 3 * k].toFloatOrNull() ?: return@mapNotNull null
                catalog.species.getOrNull(idx)?.let { Candidate(it, p) }
            }
            LogEntry(c[0], File(dir, c[3]), top, c[19] == "1", c.getOrElse(23) { "" })
        }.reversed()
    }

    /**
     * Send the log to the home server (scripts/bundle_server.py): the CSV every time, photos once each.
     * Blocking; returns how many photos went up.
     */
    fun upload(server: String, device: String, onProgress: (Int, Int) -> Unit): Int {
        val base = (if (server.endsWith("/")) server else "$server/") + "upload/$device/"
        val sent = File(dir, "uploaded.txt")
        val done = if (sent.exists()) sent.readLines().toMutableSet() else mutableSetOf()
        val photos = dir.listFiles { f -> f.name.endsWith(".jpg") && f.name !in done }.orEmpty().sortedBy { it.name }
        photos.forEachIndexed { i, f ->
            post(base + f.name, f.readBytes(), "image/jpeg")
            sent.appendText(f.name + "\n")
            onProgress(i + 1, photos.size)
        }
        if (csv.exists()) post(base + csv.name, csv.readBytes(), "text/csv")
        return photos.size
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
