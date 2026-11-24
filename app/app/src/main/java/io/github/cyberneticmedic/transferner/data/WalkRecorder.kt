package io.github.cyberneticmedic.transferner.data

import java.io.BufferedWriter
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Walk recordings (D-044; audit data_domain-06): the live model's output on every analysed frame as one JSON
 * line, plus about one frame a second as a small JPEG, so thresholds, the tracker and alert rules can be tuned
 * on real walking frames and the frames can become training negatives and pseudo-labels. Files sit next to the
 * observation log as walk-<session>.jsonl and walk-<session>-<ms>.jpg; frames stop being kept above
 * [maxBytes] (the JSON lines continue). A session's JSONL is uploaded once the session has ended.
 */
class WalkRecorder(private val dir: File, private val maxBytes: Long = 1_500_000_000L) {
    var session: String? = null
        private set
    private var out: BufferedWriter? = null
    private var lines = 0
    private var bytes = -1L
    private var part = 1
    private var current: File? = null                  // the JSONL being written (never uploaded while open)
    private var written = 0L

    @Synchronized fun start(): String {
        stop()
        dir.mkdirs()
        val s = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
        session = s
        lines = 0; part = 1; written = 0
        bytes = used()
        return s
    }

    @Synchronized fun stop() {
        runCatching { out?.run { flush(); close() } }
        out = null
        current = null
        session = null
    }

    /** The JSONL opens with its first line, so a session that analysed nothing leaves no empty file (the
     *  server refuses empty uploads, and 0.6's first build then stalled every later upload). */
    @Synchronized fun line(json: String) {
        val s = session ?: return
        if (out != null && written > PART_BYTES) {          // stay under the server's 25 MB per file
            runCatching { out?.run { flush(); close() } }
            out = null; part++; written = 0
        }
        val o = out ?: runCatching {
            File(dir, if (part == 1) "walk-$s.jsonl" else "walk-$s.$part.jsonl").also { current = it }.bufferedWriter()
        }.getOrNull()?.also { out = it } ?: return
        runCatching {
            o.write(json); o.newLine()
            written += json.length + 1
            if (++lines % 20 == 0) o.flush()
        }.onFailure { stop() }
    }

    /** Keeps [jpeg] as walk-<session>-<ms><tag>.jpg; returns its name, or null when off or over the cap. */
    @Synchronized fun frame(jpeg: ByteArray, ms: Long, tag: String = ""): String? {
        val s = session ?: return null
        if (bytes + jpeg.size > maxBytes || dir.usableSpace < 300L * 1024 * 1024) return null
        val f = File(dir, "walk-$s-$ms$tag.jpg")
        if (runCatching { f.writeBytes(jpeg) }.isFailure) return null
        bytes += jpeg.size
        return f.name
    }

    private fun used(): Long = dir.listFiles { f -> f.name.startsWith("walk-") }.orEmpty().sumOf { it.length() }

    /** Files of finished sessions, for upload: all frames, and the JSONL of sessions that are not running. */
    @Synchronized fun uploadable(): List<File> = dir.listFiles { f ->
        f.name.startsWith("walk-") && f.length() > 0 && (f.name.endsWith(".jpg") || (f.name.endsWith(".jsonl") && f.name != current?.name))
    }.orEmpty().sortedBy { it.name }

    companion object {
        const val PART_BYTES = 20L * 1024 * 1024
        fun of(ctx: android.content.Context) = WalkRecorder(File(ctx.filesDir, "log"))
    }
}
