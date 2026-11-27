package io.github.cyberneticmedic.transferner

import com.sun.net.httpserver.HttpServer
import io.github.cyberneticmedic.transferner.core.Candidate
import io.github.cyberneticmedic.transferner.core.Catalog
import io.github.cyberneticmedic.transferner.core.Species
import io.github.cyberneticmedic.transferner.data.ObservationLog
import io.github.cyberneticmedic.transferner.data.WalkRecorder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.io.File
import java.io.IOException
import java.net.InetSocketAddress
import java.nio.file.Files
import java.text.SimpleDateFormat
import java.util.Collections
import java.util.Locale
import java.util.TimeZone

/**
 * The field log end to end on the JVM (D-014/D-016/D-036/D-044): add() writes the photo, the optional second
 * photo and sidecar, and a CSV row; entries() reads them back for the Log screen; an older log keeps its rows under
 * the current header; upload() sends the CSV first, each file once (uploaded.txt), skips what the server refuses
 * for its size (413) and ends with the device report. [FakeServer] applies scripts/bundle_server.py's rules.
 */
class ObservationLogTest {
    private val catalog = Catalog(listOf(
        Species(0, "Ficus carica", "fig", "Ficus", "Moraceae"),
        Species(1, "Morus alba", "white mulberry", "Morus", "Moraceae"),
        Species(2, "Cicuta maculata", "water hemlock", "Cicuta", "Apiaceae"),
        Species(3, "Daucus carota", "wild carrot", "Daucus", "Apiaceae"),
        Species(4, "Quercus alba", "white oak", "Quercus", "Fagaceae"),
        Species(5, "Acer rubrum", "red maple", "Acer", "Sapindaceae"),
    ))
    private val top5 = (1..5).joinToString(",") { "species_$it,class_$it,p_$it" }
    private val header04 = "time_utc,latitude,longitude,photo,$top5,caution,bundle,species_model,detector_model,user_note"
    private val header = "$header04,id_ms,text_score,level,photos"
    private val device = "moto-g-power-2025-7c5f3b"
    private val report = "{\"app\":\"0.7.1\",\"version_code\":12,\"model\":\"moto g power - 2025\"}"

    private fun c(i: Int, p: Float) = Candidate(catalog.species[i], p)
    private fun tmp(): File = Files.createTempDirectory("obslog").toFile()
    private fun jpeg(tag: Int) = ByteArray(2_000 + tag) { (it * 31 + tag).toByte() }
    private val sidecar = { photo: String -> "{\"photo\":\"$photo\",\"level\":\"genus\"}" }

    /** add() names files by the second; wait for the next one so each capture gets its own names. */
    private fun nextSecond() = Thread.sleep(1_020L - System.currentTimeMillis() % 1_000L)

    private fun obsNames(dir: File) = dir.list().orEmpty().filter { it.startsWith("obs-") }.sorted()
    private fun stampOf(photo: String) = photo.removePrefix("obs-").removeSuffix(".jpg")

    /** The CSV as the app writes it: every cell quoted, "" inside a cell for a quote. */
    private fun cells(line: String): List<String> {
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

    private fun row(vararg v: String) = v.joinToString(",") { "\"" + it.replace("\"", "\"\"") + "\"" }

    @Test fun addWritesPhotosSidecarsAndRowsThatEntriesReadBack() {
        val dir = File(tmp(), "log")                                 // add() must create the directory
        val log = ObservationLog(dir)
        assertTrue(log.entries(catalog).isEmpty())

        log.add(jpeg(1), listOf(c(0, 0.91f), c(1, 0.05f), c(2, 0.02f), c(3, 0.01f), c(4, 0.005f)), true, "0.8",
            "bioclip_visual_w8dyn", "mnv4_live_d046", note = "by the gate\nleft side ")
        nextSecond()
        log.add(jpeg(2), listOf(c(2, 0.62f), c(3, 0.30f)), true, "0.8", "bioclip_visual_w8dyn", "mnv4_live_d046",
            idMs = 1234.4f, text = 0.5f, level = "genus", second = jpeg(3), sidecar = sidecar)
        nextSecond()
        log.add(jpeg(4), listOf(c(4, 0.99f)), false, "0.8", "bioclip_visual_w8dyn", "mnv4_live_d046",
            note = "said \"oak\", maybe", idMs = 800f, level = "species", sidecar = sidecar)

        // files: three first photos, one second photo, two sidecars
        val names = obsNames(dir)
        val photos = names.filter { Regex("obs-\\d{8}-\\d{6}\\.jpg").matches(it) }
        assertEquals("three captures, three first photos: $names", 3, photos.size)
        val s = photos.map(::stampOf)
        assertEquals(listOf("obs-${s[1]}-2.jpg"), names.filter { it.endsWith("-2.jpg") })
        assertEquals(listOf("obs-${s[1]}.json", "obs-${s[2]}.json"), names.filter { it.endsWith(".json") })
        assertEquals(6, names.size)
        assertArrayEquals(jpeg(1), File(dir, photos[0]).readBytes())
        assertArrayEquals(jpeg(3), File(dir, "obs-${s[1]}-2.jpg").readBytes())
        assertEquals(sidecar("obs-${s[1]}.jpg"), File(dir, "obs-${s[1]}.json").readText())

        // CSV: the current header, one 28-cell row per capture, the 0.6+ columns filled
        val lines = log.csv.readLines()
        assertEquals(header, lines[0])
        assertEquals(4, lines.size)
        val rows = lines.drop(1).map(::cells)
        rows.forEach { assertEquals("cells in $it", 28, it.size) }
        assertEquals(photos, rows.map { it[3] })
        assertEquals("by the gate left side", rows[0][23])
        assertEquals(listOf("0", "", "", "1"), rows[0].takeLast(4))
        assertEquals(listOf("1234", "0.5000", "genus", "2"), rows[1].takeLast(4))
        assertEquals(listOf("800", "", "species", "1"), rows[2].takeLast(4))
        assertEquals(listOf("1", "0.8", "bioclip_visual_w8dyn", "mnv4_live_d046"), rows[1].subList(19, 23))
        assertEquals(listOf("Cicuta maculata", "2", "0.6200", "Daucus carota", "3", "0.3000", "", "", ""), rows[1].subList(4, 13))

        // entries(): newest first, photo, top, caution and note as saved
        val e = log.entries(catalog)
        assertEquals(photos.reversed(), e.map { it.photo.name })
        e.forEach { assertTrue("${it.photo} exists", it.photo.isFile) }
        assertEquals(listOf("Quercus alba"), e[0].top.map { it.species.name })
        assertEquals(0.99f, e[0].top[0].p, 1e-4f)
        assertFalse(e[0].caution)
        assertEquals("said \"oak\", maybe", e[0].note)
        assertEquals(listOf("Cicuta maculata", "Daucus carota"), e[1].top.map { it.species.name })
        assertTrue(e[1].caution)
        assertEquals("", e[1].note)
        assertEquals(listOf(0, 1, 2, 3, 4), e[2].top.map { it.species.i })
        listOf(0.91f, 0.05f, 0.02f, 0.01f, 0.005f).zip(e[2].top).forEach { (p, cand) -> assertEquals(p, cand.p, 1e-4f) }
        assertTrue(e[2].caution)
        assertArrayEquals(jpeg(1), e[2].photo.readBytes())

        // the UTC time in the row is the same instant as the local-time stamp in the file name
        val utc = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.US).apply { timeZone = TimeZone.getTimeZone("UTC") }
        val local = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US)
        e.forEach { assertEquals(local.parse(stampOf(it.photo.name)), utc.parse(it.time)) }
    }

    @Test fun anOlderLogKeepsItsRowsUnderTheCurrentHeader() {
        val dir = tmp()
        // the phone's log as 0.4.1 left it: the 0.4 header, 0.2 rows (no user_note) and 0.4 rows
        val r02 = row("2026-11-24T03:27:36Z", "", "", "obs-20261123-222736.jpg", "Ficus carica", "0", "0.9999",
            "Morus alba", "1", "0.0001", "Quercus alba", "4", "0.0000", "Acer rubrum", "5", "0.0000", "Daucus carota", "3", "0.0000",
            "1", "0.2", "bioclip_visual_w8dyn", "detector.tflite")
        val r04 = row("2026-11-24T07:06:44Z", "", "", "obs-20261124-020644.jpg", "Cicuta maculata", "2", "0.7000",
            "Daucus carota", "3", "0.2000", "", "", "", "", "", "", "", "", "", "1", "0.4", "bioclip_visual_w8dyn", "mnv4_live", "two trunks")
        assertEquals(23, cells(r02).size)
        assertEquals(24, cells(r04).size)
        val csv = File(dir, "observations.csv")
        csv.writeText("$header04\n$r02\n$r04\n")
        val log = ObservationLog(dir)
        assertEquals("an old log is readable before anything is added", 2, log.entries(catalog).size)

        log.add(jpeg(1), listOf(c(4, 0.8f)), false, "0.8", "bioclip_visual_w8dyn", "mnv4_live_d046", note = "new", sidecar = sidecar)
        val lines = csv.readLines()
        assertEquals(listOf(header, r02, r04), lines.take(3))
        assertEquals(4, lines.size)

        val e = log.entries(catalog)
        assertEquals(listOf("new", "two trunks", ""), e.map { it.note })
        assertEquals(listOf("obs-20261124-020644.jpg", "obs-20261123-222736.jpg"), e.drop(1).map { it.photo.name })
        assertEquals(listOf("Cicuta maculata", "Daucus carota"), e[1].top.map { it.species.name })
        assertEquals(5, e[2].top.size)
        assertTrue(e[1].caution && e[2].caution)

        // once migrated, a later add() only appends
        nextSecond()
        log.add(jpeg(2), listOf(c(5, 0.6f)), false, "0.8", "bioclip_visual_w8dyn", "mnv4_live_d046")
        val again = csv.readLines()
        assertEquals(lines, again.take(4))
        assertEquals(5, again.size)
        assertEquals(4, log.entries(catalog).size)
    }

    @Test fun uploadSendsTheCsvFirstEachFileOnceSkips413AndEndsWithTheReport() {
        val dir = tmp()
        val log = ObservationLog(dir)
        log.add(jpeg(1), listOf(c(0, 0.9f)), false, "0.8", "v", "d", second = jpeg(2), sidecar = sidecar)
        nextSecond()
        log.add(jpeg(3), listOf(c(1, 0.8f)), false, "0.8", "v", "d")
        nextSecond()
        val walk = WalkRecorder(dir)
        walk.start()
        walk.line("{\"t\":1}")
        assertNotNull(walk.frame(jpeg(5), 1_000L))
        assertNotNull(walk.frame(jpeg(6), 2_000L, "-alert"))
        walk.stop()
        val empty = "obs-20261124-000000.jpg"                           // a photo whose write failed: never posted
        File(dir, empty).writeBytes(ByteArray(0))

        val obs = obsNames(dir)
        val walkFiles = walk.uploadable()
        assertEquals(3, walkFiles.size)
        val expected = obs + walkFiles.map { it.name }
        val photos = obs.filter { Regex("obs-\\d{8}-\\d{6}\\.jpg").matches(it) && it != empty }
        val refused = photos[1]                                         // capture 2: the server answers 413

        FakeServer().use { srv ->
            srv.refuse += refused
            val progress = mutableListOf<Pair<Int, Int>>()
            val n = log.upload(srv.url, device, report, walkFiles) { i, t -> progress += i to t }
            assertEquals(expected.size, n)
            assertEquals((1..n).map { it to n }, progress)

            val posts = srv.posts.toList()
            assertTrue("every post went to the device's folder", posts.all { it.device == device })
            assertTrue("every post carries its Content-Length", posts.all { it.length == it.body.size.toString() })
            assertEquals("observations.csv", posts.first().name)
            assertEquals("text/csv", posts.first().type)
            assertArrayEquals(log.csv.readBytes(), posts.first().body)
            assertEquals(expected - refused - empty, posts.drop(1).dropLast(1).map { it.name })
            posts.drop(1).dropLast(1).forEach { p ->
                assertArrayEquals(p.name, File(dir, p.name).readBytes(), p.body)
                assertEquals(p.name, if (p.name.endsWith(".jpg")) "image/jpeg" else "application/json", p.type)
            }
            assertTrue(posts.last().name, Regex("device-\\d{8}-\\d{6}\\.json").matches(posts.last().name))
            assertEquals(report, String(posts.last().body))
            assertEquals(listOf(413), srv.tries.filter { it.first == refused }.map { it.second })
            assertFalse("an empty file is never posted", srv.tries.any { it.first == empty })
            assertEquals(expected, File(dir, "uploaded.txt").readLines())

            // Send again with nothing new: the CSV and a report only, and the refused photo is not retried
            srv.clear()
            assertEquals(0, log.upload(srv.url + "/", device, report, walk.uploadable()) { _, _ -> })
            assertEquals(listOf("observations.csv"), srv.posts.map { it.name }.dropLast(1))
            assertTrue(Regex("device-\\d{8}-\\d{6}\\.json").matches(srv.posts.last().name))
            assertFalse(srv.tries.any { it.first == refused })

            // a new capture: only its photo and sidecar go, after the CSV
            nextSecond()
            log.add(jpeg(7), listOf(c(4, 0.7f)), false, "0.8", "v", "d", sidecar = sidecar)
            val fresh = obsNames(dir) - obs.toSet()
            assertEquals(2, fresh.size)
            srv.clear()
            assertEquals(2, log.upload(srv.url, device, report, walk.uploadable()) { _, _ -> })
            assertEquals(listOf("observations.csv") + fresh.sorted(), srv.posts.map { it.name }.dropLast(1))
            assertEquals(expected + fresh.sorted(), File(dir, "uploaded.txt").readLines())
        }
    }

    @Test fun aServerErrorStopsTheUploadAndTheFileIsSentNextTime() {
        val dir = tmp()
        val log = ObservationLog(dir)
        log.add(jpeg(1), listOf(c(0, 0.9f)), false, "0.8", "v", "d", sidecar = sidecar)
        nextSecond()
        log.add(jpeg(2), listOf(c(1, 0.8f)), false, "0.8", "v", "d", sidecar = sidecar)
        val obs = obsNames(dir)                                         // [jpg1, json1, jpg2, json2]
        assertEquals(4, obs.size)
        FakeServer().use { srv ->
            srv.broken += obs[2]
            try {
                log.upload(srv.url, device, report) { _, _ -> }
                fail("a 500 must stop the upload")
            } catch (e: IOException) {
                assertTrue(e.message, e.message!!.startsWith("HTTP 500"))
            }
            assertEquals(obs.take(2), File(dir, "uploaded.txt").readLines())
            srv.broken.clear()
            srv.clear()
            assertEquals(2, log.upload(srv.url, device, report) { _, _ -> })
            assertEquals(listOf("observations.csv") + obs.drop(2), srv.posts.map { it.name }.dropLast(1))
            assertEquals(obs, File(dir, "uploaded.txt").readLines())
        }
    }

    /** scripts/bundle_server.py's POST rules: /upload/<device>/<file>, names [A-Za-z0-9._-], 1 byte to 25 MB, 201. */
    private class FakeServer : AutoCloseable {
        class Post(val device: String, val name: String, val type: String?, val length: String?, val body: ByteArray)

        val posts: MutableList<Post> = Collections.synchronizedList(mutableListOf())
        val tries: MutableList<Pair<String, Int>> = Collections.synchronizedList(mutableListOf())   // file name to status
        val refuse: MutableSet<String> = Collections.synchronizedSet(mutableSetOf())            // answered 413 (as if over 25 MB)
        val broken: MutableSet<String> = Collections.synchronizedSet(mutableSetOf())            // answered 500
        private val http = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        val url get() = "http://127.0.0.1:${http.address.port}"

        init {
            http.createContext("/") { ex ->
                val body = ex.requestBody.readBytes()
                val parts = ex.requestURI.path.trim('/').split('/')
                val code = when {
                    ex.requestMethod != "POST" -> 501
                    parts.size != 3 || parts[0] != "upload" || !parts.drop(1).all(NAME::matches) -> 400
                    body.isEmpty() || body.size > MAX_BYTES || parts[2] in refuse -> 413
                    parts[2] in broken -> 500
                    else -> 201
                }
                tries += parts.last() to code
                if (code == 201) posts += Post(parts[1], parts[2], ex.requestHeaders.getFirst("Content-Type"),
                    ex.requestHeaders.getFirst("Content-Length"), body)
                ex.sendResponseHeaders(code, -1)
                ex.close()
            }
            http.start()
        }

        fun clear() { posts.clear(); tries.clear() }
        override fun close() = http.stop(0)

        companion object {
            val NAME = Regex("^[A-Za-z0-9][A-Za-z0-9._-]{0,120}$")
            const val MAX_BYTES = 25 * 1024 * 1024
        }
    }
}
