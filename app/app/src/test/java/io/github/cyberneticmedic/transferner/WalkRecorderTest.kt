package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.data.WalkRecorder
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.file.Files

/** D-044 walk recordings: no empty files (the server refuses them), the open file is never uploaded. */
class WalkRecorderTest {
    @Test fun aSessionWithoutLinesLeavesNoFile() {
        val dir = Files.createTempDirectory("walk").toFile()
        val r = WalkRecorder(dir)
        r.start(); r.stop()
        assertTrue(r.uploadable().isEmpty())
        assertTrue(dir.listFiles().orEmpty().isEmpty())
    }

    @Test fun theRunningSessionsLogIsHeldBackUntilItEnds() {
        val dir = Files.createTempDirectory("walk").toFile()
        val r = WalkRecorder(dir)
        r.start()
        r.line("{\"t\":1}")
        r.frame(ByteArray(10), 5L)
        assertEquals(listOf("jpg"), r.uploadable().map { it.extension })
        r.stop()
        assertEquals(setOf("jpg", "jsonl"), r.uploadable().map { it.extension }.toSet())
    }
}
