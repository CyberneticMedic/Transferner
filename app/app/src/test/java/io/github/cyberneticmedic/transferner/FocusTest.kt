package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.Detections
import io.github.cyberneticmedic.transferner.core.FocusCategory
import io.github.cyberneticmedic.transferner.core.FocusMeta
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.ln

/** The focus chip's soft prior (D-037): in-category genera untouched, others down-weighted, never removed. */
class FocusTest {
    private val meta = FocusMeta(0.1f, listOf(FocusCategory("vine", "Vine", listOf(0, 2)), FocusCategory("tree", "Tree", listOf(1))))

    @Test fun biasIsZeroInsideAndLnEpsOutside() {
        val b = meta.bias("vine", 4)!!
        assertEquals(0f, b[0]); assertEquals(0f, b[2])
        assertEquals(ln(0.1f), b[1], 1e-6f); assertEquals(ln(0.1f), b[3], 1e-6f)
        assertNull(meta.bias("moss", 4))
    }

    @Test fun strongOutOfCategoryEvidenceStillWins() {
        // the camera is sure of genus 1 (a tree), the user picked "vine": soft prior keeps genus 1 on top
        val logits = floatArrayOf(0f, 5f, 0f, 0f)
        val b = meta.bias("vine", 4)!!
        val p = Detections.softmax(FloatArray(4) { logits[it] + b[it] })
        assertEquals(1, Detections.topK(p, 1)[0])
        // ...and an ambiguous frame tips toward the chosen form
        val q = Detections.softmax(FloatArray(4) { floatArrayOf(1f, 1.5f, 0f, 0f)[it] + b[it] })
        assertTrue(q[0] > q[1])
    }
}
