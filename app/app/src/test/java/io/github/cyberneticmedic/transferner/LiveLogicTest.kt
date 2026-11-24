package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.AlertAccumulator
import io.github.cyberneticmedic.transferner.core.Box
import io.github.cyberneticmedic.transferner.core.BoxRules
import io.github.cyberneticmedic.transferner.core.BoxTracker
import io.github.cyberneticmedic.transferner.core.ForageGenus
import io.github.cyberneticmedic.transferner.core.ForageMeta
import io.github.cyberneticmedic.transferner.core.ForageScorer
import io.github.cyberneticmedic.transferner.core.HoldDetector
import io.github.cyberneticmedic.transferner.core.Tier
import io.github.cyberneticmedic.transferner.core.TimeEma
import io.github.cyberneticmedic.transferner.core.Tuning
import io.github.cyberneticmedic.transferner.core.topMass
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.exp

/** D-044 live logic: box organisation and tracking, smoothing, forage scoring and alert hysteresis. */
class LiveLogicTest {
    private val t = Tuning()

    @Test fun secondaryBoxesNeedTheHigherThreshold() {
        val c = listOf(Box(10f, 10f, 200f, 200f, 0.6f), Box(250f, 250f, 400f, 400f, 0.35f), Box(300f, 20f, 440f, 150f, 0.45f))
        val kept = BoxRules.organize(c, 0.25f, t, 448)
        assertEquals(listOf(0.6f, 0.45f), kept.map { it.score })        // 0.35 < max(0.25 + 0.15, 0.4)
        assertEquals(3, BoxRules.organize(c, 0.25f, t.copy(secondaryFloor = 0.3f, secondaryOffset = 0f), 448).size)
    }

    @Test fun nestedBoxesAreDroppedUnlessTheContainerFillsTheFrame() {
        val outer = Box(50f, 50f, 300f, 300f, 0.9f)
        val inner = Box(100f, 100f, 200f, 200f, 0.8f)                   // entirely inside
        assertEquals(1, BoxRules.organize(listOf(outer, inner), 0.25f, t, 448).size)
        val full = Box(0f, 0f, 448f, 440f, 0.95f)                       // fills the frame: keep sub-frame boxes
        assertEquals(2, BoxRules.organize(listOf(full, inner), 0.25f, t, 448).size)
    }

    @Test fun trackerConfirmsAfterTwoHitsAndCoastsThroughMisses() {
        val tr = BoxTracker()
        val b = Box(10f, 10f, 100f, 100f, 0.7f)
        assertTrue(tr.update(listOf(b)).isEmpty())                       // first sighting: not shown yet
        assertEquals(1, tr.update(listOf(b.copy(x0 = 12f))).size)       // second: shown, smoothed
        assertEquals(1, tr.update(emptyList()).size)                     // one miss: still shown
        assertEquals(1, tr.update(emptyList()).size)                     // two misses
        assertTrue(tr.update(emptyList()).isEmpty())                     // three: gone
    }

    @Test fun trackerSmoothsBoxes() {
        val tr = BoxTracker()
        tr.update(listOf(Box(0f, 0f, 100f, 100f, 0.5f)))
        val s = tr.update(listOf(Box(20f, 0f, 120f, 100f, 0.5f))).single()
        assertEquals(10f, s.x0, 1e-4f)                                    // EMA 0.5 between 0 and 20
    }

    @Test fun timeEmaWeightDependsOnElapsedTime() {
        val e = TimeEma(0.56f)
        e.update(floatArrayOf(1f, 0f), 0L)
        val a = e.update(floatArrayOf(0f, 1f), 125L)                     // 8 fps: alpha 1 - exp(-0.125/0.56) = 0.20
        assertEquals(1 - exp(-0.125f / 0.56f), a[1], 1e-5f)
        val e2 = TimeEma(0.56f)
        e2.update(floatArrayOf(1f, 0f), 0L)
        assertTrue(e2.update(floatArrayOf(0f, 1f), 500L)[1] > a[1])      // a slower frame moves further
    }

    @Test fun topMassIsPlantPresence() {
        assertEquals(0.95f, topMass(floatArrayOf(0.5f, 0.2f, 0.1f, 0.05f, 0.05f, 0.1f), 5), 1e-5f)
    }

    private val forage = ForageMeta(listOf(
        ForageGenus(1, "Sambucus", "alert", 0.9f, "elderberry"),
        ForageGenus(2, "Plantago", "show", 1.0f, "plantain"),
    ))

    @Test fun forageScoreWeightsByTierAndEdibleShare() {
        val p = floatArrayOf(0.3f, 0.5f, 0.2f)
        val s = ForageScorer(forage, 3, emptyMap()).score(p)
        assertEquals(0.45f, s.alert, 1e-6f); assertEquals(1, s.alertGenus)
        assertEquals(0.2f, s.show, 1e-6f); assertEquals(2, s.showGenus)
        val off = ForageScorer(forage, 3, mapOf("Sambucus" to Tier.OFF, "Plantago" to Tier.ALERT)).score(p)
        assertEquals(0.2f, off.alert, 1e-6f); assertEquals(2, off.alertGenus)
    }

    @Test fun alertTierGeneraGetTheWeightFloor() {
        val low = ForageMeta(listOf(ForageGenus(0, "Pyrus", "alert", 0.08f, "pear"), ForageGenus(1, "Plantago", "show", 0.2f, "plantain")))
        val s = ForageScorer(low, 2, emptyMap(), alertFloor = 0.8f).score(floatArrayOf(0.9f, 0.1f))
        assertEquals(0.72f, s.alert, 1e-6f)                                // 0.9 x max(0.08, 0.8): can reach 0.5
        assertEquals(0.02f, s.show, 1e-6f)                                 // the floor is for the alert tier only
    }

    @Test fun alertNeedsTheScoreHeldThenRespectsTheCheckOutcome() {
        val acc = AlertAccumulator(t.copy(alertTauS = 0f))              // no smoothing: test the hysteresis alone
        assertEquals(-1, acc.update(0.6f, 1, 0))                          // above enter, but not held yet
        assertEquals(-1, acc.update(0.6f, 1, 500))
        assertEquals(1, acc.update(0.6f, 1, 800))                         // held >= 0.75 s: a candidate
        assertEquals(-1, acc.update(0.6f, 1, 900))                        // active: no repeat
        acc.confirmed(1, 900)
        acc.update(0.2f, 1, 1000)                                         // clears below exit
        acc.update(0.6f, 1, 2000)
        assertEquals(-1, acc.update(0.6f, 1, 3000))                       // confirmed genus: quiet for 10 min
        acc.update(0.2f, 1, 3100)
        acc.update(0.6f, 7, 4000)
        assertEquals(7, acc.update(0.6f, 7, 4800))                        // another genus may alert
        acc.rejected(7, 4800)
        acc.update(0.2f, 7, 4900); acc.update(0.6f, 7, 5000)
        assertEquals(-1, acc.update(0.6f, 7, 6000))                       // rejected: quiet for 30 s
        acc.update(0.2f, 7, 6100); acc.update(0.6f, 7, 40_000)
        assertEquals(7, acc.update(0.6f, 7, 41_000))                      // ...then it may try again
    }

    @Test fun anUncheckedCandidateIsReleasedAtOnce() {
        val acc = AlertAccumulator(t.copy(alertTauS = 0f))
        acc.update(0.6f, 3, 0)
        assertEquals(3, acc.update(0.6f, 3, 800))
        acc.released(3)                                                   // BioCLIP was busy or not loaded
        acc.update(0.6f, 3, 900)
        assertEquals(3, acc.update(0.6f, 3, 1700))                        // starts again after the hold, no lockout
    }

    @Test fun aSustainedScoreAlertsWithinAboutTwoSeconds() {
        val acc = AlertAccumulator(t)
        val fired = (0 until 12).map { acc.update(0.9f, 1, it * 250L) }.indexOfFirst { it == 1 }
        assertTrue("fired at frame $fired", fired in 4..8)               // 1.0-2.0 s at 4 fps
    }

    @Test fun briefSpikesDoNotAlert() {
        val acc = AlertAccumulator(t)                                     // tau 1 s
        var fired = false
        for (i in 0 until 3) fired = fired || acc.update(0.9f, 1, i * 250L) >= 0   // 0.75 s of a strong score
        for (i in 3 until 12) fired = fired || acc.update(0.0f, 1, i * 250L) >= 0
        assertFalse(fired)
    }

    @Test fun toxicHoldNeedsASustainedLevel() {
        val h = HoldDetector(0.3f, 1.0f, 0f, 120f)
        assertFalse(h.update(0.5f, 0)); assertFalse(h.update(0.5f, 600))
        assertTrue(h.update(0.5f, 1100))
        assertFalse(h.update(0.5f, 1500))                                 // refractory
    }
}
