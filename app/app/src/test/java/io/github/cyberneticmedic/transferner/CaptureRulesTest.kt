package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.CaptureRules
import io.github.cyberneticmedic.transferner.core.Catalog
import io.github.cyberneticmedic.transferner.core.Level
import io.github.cyberneticmedic.transferner.core.NameAliases
import io.github.cyberneticmedic.transferner.core.Species
import io.github.cyberneticmedic.transferner.core.TextIndex
import io.github.cyberneticmedic.transferner.core.Trait
import io.github.cyberneticmedic.transferner.core.Tuning
import io.github.cyberneticmedic.transferner.core.WarningIndex
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** D-044 capture rules: name twins, result levels, the summed caution banner, second photo and fusion. */
class CaptureRulesTest {
    private val edible = Trait(part = "fruit", category = "edible")
    private val toxic = Trait(part = "berries", category = "toxic", ingestionToxic = 1, severity = "high")
    private val catalog = Catalog(listOf(
        Species(0, "Sambucus canadensis", "elderberry", "Sambucus", "Adoxaceae", traits = listOf(edible)),
        Species(1, "Sambucus racemosa", "red elderberry", "Sambucus", "Adoxaceae", traits = listOf(toxic)),
        Species(2, "Cicuta maculata", "water hemlock", "Cicuta", "Apiaceae", traits = listOf(toxic)),
        Species(3, "Erodium cicutarium", "storksbill", "Erodium", "Geraniaceae"),
        Species(4, "Erodium cicutarium", "storksbill", "Erodium", "Geraniaceae"),     // an authorship twin
        Species(5, "Quercus alba", "white oak", "Quercus", "Fagaceae"),
    ))
    private val rules = CaptureRules(catalog, WarningIndex(catalog), Tuning())

    private fun p(vararg v: Float) = v

    @Test fun twinsAreSummedOntoTheFirstIndex() {
        val a = NameAliases(catalog)
        assertEquals(1, a.twins)
        val m = a.merge(p(0f, 0f, 0f, 0.36f, 0.6f, 0.04f))
        assertEquals(0.96f, m[3], 1e-6f); assertEquals(0f, m[4], 0f)
    }

    @Test fun levelsFollowSpeciesThenGenusThenUncertain() {
        assertEquals(Level.SPECIES, rules.verdict(p(0.85f, 0.05f, 0f, 0f, 0f, 0.1f), null).level)
        val g = rules.verdict(p(0.5f, 0.4f, 0f, 0f, 0f, 0.1f), null)
        assertEquals(Level.GENUS, g.level); assertEquals("Sambucus", g.genus); assertEquals(0.9f, g.genusP, 1e-6f)
        assertEquals(Level.UNCERTAIN, rules.verdict(p(0.4f, 0.1f, 0.2f, 0f, 0f, 0.3f), null).level)
    }

    @Test fun aLowTextScoreDropsOneLevel() {
        val sp = rules.verdict(p(0.85f, 0.1f, 0f, 0f, 0f, 0.05f), 0.50f)
        assertEquals(Level.GENUS, sp.level); assertTrue(sp.demoted); assertFalse(sp.maybeOutside)
        val out = rules.verdict(p(0.5f, 0.4f, 0f, 0f, 0f, 0.1f), 0.50f)
        assertEquals(Level.UNCERTAIN, out.level); assertTrue(out.maybeOutside)
        assertEquals(Level.SPECIES, rules.verdict(p(0.85f, 0.1f, 0f, 0f, 0f, 0.05f), 0.60f).level)
    }

    @Test fun bannerUsesTheSummedToxicProbability() {
        val quiet = rules.analyze(p(0f, 0.0005f, 0.0005f, 0f, 0f, 0.999f), null)
        assertFalse(quiet.caution.show)                                   // 0.001 < 0.002, though toxic species are in the top 5
        val low = rules.analyze(p(0f, 0.0015f, 0.0015f, 0f, 0f, 0.997f), null)
        assertTrue(low.caution.show); assertFalse(low.caution.strong)     // 0.003
        val strong = rules.analyze(p(0f, 0.05f, 0.05f, 0f, 0f, 0.9f), null)
        assertTrue(strong.caution.strong)
    }

    @Test fun edibleToxicConflictAsksForASecondPhoto() {
        assertTrue(rules.analyze(p(0.6f, 0.3f, 0.05f, 0f, 0f, 0.05f), null).secondPhoto)
        assertFalse(rules.analyze(p(0.95f, 0.03f, 0.01f, 0f, 0f, 0.01f), null).secondPhoto)
        assertFalse(rules.analyze(p(0.6f, 0f, 0f, 0f, 0f, 0.4f), null).secondPhoto)    // no toxic candidate
    }

    @Test fun twoPhotoBannerIsTheUnionOfAllThree() {
        val first = rules.analyze(p(0.6f, 0f, 0.3f, 0f, 0f, 0.1f), null).caution          // Cicuta 0.30: strong
        val weak = rules.analyze(p(0f, 0.003f, 0f, 0f, 0f, 0.997f), null).caution          // red elderberry 0.003
        val none = rules.analyze(p(0f, 0f, 0f, 0f, 0f, 1f), null).caution
        val u = io.github.cyberneticmedic.transferner.core.Caution.union(listOf(weak, first, none))
        assertTrue(u.show); assertTrue(u.strong)
        assertTrue(u.inTop.any { it.first.species.name == "Cicuta maculata" })
        assertEquals(1, u.inTop.count { it.first.species.name == "Sambucus racemosa" })   // de-duplicated
        assertFalse(io.github.cyberneticmedic.transferner.core.Caution.union(listOf(none, none)).show)
    }

    /** Genus union (D-045): edible elderberry's own rows list no toxic part, but red elderberry lists its berries,
     *  so a confident elderberry photo still raises the banner, headed as a relative of a poisonous species. */
    @Test fun edibleSpeciesWithToxicCongenersRaiseTheBanner() {
        val c = rules.analyze(p(0.999f, 0f, 0f, 0f, 0f, 0.001f), null).caution
        assertTrue(c.show && c.strong)
        val t = c.inTop.single { it.first.species.name == "Sambucus canadensis" }.second
        assertEquals(listOf("berries"), t.congener?.parts)
        assertEquals("Caution: relative of a poisonous species",
            io.github.cyberneticmedic.transferner.core.cautionHeadline(c.inTop.filter { it.first.species.name == "Sambucus canadensis" }))
    }

    @Test fun carrotFamilySpeciesWithoutRowsAreHemlockRelatives() {
        val c = Catalog(listOf(
            Species(0, "Conium maculatum", "poison hemlock", "Conium", "Apiaceae", traits = listOf(toxic)),
            Species(1, "Ptilimnium capillaceum", "mock bishop's weed", "Ptilimnium", "Apiaceae"),
            Species(2, "Quercus alba", "white oak", "Quercus", "Fagaceae"),
        ))
        val idx = WarningIndex(c)
        val t = io.github.cyberneticmedic.transferner.core.TraitSummary.of(c.species[1], idx)
        assertTrue(t.relative && t.relativeSerious); assertEquals("family", t.relativeLevel)
        assertTrue(t.detail.contains("same family as poison hemlock"))
        assertEquals(2, idx.warnLevel("Ptilimnium", "Apiaceae"))
        assertEquals(0, idx.warnLevel("Quercus", "Fagaceae"))
        assertFalse(io.github.cyberneticmedic.transferner.core.TraitSummary.of(c.species[2], idx).relative)
    }

    @Test fun fusionIsTheNormalisedGeometricMean() {
        val f = CaptureRules.fuse(p(0.8f, 0.2f), p(0.2f, 0.8f))
        assertEquals(0.5f, f[0], 1e-6f)
        val g = CaptureRules.fuse(p(0.9f, 0.1f), p(0.6f, 0.4f))
        assertTrue(g[0] > 0.6f && g[0] < 0.9f)
    }

    @Test fun textScoreIsTheBestCosine() {
        val t = floatArrayOf(1f, 0f, 0f, 1f)                              // two unit name embeddings, dim 2
        val bb = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN)
        for (v in t) bb.putShort(if (v == 1f) 0x3C00.toShort() else 0)
        val idx = TextIndex(bb.array(), 2, 2)
        assertEquals(0.8f, idx.score(floatArrayOf(3f, 4f)), 1e-3f)       // normalised: (0.6, 0.8)
    }
}
