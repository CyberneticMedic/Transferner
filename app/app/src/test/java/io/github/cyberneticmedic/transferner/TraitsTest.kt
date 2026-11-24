package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.Candidate
import io.github.cyberneticmedic.transferner.core.Catalog
import io.github.cyberneticmedic.transferner.core.ChipKind
import io.github.cyberneticmedic.transferner.core.Species
import io.github.cyberneticmedic.transferner.core.Trait
import io.github.cyberneticmedic.transferner.core.TraitSummary
import io.github.cyberneticmedic.transferner.core.WarningIndex
import io.github.cyberneticmedic.transferner.core.cautionHeadline
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * Genus-union caution (audit capture_lookalike_resolution-3): a species WITH its own rows also shows the toxic parts
 * its genus' cited toxic species list and it does not, as "Other <Genus> species: <parts> listed toxic (<severity>)",
 * and that counts in needsCaution. The rows mirror species_traits.csv: Prunus mexicana has only the NC State genus
 * page (fruit, flower conditional; the page states no poison), while P. americana and P. caroliniana list
 * cyanogenic leaf, seed and stem.
 */
class TraitsTest {
    private fun tox(part: String, sev: String) = Trait(part = part, category = "toxic", ingestionToxic = 1, severity = sev)
    private fun cond(part: String, genus: Boolean = false) = Trait(part = part, category = "conditional", genusLevel = genus)
    private fun ed(part: String) = Trait(part = part, category = "edible")

    private val catalog = Catalog(listOf(
        Species(0, "Prunus mexicana", "Mexican plum", "Prunus", "Rosaceae", traits = listOf(cond("fruit", true), cond("flower", true))),
        Species(1, "Prunus americana", "American plum", "Prunus", "Rosaceae",
            traits = listOf(cond("fruit"), tox("leaf", "moderate"), tox("seed", "moderate"), tox("stem", "moderate"))),
        Species(2, "Prunus caroliniana", "cherry laurel", "Prunus", "Rosaceae",
            traits = listOf(tox("leaf", "high"), tox("seed", "high"), tox("stem", "high"))),
        Species(3, "Prunus persica", "peach", "Prunus", "Rosaceae", traits = listOf(cond("fruit"), tox("seed", "moderate"))),
        Species(4, "Prunus umbellata", "flatwoods plum", "Prunus", "Rosaceae"),            // no rows: a relative
        Species(5, "Prunus mexicana", "Mexican plum", "Prunus", "Rosaceae", traits = listOf(cond("fruit", true), cond("flower", true))),  // name twin
        Species(6, "Tradescantia ohiensis", "Ohio spiderwort", "Tradescantia", "Commelinaceae", traits = listOf(ed("leaf"), ed("stem"))),
        Species(7, "Tradescantia virginiana", "Virginia spiderwort", "Tradescantia", "Commelinaceae",
            traits = listOf(cond("flower"), tox("sap", "low"))),
        Species(8, "Ilex glabra", "inkberry", "Ilex", "Aquifoliaceae", traits = listOf(tox("whole", "low"), tox("fruit", "low"))),
        Species(9, "Ilex vomitoria", "yaupon", "Ilex", "Aquifoliaceae", traits = listOf(tox("fruit", "low"))),
        Species(10, "Quercus alba", "white oak", "Quercus", "Fagaceae", traits = listOf(cond("fruit"), tox("leaf", ""))),
        Species(11, "Quercus rubra", "red oak", "Quercus", "Fagaceae", traits = listOf(cond("fruit"))),
    ))
    private val idx = WarningIndex(catalog)
    private fun of(i: Int) = TraitSummary.of(catalog.species[i], idx)
    private fun headline(i: Int) = cautionHeadline(listOf(Candidate(catalog.species[i], 0.5f) to of(i)))

    @Test fun genusPageOnlyPlumGetsItsCongenersCyanogenicParts() {
        val t = of(0)
        val cg = t.congener!!
        assertEquals("Prunus", cg.genus)
        assertEquals(listOf("leaf", "seed", "stem"), cg.parts)
        assertEquals("high", cg.severity)                                   // the highest across the congeners
        assertEquals("Other Prunus species: leaf, seed, stem listed toxic (high)", cg.line)
        // the contributing congeners, not the species itself or its name twin
        assertEquals(setOf("Prunus americana", "Prunus caroliniana", "Prunus persica"), cg.species.map { it.name }.toSet())
        assertTrue(t.needsCaution); assertEquals(ChipKind.CAUTION, t.chipKind)
        assertFalse(t.toxic); assertFalse(t.serious); assertFalse(t.relative)   // never claimed for the species itself
        assertEquals("Fruit, flower edible as described · other Prunus species toxic", t.chip)
        assertTrue(t.detail, t.detail.endsWith("; other Prunus species: leaf, seed, stem listed toxic (high)"))
        assertEquals("Caution: relative of a poisonous species", headline(0))
    }

    @Test fun onlyThePartsItLacks() {
        val peach = of(3)                                                   // lists its own seed
        assertEquals(listOf("leaf", "stem"), peach.congener!!.parts)
        assertEquals("high", peach.congener!!.severity)
        assertEquals("Poisonous if eaten · moderate", peach.chip)           // its own toxicity still leads the chip
        assertNull(of(1).congener)                                          // americana lists every congener part
        assertNull(of(2).congener)
    }

    @Test fun anEdibleOnlySpeciesTurnsToCautionForAMildCongener() {
        assertEquals(ChipKind.EDIBLE, TraitSummary.of(catalog.species[6]).chipKind)     // without the index: as before
        val t = of(6)
        assertEquals("Other Tradescantia species: sap listed toxic (low)", t.congener!!.line)
        assertFalse(t.congener!!.serious)
        assertEquals(ChipKind.CAUTION, t.chipKind)
        assertEquals("Leaf, stem edible · other Tradescantia species toxic", t.chip)
        assertEquals("Caution: relative of a mildly toxic species", headline(6))
    }

    @Test fun wholePlantRowsRelativesAndUnstatedSeverity() {
        assertNull(of(8).congener)                                          // its own "whole" row covers every part
        assertEquals("Other Ilex species: all parts listed toxic (low)", of(9).congener!!.line)   // a congener's "whole"
        val rel = of(4)                                                     // no rows: D-038 relative, not a union
        assertTrue(rel.relative); assertNull(rel.congener); assertTrue(rel.needsCaution)
        val oak = of(11)                                                    // white oak's leaf row states no severity
        assertEquals("Other Quercus species: leaf listed toxic (severity not stated)", oak.congener!!.line)
        assertTrue(oak.congener!!.serious)
        assertEquals("Caution: relative of a poisonous species", headline(11))
    }

    @Test fun withoutTheIndexNothingChanges() {
        for (s in catalog.species) assertNull(TraitSummary.of(s).congener)
        assertEquals("Edible only with conditions", TraitSummary.of(catalog.species[0]).chip)
        assertEquals("Edible part listed: leaf, stem", TraitSummary.of(catalog.species[6]).chip)
    }
}
