package io.github.cyberneticmedic.transferner.core

/**
 * What the cited trait rows say about one species (D-004/D-005/D-027), per part and with the source's
 * severity. Nothing here is inferred: a species with no rows has no data, and absence of a toxic row never
 * means safe. Wording follows the source's emphasis (app 0.3 feedback): a fig, whose sap irritates skin
 * (low) and whose fruit the source calls edible, must not read like dwarf larkspur (poisonous, high).
 */
data class TraitSummary(
    val hasData: Boolean,
    val ingestionToxic: Boolean,
    val contactToxic: Boolean,
    val severity: String,                       // highest listed severity of the toxic rows: low/moderate/high, or ""
    val toxicParts: List<String>,
    val edibleParts: List<String>,
    val conditionalParts: List<String>,
    val genusLevel: Boolean,                    // some rows come from a genus page, not this species
    val relatives: List<Pair<Species, TraitSummary>> = emptyList(),   // no rows of its own; cited toxic congeners
) {
    val toxic get() = ingestionToxic || contactToxic
    val relative get() = !hasData && relatives.isNotEmpty()
    val relativeSerious get() = relative && relatives.any { it.second.serious }
    val needsCaution get() = toxic || conditionalParts.isNotEmpty() || relative

    /** Moderate or high, or a toxic row without a stated severity (treated as serious). */
    val serious get() = toxic && severity != "low"
    private val edibleAny get() = edibleParts.isNotEmpty() || conditionalParts.isNotEmpty()

    private fun parts(ps: List<String>, max: Int = 2): String {
        val named = ps.filter { it != "unspecified" && it != "whole" }
        if (named.isEmpty() || ps.contains("whole")) return "all parts"
        return named.take(max).joinToString(", ") + if (named.size > max) " +${named.size - max}" else ""
    }

    private val edibleDesc: String
        get() {
            val all = (edibleParts + conditionalParts).distinct()
            val named = all.filter { it != "unspecified" }
            val what = if (named.isEmpty()) "Edible" else named.take(2).joinToString(", ").replaceFirstChar { it.uppercase() } + " edible"
            return if (conditionalParts.isNotEmpty()) "$what as described" else what
        }

    private val route: String
        get() = when {
            contactToxic && ingestionToxic -> "touch and eat"
            contactToxic -> "touch"
            else -> "eat"
        }

    /** Short label for a result row. */
    val chip: String
        get() = when {
            relative -> if (relativeSerious) "Relative of a poisonous species" else "Relative of a mildly toxic species"
            !hasData -> "No trait data"
            serious && contactToxic -> "Toxic to $route · ${severity.ifBlank { "severity not stated" }}"
            serious -> "Poisonous if eaten · ${severity.ifBlank { "severity not stated" }}"
            toxic && edibleAny -> "$edibleDesc · some parts mildly toxic"
            contactToxic -> "Mild skin irritant (low)"
            toxic -> "Mildly toxic if eaten (low)"
            conditionalParts.isNotEmpty() -> "Edible only with conditions"
            edibleParts.any { it != "unspecified" } -> "Edible part listed: " + edibleParts.filter { it != "unspecified" }.joinToString(", ")
            edibleParts.isNotEmpty() -> "Edible · part not stated"
            else -> "Listed, no edible or toxic part"
        }

    /** One line for the caution banner: what the source says, part by part. */
    val detail: String
        get() = if (relative) "no trait data of its own; same genus as " + relatives.take(2).joinToString(" and ") { (sp, t) ->
            "${sp.displayName} (${t.chip.lowercase()})" } + if (relatives.size > 2) " and ${relatives.size - 2} more" else ""
        else buildList {
            if (edibleAny) add(edibleDesc.replaceFirstChar { it.lowercase() } + if (conditionalParts.isNotEmpty()) " by the source" else "")
            if (toxic) add("${parts(toxicParts, 3)} listed as toxic to $route (${severity.ifBlank { "severity not stated" }})")
        }.joinToString("; ")

    val chipKind: ChipKind
        get() = when {
            !hasData -> ChipKind.NONE
            needsCaution -> ChipKind.CAUTION
            edibleParts.isNotEmpty() -> ChipKind.EDIBLE
            else -> ChipKind.NONE
        }

    companion object {
        private val order = listOf("low", "moderate", "high")

        /** With [index], a species without rows of its own gets its cited toxic congeners as relatives (D-038). */
        fun of(s: Species, index: WarningIndex? = null): TraitSummary =
            own(s).let { t -> if (!t.hasData && index != null) t.copy(relatives = index.toxicIn(s.genus).filter { it.first.i != s.i }) else t }

        private fun own(s: Species): TraitSummary {
            val t = s.traits
            val toxic = t.filter { it.category == "toxic" }
            // A toxic row's severity; the source's "Medium" is stored as "moderate". Unknown ranks as serious.
            val sevs = toxic.map { it.severity.lowercase() }
            val sev = when {
                sevs.isEmpty() -> ""
                sevs.any { it !in order } -> ""
                else -> sevs.maxBy { order.indexOf(it) }
            }
            return TraitSummary(
                hasData = t.isNotEmpty(),
                ingestionToxic = toxic.any { it.ingestionToxic == 1 },
                contactToxic = toxic.any { it.contactToxic == 1 },
                severity = sev,
                toxicParts = toxic.map { it.part }.distinct(),
                edibleParts = t.filter { it.category == "edible" }.map { it.part }.distinct(),
                conditionalParts = t.filter { it.category == "conditional" }.map { it.part }.distinct(),
                genusLevel = t.any { it.genusLevel },
            )
        }
    }
}

enum class ChipKind { EDIBLE, CAUTION, NONE }

/**
 * Cited toxic species per genus, from the catalog (D-038 warning flags; false positives over false negatives).
 * Live: a genus is flagged when it holds any cited toxic species. Capture: a species without trait rows is a
 * "relative" when its genus holds one.
 */
class WarningIndex(catalog: Catalog) {
    private val byGenus: Map<String, List<Pair<Species, TraitSummary>>> = catalog.species
        .map { it to TraitSummary.of(it) }.filter { it.second.toxic }.groupBy { it.first.genus }
        .mapValues { (_, v) -> v.sortedWith(compareByDescending<Pair<Species, TraitSummary>> { it.second.serious }.thenBy { it.first.name }) }

    fun toxicIn(genus: String): List<Pair<Species, TraitSummary>> = byGenus[genus].orEmpty()
    fun genusToxic(genus: String) = byGenus.containsKey(genus)
    fun genusSerious(genus: String) = byGenus[genus]?.any { it.second.serious } == true

    companion object {
        /** Capture: flag cited-toxic or relative species from this probability anywhere in the 4,411, not only
         *  in the top 5 (regional val misses 0.44% -> 0.17%, false alarms 27% -> 36%; test 0.70% -> 0.18%). */
        const val CAPTURE_MIN_P = 0.005f
        /** Live: flag a toxic genus from this whole-frame probability (regional val misses 5.5%, false alarms 37%). */
        const val LIVE_MIN_P = 0.02f
    }
}

/** D-013 + D-038: a caution sign when any shown candidate is toxic, conditional or a toxic species' relative. */
fun cautionCandidates(top: List<Candidate>, index: WarningIndex? = null): List<Pair<Candidate, TraitSummary>> =
    top.map { it to TraitSummary.of(it.species, index) }.filter { it.second.needsCaution }

/** Banner headline: cited serious toxicity first, then relatives of poisonous species, mild, relatives of mild, conditions. */
fun cautionHeadline(c: List<Pair<Candidate, TraitSummary>>): String = when {
    c.any { it.second.serious } -> "Caution: a poisonous candidate"
    c.any { it.second.relativeSerious } -> "Caution: relative of a poisonous species"
    c.any { it.second.toxic } -> "Caution: some parts listed as mildly toxic"
    c.any { it.second.relative } -> "Caution: relative of a mildly toxic species"
    else -> "Caution: edible only with conditions"
}
