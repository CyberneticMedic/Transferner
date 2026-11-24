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
    val congener: CongenerToxic? = null,        // has rows; toxic parts its cited congeners list and it does not
) {
    val toxic get() = ingestionToxic || contactToxic
    val relative get() = !hasData && relatives.isNotEmpty()
    val relativeSerious get() = relative && relatives.any { it.second.serious }
    val needsCaution get() = toxic || conditionalParts.isNotEmpty() || relative || congener != null

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
            congener != null && edibleAny -> "$edibleDesc · other ${congener.genus} species toxic"
            congener != null -> "Other ${congener.genus} species toxic"
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
            congener?.let { add(it.line.replaceFirstChar { c -> c.lowercase() }) }
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
        private val partOrder = listOf("leaf", "flower", "fruit", "seed", "root", "stem", "bark", "sap", "whole")

        /**
         * With [index], a species without rows of its own gets its cited toxic congeners as relatives (D-038), and a
         * species with rows gets the genus union (audit capture_lookalike_resolution-3): the toxic parts its genus'
         * other cited species list and its own rows do not. Half the dangerous edible/toxic confusions on regional
         * test stay inside a genus, and a congener's thinner rows read as edible-only: Prunus mexicana has only
         * the NC State genus page, which states no poison, while P. americana lists leaf, seed and stem.
         */
        fun of(s: Species, index: WarningIndex? = null): TraitSummary {
            val t = own(s)
            if (index == null) return t
            if (!t.hasData) return t.copy(relatives = index.toxicIn(s.genus).filter { it.first.i != s.i })
            return t.copy(congener = congenerToxic(s, t, index))
        }

        private fun congenerToxic(s: Species, t: TraitSummary, index: WarningIndex): CongenerToxic? {
            if ("whole" in t.toxicParts) return null          // its own rows already list every part
            // other species of the genus; a name twin (D-044) carries the same rows, so it adds nothing
            val others = index.toxicIn(s.genus).map { it.first }.filter { it.i != s.i && it.name != s.name }
            val rows = others.flatMap { sp -> sp.traits.filter { it.category == "toxic" && it.part !in t.toxicParts }.map { sp to it } }
            if (rows.isEmpty()) return null
            val sevs = rows.map { it.second.severity.lowercase() }
            return CongenerToxic(
                genus = s.genus,
                parts = rows.map { it.second.part }.distinct().sortedBy { p -> partOrder.indexOf(p).let { if (it < 0) partOrder.size else it } },
                // the highest severity of those rows; one without a stated severity ranks as serious, as in own()
                severity = if (sevs.any { it !in order }) "" else sevs.maxBy { order.indexOf(it) },
                species = rows.map { it.first }.distinct(),
            )
        }

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
 * Toxic parts that cited OTHER species of the genus list and this species' own rows do not (genus union, audit
 * capture_lookalike_resolution-3). It is what the source says about the congeners, never a claim about this
 * species, and it only ever adds caution: a part missing from both still means "no listing", not safe.
 */
data class CongenerToxic(val genus: String, val parts: List<String>, val severity: String, val species: List<Species>) {
    /** Moderate, high, or unstated (treated as serious, as for a species' own rows). */
    val serious get() = severity != "low"

    /** "Other Prunus species: leaf, seed, stem listed toxic (high)" */
    val line: String
        get() {
            val named = if ("whole" in parts) "all parts" else
                parts.take(4).joinToString(", ") + if (parts.size > 4) " +${parts.size - 4}" else ""
            return "Other $genus species: $named listed toxic (${severity.ifBlank { "severity not stated" }})"
        }
}

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

/**
 * Banner headline: cited serious toxicity first, then relatives of poisonous species, mild, relatives of mild,
 * conditions. A species with rows whose congeners list more toxic parts (genus union) counts as a relative too.
 */
fun cautionHeadline(c: List<Pair<Candidate, TraitSummary>>): String = when {
    c.any { it.second.serious } -> "Caution: a poisonous candidate"
    c.any { it.second.relativeSerious || it.second.congener?.serious == true } -> "Caution: relative of a poisonous species"
    c.any { it.second.toxic } -> "Caution: some parts listed as mildly toxic"
    c.any { it.second.relative || it.second.congener != null } -> "Caution: relative of a mildly toxic species"
    else -> "Caution: edible only with conditions"
}
