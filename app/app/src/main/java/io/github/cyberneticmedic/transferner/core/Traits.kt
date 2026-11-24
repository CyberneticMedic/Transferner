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
) {
    val toxic get() = ingestionToxic || contactToxic
    val needsCaution get() = toxic || conditionalParts.isNotEmpty()

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
        get() = buildList {
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

        fun of(s: Species): TraitSummary {
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

/** D-013: a caution sign when any shown candidate is toxic or conditional; no safe/unsafe verdict. */
fun cautionCandidates(top: List<Candidate>): List<Pair<Candidate, TraitSummary>> =
    top.map { it to TraitSummary.of(it.species) }.filter { it.second.needsCaution }

/** Banner headline for the caution candidates: serious toxicity first, then mild, then conditions only. */
fun cautionHeadline(c: List<Pair<Candidate, TraitSummary>>): String = when {
    c.any { it.second.serious } -> "Caution: a poisonous candidate"
    c.any { it.second.toxic } -> "Caution: some parts listed as mildly toxic"
    else -> "Caution: edible only with conditions"
}
