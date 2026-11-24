package io.github.cyberneticmedic.transferner.core

/**
 * What the cited trait rows say about one species (D-004/D-005/D-027). Nothing here is inferred: a species
 * with no rows has no data, and absence of a toxic row never means safe.
 */
data class TraitSummary(
    val hasData: Boolean,
    val ingestionToxic: Boolean,
    val contactToxic: Boolean,
    val severity: String,                       // highest listed ingestion severity, or ""
    val toxicParts: List<String>,
    val edibleParts: List<String>,
    val conditionalParts: List<String>,
    val genusLevel: Boolean,                    // some rows come from a genus page, not this species
) {
    val toxic get() = ingestionToxic || contactToxic
    val needsCaution get() = toxic || conditionalParts.isNotEmpty()

    /** Short label for a result row. */
    val chip: String
        get() = when {
            !hasData -> "No trait data"
            ingestionToxic && severity.isNotBlank() -> "Poisonous if eaten · $severity"
            ingestionToxic -> "Poisonous if eaten"
            contactToxic -> "Skin contact: dermatitis listed"
            conditionalParts.isNotEmpty() -> "Edible only with conditions"
            edibleParts.any { it != "unspecified" } -> "Edible part listed: " + edibleParts.filter { it != "unspecified" }.joinToString(", ")
            edibleParts.isNotEmpty() -> "Edible · part not stated"
            else -> "Listed, no edible or toxic part"
        }

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
            val toxic = t.filter { it.category == "toxic" || it.ingestionToxic == 1 || it.contactToxic == 1 }
            val sev = t.filter { it.ingestionToxic == 1 }.map { it.severity.lowercase() }
                .maxByOrNull { order.indexOf(it) } ?: ""
            return TraitSummary(
                hasData = t.isNotEmpty(),
                ingestionToxic = t.any { it.ingestionToxic == 1 },
                contactToxic = t.any { it.contactToxic == 1 },
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
