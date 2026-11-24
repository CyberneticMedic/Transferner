package io.github.cyberneticmedic.transferner.core

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln
import kotlin.math.max

/**
 * 28 scientific names appear twice in the label set (ACCEPTED/DOUBTFUL authorship twins; audit capture_model-1)
 * and split their probability between the two classes. The app sums each name onto its first index.
 */
class NameAliases(catalog: Catalog) {
    val canonical: IntArray
    val twins: Int

    init {
        val first = HashMap<String, Int>()
        canonical = IntArray(catalog.species.size) { i -> first.getOrPut(catalog.species[i].name) { i } }
        twins = canonical.indices.count { canonical[it] != it }
    }

    fun merge(p: FloatArray): FloatArray {
        if (twins == 0) return p
        val out = FloatArray(p.size)
        for (i in p.indices) out[canonical[i]] += p[i]
        return out
    }
}

/**
 * BioCLIP text embeddings of the 4,411 species names (text_emb.bin, fp16, L2-normalised). The text score, the
 * largest cosine between the photo's embedding and any name, separates plants outside the label set from
 * those inside better than the top probability does (audit capture_open_set-3: AUROC 0.893 vs 0.842).
 */
class TextIndex(bytes: ByteArray, val n: Int, val dim: Int) {
    private val t = FloatArray(n * dim)

    init {
        require(bytes.size == n * dim * 2) { "text_emb.bin has ${bytes.size} bytes, expected ${n * dim * 2}" }
        val sb = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        for (i in t.indices) t[i] = SpeciesHead.halfToFloat(sb.get(i))
    }

    fun score(e: FloatArray): Float {
        require(e.size == dim)
        var norm = 0f
        for (v in e) norm += v * v
        val inv = if (norm > 0f) 1f / kotlin.math.sqrt(norm) else 1f
        var best = Float.NEGATIVE_INFINITY
        for (j in 0 until n) {
            var s = 0f
            val o = j * dim
            for (k in 0 until dim) s += t[o + k] * e[k]
            best = max(best, s * inv)
        }
        return best
    }
}

enum class Level { SPECIES, GENUS, UNCERTAIN }

/**
 * What the result screen headlines (audit capture_model-6 + capture_open_set-4): the species when p1 >= 0.8,
 * else "<Genus> sp." when that genus' summed probability >= 0.8, else "Uncertain" (the shown label is right
 * 96.6% of the time on regional test against 85.3% for always naming the species). A text score below the
 * threshold moves the result down one level; "uncertain" with a low text score means the plant may not be in
 * the label set at all, so its toxicity is unknown.
 */
data class Verdict(
    val level: Level,
    val species: Species,
    val p1: Float,
    val genus: String,
    val genusP: Float,
    val text: Float?,
    val demoted: Boolean,
) {
    val maybeOutside get() = demoted && level == Level.UNCERTAIN
}

/** The capture caution banner (D-038 revised, D-044): see [CaptureRules.caution]. */
data class Caution(
    val show: Boolean,
    val strong: Boolean,                         // summed toxic + relative probability >= banner_strong
    val sum: Float,
    val inTop: List<Pair<Candidate, TraitSummary>>,
    val also: List<Pair<Candidate, TraitSummary>>,   // contributors outside the shown top 5, most likely first
) {
    companion object {
        /**
         * Two photos (D-044): the banner of the fused result united with each photo's own banner, so a candidate
         * one photo flagged is never dropped by the fusion (false positives over false negatives). Each species
         * keeps its highest probability; the headline then ranks the most serious one first.
         */
        fun union(cs: List<Caution>): Caution {
            fun merge(rows: List<Pair<Candidate, TraitSummary>>) =
                rows.groupBy { it.first.species.i }.values.map { g -> g.maxBy { it.first.p } }.sortedByDescending { it.first.p }
            val shown = cs.filter { it.show }
            val inTop = merge(shown.flatMap { it.inTop })
            val ids = inTop.map { it.first.species.i }.toSet()
            val also = merge(shown.flatMap { it.also }).filter { it.first.species.i !in ids }.take(3)
            return Caution(shown.isNotEmpty(), shown.any { it.strong }, cs.maxOf { it.sum }, inTop, also)
        }
    }
}

data class CaptureAnalysis(
    val probs: FloatArray,                       // merged over name twins, sums to 1
    val top: List<Candidate>,
    val verdict: Verdict,
    val caution: Caution,
    val secondPhoto: Boolean,                    // edible and toxic candidates both >= 5% with p1 < 0.9
    val photos: Int = 1,
)

class CaptureRules(private val catalog: Catalog, private val index: WarningIndex, val tuning: Tuning) {
    private val summaries = catalog.species.map { TraitSummary.of(it, index) }
    // D-045: plus the genus union (a species whose cited congeners list toxic parts its own rows do not). Regional
    // test: serious misses 0.11% -> 0.09% of photos (encounter-weighted 0.14% -> 0.09%), false alarms +0.4 pt
    private val toxicOrRelative = BooleanArray(summaries.size) {
        summaries[it].toxic || summaries[it].relative || summaries[it].congener != null }
    private val genusOf: Map<String, List<Int>> = catalog.species.indices.groupBy { catalog.species[it].genus }

    fun summary(i: Int) = summaries[i]

    fun analyze(probs: FloatArray, text: Float?, photos: Int = 1): CaptureAnalysis {
        val top = Detections.topK(probs, 5).map { Candidate(catalog.species[it], probs[it]) }
        return CaptureAnalysis(probs, top, verdict(probs, text), caution(probs, top), secondPhoto(top), photos)
    }

    fun verdict(p: FloatArray, text: Float?): Verdict {
        val i1 = Detections.topK(p, 1)[0]
        var bestGenus = catalog.species[i1].genus
        var bestP = 0f
        for ((g, idx) in genusOf) {
            var s = 0f
            for (i in idx) s += p[i]
            if (s > bestP) { bestP = s; bestGenus = g }
        }
        val t = tuning
        val base = when {
            p[i1] >= t.speciesMin -> Level.SPECIES
            bestP >= t.genusMin -> Level.GENUS
            else -> Level.UNCERTAIN
        }
        val low = text != null && text < t.textMin
        val level = if (!low) base else when (base) {
            Level.SPECIES -> if (bestP >= t.genusMin) Level.GENUS else Level.UNCERTAIN
            else -> Level.UNCERTAIN
        }
        return Verdict(level, catalog.species[i1], p[i1], bestGenus, bestP, text, low)
    }

    /**
     * Banner when the summed probability of cited-toxic species, their relatives (D-038; carrot family, D-045) and
     * species with congener-only toxic parts (genus union, D-045) reaches banner_sum
     * (0.002), or a conditional-only candidate in the top 5 reaches conditional_min. The read-only audit
     * measured this against D-038's "any toxic species in the top 5, at any probability" rule: regional test
     * misses 57 -> 43 photos, false alarms 32.9% -> 23.7%. Toxic chips stay on every row regardless.
     */
    fun caution(p: FloatArray, top: List<Candidate>): Caution {
        var sum = 0.0
        for (i in p.indices) if (toxicOrRelative[i]) sum += p[i]
        val inTop = top.map { it to summaries[it.species.i] }.filter { (c, t) ->
            t.toxic || t.relative || t.congener != null || (t.conditionalParts.isNotEmpty() && c.p >= tuning.conditionalMin)
        }
        val conditional = inTop.any { (_, t) -> !t.toxic && !t.relative && t.congener == null }
        val shown = top.map { it.species.i }.toSet()
        val also = p.indices.filter { toxicOrRelative[it] && it !in shown && p[it] >= 0.0005f }
            .sortedByDescending { p[it] }.take(3).map { Candidate(catalog.species[it], p[it]) to summaries[it] }
        val show = sum >= tuning.bannerSum || conditional
        return Caution(show, sum >= tuning.bannerStrong, sum.toFloat(), if (show) inTop else emptyList(), if (show) also else emptyList())
    }

    /**
     * Second photo of a different part (audit capture_lookalike_resolution-5): asked when the top match is
     * below 0.9 and the candidates at >= 5% include both an edible/conditional species and a cited-toxic one.
     * It fires on about 2% of captures and cuts dangerous edible/toxic mix-ups by about 31%.
     */
    fun secondPhoto(top: List<Candidate>): Boolean {
        if ((top.firstOrNull()?.p ?: 1f) >= 0.9f) return false
        val near = top.filter { it.p >= 0.05f }.map { summaries[it.species.i] }
        return near.any { it.edibleParts.isNotEmpty() || it.conditionalParts.isNotEmpty() } && near.any { it.toxic }
    }

    companion object {
        /** Two photos of one plant: the normalised geometric mean of their probabilities (mean log-probability). */
        fun fuse(a: FloatArray, b: FloatArray): FloatArray {
            val l = FloatArray(a.size) { 0.5f * (ln(max(a[it], 1e-12f)) + ln(max(b[it], 1e-12f))) }
            return Detections.softmax(l)
        }
    }
}
