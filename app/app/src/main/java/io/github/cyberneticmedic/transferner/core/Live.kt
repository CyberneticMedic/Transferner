package io.github.cyberneticmedic.transferner.core

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/**
 * tuning.json (bundle 0.6+, export/bundle.py; D-044): live and capture thresholds that belong to the shipped
 * models. Older bundles have no file and get these defaults, which are the read-only audit's values.
 */
@Serializable
data class Tuning(
    /** Plant-presence gate: top-5 whole-frame genus mass below which no box is drawn (0 = off). */
    @SerialName("presence_boxes") val presenceBoxes: Float = 0f,
    /** ... and below which the live toxic row is hidden; lower than the box gate (false positives over negatives). */
    @SerialName("presence_warn") val presenceWarn: Float = 0f,
    /** Boxes after the first need the slider value plus this, and at least [secondaryFloor] (D-039 regression fix). */
    @SerialName("secondary_offset") val secondaryOffset: Float = 0.15f,
    @SerialName("secondary_floor") val secondaryFloor: Float = 0.4f,
    /** A box this much inside a kept box is dropped, unless the kept box fills the frame. */
    val containment: Float = 0.8f,
    /** Whole-frame label EMA time constant (alpha = 1 - exp(-dt / tau)); 0.56 s = alpha 0.2 at 8 fps. */
    @SerialName("label_tau_s") val labelTauS: Float = 0.56f,
    /** Warning (and walk-scoring) EMA: faster than the label, as 0.5's alpha 0.4 at 8 fps, so a brief toxic
     *  sighting still reaches the 2% flag (review of 0.6: at 0.56 s a 3-frame p = 0.04 pass never warned). */
    @SerialName("warn_tau_s") val warnTauS: Float = 0.245f,
    /** The live toxic row stays up this many frames after its last firing frame. */
    @SerialName("warn_hold_frames") val warnHoldFrames: Int = 4,
    @SerialName("walk_fps") val walkFps: Int = 4,
    @SerialName("alert_tau_s") val alertTauS: Float = 1.0f,
    @SerialName("alert_enter") val alertEnter: Float = 0.5f,
    @SerialName("alert_hold_s") val alertHoldS: Float = 0.75f,
    @SerialName("alert_exit") val alertExit: Float = 0.35f,
    /** Alert-tier genera score at least this weight (D-045): 9 of the 47 default alert genera had e_g < 0.5
     *  (chestnut, prickly pear, hawthorn, pear...) and could never reach alert_enter. The BioCLIP check, now on
     *  the genus' cited edible species, keeps their non-edible congeners (Bradford pear) from buzzing. */
    @SerialName("alert_weight_floor") val alertWeightFloor: Float = 0.8f,
    /** Forage chip (no haptic) from this accumulated score. */
    @SerialName("chip_min") val chipMin: Float = 0.3f,
    @SerialName("refractory_s") val refractoryS: Float = 600f,
    /** Serious-toxic haptic: accumulated serious-toxic genus mass held above this for [toxicHoldS]. */
    @SerialName("toxic_haptic") val toxicHaptic: Float = 0.3f,
    @SerialName("toxic_hold_s") val toxicHoldS: Float = 1.0f,
    /** Walk alerts are confirmed when BioCLIP puts at least this probability on the alerted genus. */
    @SerialName("verify_genus_p") val verifyGenusP: Float = 0.25f,
    /** Capture: text score (max cosine to the species-name embeddings) below which the result drops a level. */
    @SerialName("text_min") val textMin: Float = 0.561f,
    /** Capture caution banner: summed probability of cited-toxic species and their relatives. */
    @SerialName("banner_sum") val bannerSum: Float = 0.002f,
    @SerialName("banner_strong") val bannerStrong: Float = 0.02f,
    /** A conditional-only species raises the banner only from this probability. */
    @SerialName("conditional_min") val conditionalMin: Float = 0.02f,
    /** Capture levels: species from p1, genus from the genus sum. */
    @SerialName("species_min") val speciesMin: Float = 0.8f,
    @SerialName("genus_min") val genusMin: Float = 0.8f,
)

/** Exponential smoothing whose weight depends on the time since the last update, not on the frame rate. */
class TimeEma(private val tauS: Float) {
    private var value: FloatArray? = null
    private var lastMs = 0L

    fun update(x: FloatArray, nowMs: Long): FloatArray {
        val v = value
        val out = if (v == null || v.size != x.size) x.copyOf() else {
            val a = alpha((nowMs - lastMs).coerceAtLeast(0L))
            FloatArray(x.size) { a * x[it] + (1 - a) * v[it] }
        }
        value = out; lastMs = nowMs
        return out
    }

    fun alpha(dtMs: Long): Float = if (tauS <= 0f) 1f else (1 - exp(-dtMs / 1000f / tauS)).coerceIn(0f, 1f)
    fun reset() { value = null }
}

/** Scalar version for alert scores. It starts from 0 (first frame weighted as if 250 ms had passed), so one
 *  strong frame cannot put the score over an alert threshold by itself. */
class ScalarEma(private val tauS: Float) {
    var value = 0f
        private set
    private var lastMs = -1L
    fun update(x: Float, nowMs: Long): Float {
        val dt = if (lastMs < 0) 250L else (nowMs - lastMs).coerceAtLeast(0L)
        val a = if (tauS <= 0f) 1f else (1 - exp(-dt / 1000f / tauS))
        value = a * x + (1 - a) * value
        lastMs = nowMs
        return value
    }
    fun reset() { value = 0f; lastMs = -1L }
}

/** Sum of the k largest probabilities: high when the frame looks like one of the known genera (plant presence). */
fun topMass(p: FloatArray, k: Int = 5): Float = Detections.topK(p, k).sumOf { p[it].toDouble() }.toFloat()

object BoxRules {
    fun area(b: Box) = max(0f, b.x1 - b.x0) * max(0f, b.y1 - b.y0)

    fun iou(a: Box, b: Box): Float {
        val inter = max(0f, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0f, min(a.y1, b.y1) - max(a.y0, b.y0))
        val u = area(a) + area(b) - inter
        return if (u <= 0f) 0f else inter / u
    }

    /** Share of [inner]'s area that lies inside [outer]. */
    fun inside(inner: Box, outer: Box): Float {
        val inter = max(0f, min(inner.x1, outer.x1) - max(inner.x0, outer.x0)) * max(0f, min(inner.y1, outer.y1) - max(inner.y0, outer.y0))
        val a = area(inner)
        return if (a <= 0f) 0f else inter / a
    }

    fun fillsFrame(b: Box, size: Int) = area(b) >= 0.8f * size * size

    /**
     * Candidate boxes (best first, after NMS) -> the boxes worth showing: the best from [primaryThr], the rest
     * from max(primaryThr + offset, floor), and no box mostly inside a kept box unless that kept box fills the
     * frame (then the sub-frame boxes are the interesting ones). Greedy NMS cannot remove nested boxes: their
     * IoU equals their area ratio (audit: 70-87% of second boxes sat inside the first).
     */
    fun organize(cands: List<Box>, primaryThr: Float, t: Tuning, size: Int, max: Int = 4): List<Box> {
        val secondary = max(primaryThr + t.secondaryOffset, t.secondaryFloor)
        val kept = ArrayList<Box>()
        for (b in cands) {
            if (kept.size >= max) break
            if (b.score < (if (kept.isEmpty()) primaryThr else secondary)) continue
            if (kept.any { k -> !fillsFrame(k, size) && inside(b, k) >= t.containment }) continue
            kept.add(b)
        }
        return kept
    }
}

/**
 * Light IoU tracker so boxes do not pop in and out between frames (audit: box appear/vanish events per frame
 * 0.65 -> 0.27 on simulated scenes). A track starts from a box that passed [BoxRules.organize], is shown after
 * [confirmHits] consecutive hits, coasts through [maxMiss] missed frames, and smooths its box and score.
 */
class BoxTracker(private val iouMatch: Float = 0.3f, private val ema: Float = 0.5f, private val confirmHits: Int = 2,
                 private val maxMiss: Int = 2, private val maxTracks: Int = 4) {
    data class Track(val id: Int, val box: Box, val hits: Int, val misses: Int)

    private var tracks = listOf<Track>()
    private var nextId = 0

    fun update(boxes: List<Box>): List<Box> {
        val unmatched = boxes.toMutableList()
        val next = ArrayList<Track>()
        for (t in tracks.sortedByDescending { it.box.score }) {
            val m = unmatched.maxByOrNull { BoxRules.iou(it, t.box) }
            if (m != null && BoxRules.iou(m, t.box) >= iouMatch) {
                unmatched.remove(m)
                next.add(t.copy(box = blend(t.box, m), hits = t.hits + 1, misses = 0))
            } else if (t.misses < maxMiss) {
                next.add(t.copy(misses = t.misses + 1))
            }
        }
        for (b in unmatched) if (next.size < maxTracks) next.add(Track(nextId++, b, 1, 0))
        tracks = next
        return next.filter { it.hits >= confirmHits }.sortedByDescending { it.box.score }.map { it.box }
    }

    private fun blend(a: Box, b: Box) = Box(
        ema * b.x0 + (1 - ema) * a.x0, ema * b.y0 + (1 - ema) * a.y0,
        ema * b.x1 + (1 - ema) * a.x1, ema * b.y1 + (1 - ema) * a.y1, ema * b.score + (1 - ema) * a.score)

    fun reset() { tracks = emptyList() }
}

/** forage.json (export/bundle.py, data/forage/default_tiers.csv): which genera a walk alerts on. */
@Serializable
data class ForageGenus(
    val g: Int,                                  // detector genus index
    val genus: String,
    val tier: String,                            // alert | show | off (default; the user can change it)
    val weight: Float,                           // share of the genus' regional sightings that are listed edible species
    val label: String = "",                      // common name for the alert text
    val note: String = "",
)

@Serializable
data class ForageMeta(val genera: List<ForageGenus>, val source: String = "")

enum class Tier {
    ALERT, SHOW, OFF;

    companion object {
        /** forage.json's default tier: "alert", "show" or "off". */
        fun of(s: String) = when (s) { "alert" -> ALERT; "off" -> OFF; else -> SHOW }
    }
}

/**
 * Forage score on the unfocused, smoothed whole-frame distribution (audit edible_targeting_app-01/03):
 * S = sum over target genera of p_g * e_g, where e_g is the share of the genus' regional sightings that are
 * listed edible species. Alert-tier genera can buzz; show-tier genera only raise a chip.
 */
class ForageScorer(meta: ForageMeta, nGenera: Int, overrides: Map<String, Tier>, alertFloor: Float = 0f) {
    val byIndex: Map<Int, ForageGenus> = meta.genera.associateBy { it.g }
    private val alertW = FloatArray(nGenera)
    private val showW = FloatArray(nGenera)

    init {
        for (f in meta.genera) {
            if (f.g !in 0 until nGenera) continue
            val tier = overrides[f.genus] ?: Tier.of(f.tier)
            when (tier) {
                Tier.ALERT -> alertW[f.g] = max(f.weight, alertFloor)
                Tier.SHOW -> showW[f.g] = f.weight
                Tier.OFF -> {}
            }
        }
    }

    data class Score(val alert: Float, val alertGenus: Int, val show: Float, val showGenus: Int)

    fun score(p: FloatArray): Score {
        var a = 0f; var ag = -1; var am = 0f
        var s = 0f; var sg = -1; var sm = 0f
        for (g in p.indices) {
            if (g >= alertW.size) break
            val wa = p[g] * alertW[g]
            if (wa > 0f) { a += wa; if (wa > am) { am = wa; ag = g } }
            val ws = p[g] * showW[g]
            if (ws > 0f) { s += ws; if (ws > sm) { sm = ws; sg = g } }
        }
        return Score(a, ag, s, sg)
    }
}

/**
 * Alert hysteresis (audit edible_targeting_app-02/03): time-based EMA of the forage score (tau ~1 s); an alert
 * candidate starts when the smoothed score stays >= enter for [Tuning.alertHoldS] and clears below exit. The
 * caller reports the BioCLIP check: a confirmed genus stays quiet for the refractory period (10 min), a
 * rejected one for [REJECT_S], and an unchecked one (no model, another check running) is released at once.
 */
class AlertAccumulator(private val t: Tuning) {
    private val ema = ScalarEma(t.alertTauS)
    private var aboveSince = -1L
    private var active = false
    private val quietUntil = HashMap<Int, Long>()
    val smoothed get() = ema.value

    /** Returns the genus to check when an alert candidate starts this frame, else -1. */
    fun update(score: Float, genus: Int, nowMs: Long): Int {
        val s = ema.update(score, nowMs)
        if (active) {
            if (s < t.alertExit) { active = false; aboveSince = -1L }
            return -1
        }
        if (s < t.alertEnter || genus < 0) { aboveSince = -1L; return -1 }
        if (aboveSince < 0) aboveSince = nowMs
        if (nowMs - aboveSince < (t.alertHoldS * 1000).toLong()) return -1
        if (nowMs < (quietUntil[genus] ?: Long.MIN_VALUE)) return -1
        active = true
        quietUntil[genus] = nowMs + PENDING_S * 1000          // until the check reports back
        return genus
    }

    fun confirmed(genus: Int, nowMs: Long) { quietUntil[genus] = nowMs + (t.refractoryS * 1000).toLong() }
    fun rejected(genus: Int, nowMs: Long) { quietUntil[genus] = nowMs + REJECT_S * 1000 }
    /** The candidate could not be checked: forget it so it can start again. */
    fun released(genus: Int) { quietUntil.remove(genus); active = false; aboveSince = -1L }

    fun reset() { ema.reset(); aboveSince = -1L; active = false }

    companion object {
        const val PENDING_S = 60L
        const val REJECT_S = 30L
    }
}

/** Serious-toxic haptic (D-038 kept visual; the haptic channel needs more): smoothed mass held above a level. */
class HoldDetector(private val level: Float, private val holdS: Float, tauS: Float, private val refractoryS: Float) {
    private val ema = ScalarEma(tauS)
    private var since = -1L
    private var last = -1L
    fun update(x: Float, nowMs: Long): Boolean {
        val s = ema.update(x, nowMs)
        if (s < level) { since = -1L; return false }
        if (since < 0) since = nowMs
        if (nowMs - since < (holdS * 1000).toLong()) return false
        if (last >= 0 && nowMs - last < (refractoryS * 1000).toLong()) return false
        last = nowMs
        return true
    }
    fun reset() { ema.reset(); since = -1L }
}
