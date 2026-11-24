package io.github.cyberneticmedic.transferner.core

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

data class Box(val x0: Float, val y0: Float, val x1: Float, val y1: Float, val score: Float)

data class GenusGuess(val genus: String, val family: String, val p: Float)

/** One live frame: whole-frame genus guesses (the live label, D-033) and class-agnostic plant boxes. */
data class LiveResult(
    val genera: List<GenusGuess>,
    val topFamily: String,
    val topFamilyP: Float,
    val boxes: List<Box>,                       // in model input pixels (size x size, squashed frame)
    val size: Int,
    val bestScore: Float = 0f,                  // highest objectness this frame, shown or not (diagnostics)
)

object Detections {
    fun softmax(x: FloatArray): FloatArray {
        var m = Float.NEGATIVE_INFINITY
        for (v in x) m = max(m, v)
        val out = FloatArray(x.size)
        var s = 0.0
        for (i in x.indices) { val e = exp(x[i] - m); out[i] = e; s += e }
        for (i in out.indices) out[i] = (out[i] / s).toFloat()
        return out
    }

    /** Indices of the k largest values, largest first. */
    fun topK(x: FloatArray, k: Int): IntArray {
        val kk = min(k, x.size)
        val idx = IntArray(kk) { -1 }
        val v = FloatArray(kk) { Float.NEGATIVE_INFINITY }
        for (i in x.indices) {
            val xi = x[i]
            if (xi <= v[kk - 1]) continue
            var j = kk - 1
            while (j > 0 && v[j - 1] < xi) { v[j] = v[j - 1]; idx[j] = idx[j - 1]; j-- }
            v[j] = xi; idx[j] = i
        }
        return idx
    }

    fun dequant(raw: ByteArray, t: TensorInfo): FloatArray = FloatArray(raw.size) { (raw[it] - t.zeroPoint) * t.scale }

    /** torchvision BoxCoder.decode_single against xyxy anchors, clamped to [0, size]. Returns A x 4 xyxy. */
    fun decode(reg: FloatArray, meta: DetectorMeta): FloatArray {
        val (wx, wy, ww, wh) = meta.boxCoder.weights
        val clip = meta.boxCoder.clip
        val s = meta.size.toFloat()
        val out = FloatArray(reg.size)
        for (a in meta.anchors.indices) {
            val an = meta.anchors[a]
            val aw = an[2] - an[0]
            val ah = an[3] - an[1]
            val cx = an[0] + 0.5f * aw
            val cy = an[1] + 0.5f * ah
            val dx = reg[4 * a] / wx
            val dy = reg[4 * a + 1] / wy
            val dw = min(reg[4 * a + 2] / ww, clip)
            val dh = min(reg[4 * a + 3] / wh, clip)
            val pcx = dx * aw + cx
            val pcy = dy * ah + cy
            val pw = exp(dw) * aw
            val ph = exp(dh) * ah
            out[4 * a] = (pcx - 0.5f * pw).coerceIn(0f, s)
            out[4 * a + 1] = (pcy - 0.5f * ph).coerceIn(0f, s)
            out[4 * a + 2] = (pcx + 0.5f * pw).coerceIn(0f, s)
            out[4 * a + 3] = (pcy + 0.5f * ph).coerceIn(0f, s)
        }
        return out
    }

    private fun iou(b: FloatArray, i: Int, j: Int): Float {
        val ix0 = max(b[4 * i], b[4 * j]); val iy0 = max(b[4 * i + 1], b[4 * j + 1])
        val ix1 = min(b[4 * i + 2], b[4 * j + 2]); val iy1 = min(b[4 * i + 3], b[4 * j + 3])
        val inter = max(0f, ix1 - ix0) * max(0f, iy1 - iy0)
        val ai = (b[4 * i + 2] - b[4 * i]) * (b[4 * i + 3] - b[4 * i + 1])
        val aj = (b[4 * j + 2] - b[4 * j]) * (b[4 * j + 3] - b[4 * j + 1])
        val u = ai + aj - inter
        return if (u <= 0f) 0f else inter / u
    }

    /** Class-agnostic boxes: top-k anchors by objectness, greedy NMS (torchvision.ops.nms), best first. */
    fun boxes(objectness: FloatArray, decoded: FloatArray, meta: DetectorMeta, maxBoxes: Int = meta.postprocess.maxBoxes): List<Box> {
        val cand = topK(objectness, meta.postprocess.objectnessTopK)
        val keep = ArrayList<Int>()
        val removed = BooleanArray(cand.size)
        for (a in cand.indices) {
            if (removed[a]) continue
            keep.add(cand[a])
            if (keep.size >= maxBoxes) break
            for (b in a + 1 until cand.size) if (!removed[b] && iou(decoded, cand[a], cand[b]) > meta.postprocess.nmsIou) removed[b] = true
        }
        return keep.map { Box(decoded[4 * it], decoded[4 * it + 1], decoded[4 * it + 2], decoded[4 * it + 3], objectness[it]) }
    }

    /** Whole-frame genus distribution -> top-k genera and the top family (genus probabilities summed). */
    fun genera(probs: FloatArray, meta: DetectorMeta, k: Int = 4): Triple<List<GenusGuess>, String, Float> {
        val top = topK(probs, k).map { GenusGuess(meta.genera[it], meta.families[meta.familyOf[it]], probs[it]) }
        val fam = FloatArray(meta.families.size)
        for (g in probs.indices) fam[meta.familyOf[g]] += probs[g]
        val f = topK(fam, 1)[0]
        return Triple(top, meta.families[f], fam[f])
    }
}
