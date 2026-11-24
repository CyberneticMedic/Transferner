package io.github.cyberneticmedic.transferner.core

import java.nio.ByteBuffer
import java.nio.ByteOrder

data class Candidate(val species: Species, val p: Float)

/**
 * The production linear probe (D-025/D-030) on a BioCLIP embedding: logits = (scale * e.W^T + b) / T,
 * softmax over all regional species (calibrated by T, D-013), top 5.
 */
class SpeciesHead(private val meta: HeadMeta, weights: ByteArray) {
    private val w = FloatArray(meta.n * meta.dim)

    init {
        require(weights.size == meta.n * meta.dim * 2) { "head.bin has ${weights.size} bytes, expected ${meta.n * meta.dim * 2}" }
        val sb = ByteBuffer.wrap(weights).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        for (i in w.indices) w[i] = halfToFloat(sb.get(i))
    }

    fun probs(embedding: FloatArray): FloatArray {
        require(embedding.size == meta.dim)
        val logits = FloatArray(meta.n)
        for (j in 0 until meta.n) {
            var s = 0f
            val o = j * meta.dim
            for (k in 0 until meta.dim) s += w[o + k] * embedding[k]
            logits[j] = (meta.scale * s + meta.bias[j]) / meta.temperature
        }
        return Detections.softmax(logits)
    }

    fun top(embedding: FloatArray, catalog: Catalog, k: Int = 5): List<Candidate> {
        val p = probs(embedding)
        return Detections.topK(p, k).map { Candidate(catalog.species[it], p[it]) }
    }

    companion object {
        /** IEEE 754 binary16 -> binary32 (no android.util.Half, so it also runs in JVM unit tests). */
        fun halfToFloat(h: Short): Float {
            val bits = h.toInt() and 0xFFFF
            val sign = (bits ushr 15) and 1
            val exp = (bits ushr 10) and 0x1F
            val mant = bits and 0x3FF
            if (exp == 0) {                                          // zero or subnormal: mant * 2^-24, exact
                val v = mant * 5.9604645e-8f
                return if (sign == 1) -v else v
            }
            val f = when (exp) {
                0x1F -> (sign shl 31) or (0xFF shl 23) or (mant shl 13)
                else -> (sign shl 31) or ((exp - 15 + 127) shl 23) or (mant shl 13)
            }
            return java.lang.Float.intBitsToFloat(f)
        }
    }
}
