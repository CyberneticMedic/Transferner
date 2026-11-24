package io.github.cyberneticmedic.transferner.core

import kotlin.math.abs
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min

/**
 * Separable, antialiased image resampling with the same convolution as Pillow's ImagingResample and
 * torchvision's antialias=True path: the kernel support widens with the downscale factor, output pixel i
 * of a source box samples around box0 + (i + 0.5) * scale, and results are rounded to 8 bits after each
 * pass (horizontal first). Matching the training-time resize keeps the on-phone inputs close to what the
 * models were trained and calibrated on.
 */
object Resample {
    enum class Kernel(val support: Float) {
        TRIANGLE(1f),                                   // bilinear
        CUBIC(2f);                                      // bicubic, a = -0.5

        fun weight(x0: Float): Float {
            val x = abs(x0)
            return when (this) {
                TRIANGLE -> if (x < 1f) 1f - x else 0f
                CUBIC -> {
                    val a = -0.5f
                    when {
                        x < 1f -> ((a + 2f) * x - (a + 3f)) * x * x + 1f
                        x < 2f -> (((x - 5f) * x + 8f) * x - 4f) * a
                        else -> 0f
                    }
                }
            }
        }
    }

    private class Coeffs(val start: IntArray, val count: IntArray, val w: FloatArray, val k: Int)

    private fun coeffs(inSize: Int, outSize: Int, b0: Float, b1: Float, kernel: Kernel): Coeffs {
        val scale = (b1 - b0) / outSize
        val filterScale = max(scale, 1f)
        val support = kernel.support * filterScale
        val k = kotlin.math.ceil(support).toInt() * 2 + 1
        val start = IntArray(outSize)
        val count = IntArray(outSize)
        val w = FloatArray(outSize * k)
        for (i in 0 until outSize) {
            val center = b0 + (i + 0.5f) * scale
            var xmin = floor(center - support + 0.5f).toInt()
            if (xmin < 0) xmin = 0
            var xmax = floor(center + support + 0.5f).toInt()
            if (xmax > inSize) xmax = inSize
            val n = min(xmax - xmin, k)
            var total = 0f
            for (j in 0 until n) {
                val v = kernel.weight((j + xmin - center + 0.5f) / filterScale)
                w[i * k + j] = v
                total += v
            }
            if (total != 0f) for (j in 0 until n) w[i * k + j] /= total
            start[i] = xmin
            count[i] = n
        }
        return Coeffs(start, count, w, k)
    }

    private fun clip8(v: Float): Float {
        val r = floor(v + 0.5f)                         // half up, as Pillow's fixed-point (+ 1/2, shift)
        return if (r < 0f) 0f else if (r > 255f) 255f else r
    }

    /**
     * Resample the [box] (x0, y0, x1, y1, source pixels) of an ARGB image to outW x outH.
     * Returns planar RGB (3 * outH * outW), values 0..255, rounded like an 8-bit pipeline.
     */
    fun resample(
        argb: IntArray, w: Int, h: Int, box: FloatArray, outW: Int, outH: Int, kernel: Kernel,
    ): FloatArray {
        val cx = coeffs(w, outW, box[0], box[2], kernel)
        val cy = coeffs(h, outH, box[1], box[3], kernel)
        var yLo = h
        var yHi = 0
        for (i in 0 until outH) {
            yLo = min(yLo, cy.start[i]); yHi = max(yHi, cy.start[i] + cy.count[i])
        }
        val rows = yHi - yLo
        // horizontal pass over the rows the vertical pass reads: planar (3, rows, outW)
        val tmp = FloatArray(3 * rows * outW)
        val plane = rows * outW
        for (r in 0 until rows) {
            val rowOff = (yLo + r) * w
            for (i in 0 until outW) {
                var sr = 0f; var sg = 0f; var sb = 0f
                val s0 = cx.start[i]
                val wo = i * cx.k
                for (j in 0 until cx.count[i]) {
                    val p = argb[rowOff + s0 + j]
                    val c = cx.w[wo + j]
                    sr += c * ((p shr 16) and 0xFF)
                    sg += c * ((p shr 8) and 0xFF)
                    sb += c * (p and 0xFF)
                }
                val o = r * outW + i
                tmp[o] = clip8(sr); tmp[plane + o] = clip8(sg); tmp[2 * plane + o] = clip8(sb)
            }
        }
        val out = FloatArray(3 * outH * outW)
        val oplane = outH * outW
        for (y in 0 until outH) {
            val s0 = cy.start[y] - yLo
            val wo = y * cy.k
            for (x in 0 until outW) {
                for (ch in 0 until 3) {
                    var s = 0f
                    val base = ch * plane + x
                    for (j in 0 until cy.count[y]) s += cy.w[wo + j] * tmp[base + (s0 + j) * outW]
                    out[ch * oplane + y * outW + x] = clip8(s)
                }
            }
        }
        return out
    }
}
