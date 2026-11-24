package io.github.cyberneticmedic.transferner.core

/** Model inputs from ARGB pixels, reproducing the Python transforms the models were trained with. */
object Preprocess {
    /**
     * Live detector: torchvision Resize((s, s), antialias=True) on the whole frame (a squash, as in
     * training), then (x / 255 - mean) / std, then the input tensor's int8 quantisation. Returns the int8
     * NCHW tensor; [rgbOut], when given, receives the resized planar RGB (0..255) for tests and overlays.
     */
    fun detectorInput(argb: IntArray, w: Int, h: Int, meta: DetectorMeta, rgbOut: FloatArray? = null): ByteArray {
        val s = meta.size
        val rgb = Resample.resample(argb, w, h, floatArrayOf(0f, 0f, w.toFloat(), h.toFloat()), s, s,
            Resample.Kernel.TRIANGLE)
        rgbOut?.let { rgb.copyInto(it) }
        val lut = int8Lut(meta)
        val out = ByteArray(rgb.size)
        val plane = s * s
        for (ch in 0 until 3) {
            val off = ch * plane
            val l = lut[ch]
            for (i in 0 until plane) out[off + i] = l[rgb[off + i].toInt()]
        }
        return out
    }

    /** Per channel, the int8 code of each 8-bit pixel value (numpy round-half-even, like the export). */
    fun int8Lut(meta: DetectorMeta): Array<ByteArray> = Array(3) { ch ->
        ByteArray(256) { v ->
            val x = (v / 255f - meta.mean[ch]) / meta.std[ch]
            val q = Math.rint((x / meta.input.scale).toDouble()).toInt() + meta.input.zeroPoint
            q.coerceIn(-128, 127).toByte()
        }
    }

    /**
     * Species model (BioCLIP's open_clip val transform): shortest side to 224 (long side truncated, as
     * torchvision), bicubic with antialias, centre crop 224, then CLIP mean/std. Done as one resample of
     * the equivalent source box, which samples exactly the same positions.
     */
    fun speciesInput(argb: IntArray, w: Int, h: Int, meta: SpeciesModelMeta, size: Int = 224): FloatArray {
        val (rw, rh) = if (w <= h) size to (size.toLong() * h / w).toInt() else (size.toLong() * w / h).toInt() to size
        val left = Math.rint((rw - size) / 2.0).toInt()
        val top = Math.rint((rh - size) / 2.0).toInt()
        val sx = w.toFloat() / rw
        val sy = h.toFloat() / rh
        val box = floatArrayOf(left * sx, top * sy, (left + size) * sx, (top + size) * sy)
        val rgb = Resample.resample(argb, w, h, box, size, size, Resample.Kernel.CUBIC)
        val plane = size * size
        for (ch in 0 until 3) {
            val m = meta.preprocess.mean[ch]
            val sd = meta.preprocess.std[ch]
            for (i in 0 until plane) rgb[ch * plane + i] = (rgb[ch * plane + i] / 255f - m) / sd
        }
        return rgb
    }
}
