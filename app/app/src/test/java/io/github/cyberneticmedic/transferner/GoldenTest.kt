package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.BundleJson
import io.github.cyberneticmedic.transferner.core.Catalog
import io.github.cyberneticmedic.transferner.core.DetectorMeta
import io.github.cyberneticmedic.transferner.core.Detections
import io.github.cyberneticmedic.transferner.core.HeadMeta
import io.github.cyberneticmedic.transferner.core.Preprocess
import io.github.cyberneticmedic.transferner.core.SpeciesHead
import io.github.cyberneticmedic.transferner.core.SpeciesModelMeta
import io.github.cyberneticmedic.transferner.core.TraitSummary
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import javax.imageio.ImageIO
import kotlin.math.abs

/**
 * Checks the app's Kotlin pre- and post-processing against the Python pipeline on one held-out photo.
 * Goldens and the bundle are data ($TRANSFERNER_DATA/app_golden, the latest $TRANSFERNER_DATA/bundle, written by
 * src/transferner/export/bundle.py --goldens); the tests skip when they are absent.
 */
class GoldenTest {
    private val data = System.getProperty("transferner.data").orEmpty()
    private val golden = File(data, "app_golden")
    private val bundle: File by lazy {
        // the bundle the goldens were made from (golden.json "bundle"), else the published one
        val pinned = File(golden, "golden.json").takeIf { it.exists() }?.let {
            BundleJson.parseToJsonElement(it.readText()).jsonObject["bundle"]?.jsonPrimitive?.content }
        if (pinned != null) return@lazy File(data, "bundle/$pinned")
        val latest = File(data, "bundle/latest.json")
        if (!latest.exists()) File(data, "bundle/none")
        else File(data, "bundle/" + BundleJson.decodeFromString<io.github.cyberneticmedic.transferner.core.Latest>(latest.readText()).version)
    }

    private fun need() = assumeTrue("goldens/bundle not found under '$data'", golden.isDirectory && bundle.isDirectory)
    private fun floats(name: String): FloatArray {
        val b = ByteBuffer.wrap(File(golden, name).readBytes()).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
        return FloatArray(b.remaining()).also { b.get(it) }
    }
    private fun bytes(name: String) = File(golden, name).readBytes()
    private val g: JsonObject by lazy { BundleJson.parseToJsonElement(File(golden, "golden.json").readText()).jsonObject }
    private val det: DetectorMeta by lazy { BundleJson.decodeFromString(File(bundle, "detector.json").readText()) }
    private val sp: SpeciesModelMeta by lazy {
        BundleJson.decodeFromString<io.github.cyberneticmedic.transferner.core.SpeciesModels>(File(bundle, "species.json").readText()).variants.first()
    }

    private fun photo(): Triple<IntArray, Int, Int> {
        val im = ImageIO.read(File(golden, "photo.jpg"))
        val px = im.getRGB(0, 0, im.width, im.height, null, 0, im.width)
        return Triple(px, im.width, im.height)
    }

    private fun stats(a: FloatArray, b: FloatArray): Pair<Double, Double> {
        assertEquals(a.size, b.size)
        val d = DoubleArray(a.size) { abs(a[it] - b[it]).toDouble() }
        d.sort()
        return d.average() to d[(0.99 * (d.size - 1)).toInt()]
    }

    @Test fun detectorInputMatchesTorchvision() {
        need()
        val (px, w, h) = photo()
        val rgb = FloatArray(3 * det.size * det.size)
        val q = Preprocess.detectorInput(px, w, h, det, rgb)
        val ref = bytes("detector_rgb320.u8").map { (it.toInt() and 0xFF).toFloat() }.toFloatArray()
        val (mean, p99) = stats(rgb, ref)
        println("detector rgb: mean |diff| %.3f, p99 %.1f (8-bit levels)".format(mean, p99))
        assertTrue("resize differs: mean $mean", mean < 1.5 && p99 <= 6.0)
        val qref = bytes("detector_input.i8").map { it.toFloat() }.toFloatArray()
        val (qm, qp) = stats(q.map { it.toFloat() }.toFloatArray(), qref)
        println("detector int8: mean |diff| %.3f, p99 %.1f (codes)".format(qm, qp))
        assertTrue(qm < 1.5 && qp <= 6.0)
    }

    @Test fun detectorPostprocessMatchesPython() {
        need()
        val probs = Detections.softmax(floats("detector_image_logits.f32"))
        val (top, _, _) = Detections.genera(probs, det, 5)
        val ref = g["detector"]!!.jsonObject["genus_top5"]!!.jsonArray
        for ((i, r) in ref.withIndex()) {
            assertEquals(r.jsonArray[0].jsonPrimitive.content, top[i].genus)
            assertEquals(r.jsonArray[1].jsonPrimitive.float, top[i].p, 1e-4f)
        }
        val reg = Detections.dequant(bytes("detector_boxes.i8"), det.outputs.boxes)
        val boxes = Detections.boxes(floats("detector_objectness.f32"), Detections.decode(reg, det), det, 5)
        val bref = g["detector"]!!.jsonObject["boxes_top5"]!!.jsonArray
        for ((i, r) in bref.withIndex()) {
            val e = r.jsonArray.map { it.jsonPrimitive.float }
            val b = boxes[i]
            listOf(b.x0, b.y0, b.x1, b.y1, b.score).zip(e).forEach { (x, y) -> assertEquals(y, x, 0.02f) }
        }
        println("detector top genus ${top[0].genus} ${top[0].p}, top box ${boxes[0]}")
    }

    @Test fun speciesInputMatchesOpenClip() {
        need()
        val (px, w, h) = photo()
        val x = Preprocess.speciesInput(px, w, h, sp)
        val (mean, p99) = stats(x, floats("species_input.f32"))
        println("species input: mean |diff| %.4f, p99 %.3f (normalised units; 1 level = 0.015)".format(mean, p99))
        assertTrue("bicubic crop differs: mean $mean", mean < 0.03 && p99 < 0.12)
    }

    @Test fun speciesHeadMatchesPython() {
        need()
        val meta: HeadMeta = BundleJson.decodeFromString(File(bundle, "head.json").readText())
        val catalog: Catalog = BundleJson.decodeFromString(File(bundle, "catalog.json").readText())
        val head = SpeciesHead(meta, File(bundle, meta.file).readBytes())
        val top = head.top(floats("species_embedding.f32"), catalog)
        val ref = g["species_top5"]!!.jsonArray
        for ((i, r) in ref.withIndex()) {
            assertEquals(r.jsonArray[0].jsonPrimitive.int, top[i].species.i)
            assertEquals(r.jsonArray[1].jsonPrimitive.float, top[i].p, 2e-3f)
        }
        println("species top 5: " + top.joinToString { "${it.species.name} %.3f".format(it.p) })
        val larkspur = TraitSummary.of(top.first { it.species.name == "Delphinium tricorne" }.species)
        assertTrue(larkspur.ingestionToxic && larkspur.severity == "high" && larkspur.needsCaution)
        val toothwort = TraitSummary.of(top[0].species)
        assertEquals("Edible part listed: root", toothwort.chip)
        assertEquals("No trait data", TraitSummary.of(catalog.species.first { it.name == "Cardamine dissecta" }).chip)
        // source-faithful wording (app feedback 2026-11-23): fig and white mulberry are edible with mildly toxic
        // parts, not "poisonous"; poison ivy leads with touch; larkspur stays poisonous
        val chip = { n: String -> TraitSummary.of(catalog.species.first { it.name == n }) }
        for (n in listOf("Ficus carica", "Morus alba", "Sambucus canadensis", "Toxicodendron radicans", "Delphinium tricorne", "Phytolacca americana"))
            println("%-24s %-55s | %s".format(n, chip(n).chip, chip(n).detail))
        assertEquals("Edible as described · some parts mildly toxic", chip("Ficus carica").chip)
        assertEquals("Fruit edible as described · some parts mildly toxic", chip("Morus alba").chip)
        assertTrue(!chip("Ficus carica").serious && !chip("Morus alba").serious)
        assertEquals("Toxic to touch and eat · high", chip("Toxicodendron radicans").chip)
        assertEquals("Poisonous if eaten · high", chip("Delphinium tricorne").chip)
        // D-038 warning flags: a species without rows whose genus has a cited toxic species is a "relative"
        val idx = io.github.cyberneticmedic.transferner.core.WarningIndex(catalog)
        val relSp = catalog.species.first { it.traits.isEmpty() && idx.genusSerious(it.genus) }
        val rel = TraitSummary.of(relSp, idx)
        println("${relSp.name}: ${rel.chip} | ${rel.detail}")
        println("relatives in the catalog: " + catalog.species.count { it.traits.isEmpty() && idx.genusToxic(it.genus) })
        assertTrue(rel.relative && rel.needsCaution && rel.relativeSerious)
        assertTrue(idx.genusSerious("Toxicodendron") && idx.genusToxic("Morus") && !idx.genusSerious("Morus"))
        assertTrue(!TraitSummary.of(catalog.species.first { it.name == "Cardamine dissecta" }, idx).relative)
        val (t5, extra) = head.topWithFlags(floats("species_embedding.f32"), catalog, idx)
        assertEquals(top.map { it.species.i }, t5.map { it.species.i })
        println("also possible beyond the top 5: " + extra.joinToString { "${it.species.name} %.4f".format(it.p) })
    }

    @Test fun halfFloat() {
        assertEquals(1f, SpeciesHead.halfToFloat(0x3C00.toShort()))
        assertEquals(-2f, SpeciesHead.halfToFloat(0xC000.toShort()))
        assertEquals(6.1035156e-5f, SpeciesHead.halfToFloat(0x0400.toShort()))
        assertEquals(5.9604645e-8f, SpeciesHead.halfToFloat(0x0001.toShort()))
        assertEquals(65504f, SpeciesHead.halfToFloat(0x7BFF.toShort()))
    }
}
