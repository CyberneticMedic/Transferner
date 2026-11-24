package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.BundleJson
import io.github.cyberneticmedic.transferner.core.DetectorMeta
import io.github.cyberneticmedic.transferner.core.Latest
import io.github.cyberneticmedic.transferner.core.ManifestFile
import io.github.cyberneticmedic.transferner.data.BundleRepository
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File

/** D-045: a newer detector with outputs this app does not know must still load (every output gets a buffer). */
class BundleCompatTest {
    private fun sidecar(extra: String) = """{"model":"detector.tflite","signature":"serving_default","size":448,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5],
        "input":{"name":"args_0","dtype":"int8","shape":[1,3,448,448],"scale":0.02,"zero_point":0},
        "outputs":{"image_logits":{"name":"output_0","dtype":"int8","shape":[1,3]},"objectness":{"name":"output_1","dtype":"int8","shape":[1,2]},
                   "boxes":{"name":"output_2","dtype":"int8","shape":[1,2,4]}$extra},
        "box_coder":{"weights":[10,10,5,5],"clip":4.1},"anchors":[[0,0,1,1],[0,0,2,2]],
        "postprocess":{"objectness_topk":2,"nms_iou":0.55,"max_boxes":100},"genera":["A","B","C"],"families":["F"],"family_of":[0,0,0]}"""

    @Test fun unknownOutputsAreKeptForBuffers() {
        val m = BundleJson.decodeFromString<DetectorMeta>(sidecar(""",
            "region_logits":{"name":"output_3","dtype":"int8","shape":[1,9,3]},"future_map":{"name":"output_4","dtype":"int8","shape":[1,10,10]}"""))
        assertEquals(setOf("output_0", "output_1", "output_2", "output_3", "output_4"), m.outputMap.values.map { it.name }.toSet())
        assertEquals("output_3", m.outputs.regionLogits?.name)
        assertNull(BundleJson.decodeFromString<DetectorMeta>(sidecar("")).outputs.regionLogits)
    }

    @Test fun latestCarriesTheMinimumApp() {
        assertEquals(10L, BundleJson.decodeFromString<Latest>("""{"version":"0.6","manifest":"0.6/manifest.json","min_app":10}""").minApp)
        assertEquals(0L, BundleJson.decodeFromString<Latest>("""{"version":"0.5","manifest":"0.5/manifest.json"}""").minApp)
    }

    @Test fun appGroupsAreSkippedByOlderApps() {
        val rg = ManifestFile("detector_rg.tflite", 1, "x", "app:11")
        assertTrue(rg.wantedFor("cpu", 11) && rg.wantedFor("cpu", 12) && rg.wantedFor("cpu"))
        assertFalse(rg.wantedFor("cpu", 10) || rg.wantedFor("cpu", 0))
        assertTrue(ManifestFile("catalog.json", 1, "x").wantedFor("cpu", 0))
        assertFalse(ManifestFile("species_gpu.tflite", 1, "x", "species:gpu").wantedFor("cpu", 11))
    }

    /** A bundle with both detectors: app 0.6 (10) loads detector.json; 0.7 (11) loads detector_rg.json once it has both
     *  files, and until then runs the 3-output one and is offered the same bundle again for the extras. */
    @Test fun regionsDetectorOnlyForNewApps() {
        val root = File(System.getProperty("java.io.tmpdir"), "transferner-compat-test").apply { deleteRecursively(); mkdirs() }
        try {
            val dir = File(root, "0.7").apply { mkdirs() }
            val files = mapOf(
                "detector.json" to sidecar(""),
                "detector.tflite" to "m3",
                "species.json" to """{"variants":[{"id":"cpu","label":"CPU","model":"species_cpu.tflite","variant":"v","accelerator":"CPU","signature":"s",
                    "input":{"name":"i","dtype":"float32","shape":[1,3,224,224]},"output":{"name":"o","dtype":"float32","shape":[1,4]},
                    "preprocess":{"resize":"r","crop":"c","mean":[0,0,0],"std":[1,1,1],"layout":"NCHW"}}]}""",
                "head.json" to """{"file":"head.bin","n":0,"dim":4,"scale":1,"temperature":1,"bias":[]}""",
                "catalog.json" to """{"species":[]}""",
            )
            files.forEach { (k, v) -> File(dir, k).writeText(v) }
            val rgMeta = sidecar(""","region_logits":{"name":"output_3","dtype":"int8","shape":[1,9,3]}""").replace("detector.tflite", "detector_rg.tflite")
            val rgFiles = mapOf("detector_rg.json" to rgMeta, "detector_rg.tflite" to "m4")
            val entries = files.map { (k, v) -> """{"path":"$k","bytes":${v.toByteArray().size},"sha256":"x"}""" } +
                rgFiles.map { (k, v) -> """{"path":"$k","bytes":${v.toByteArray().size},"sha256":"x","group":"app:11"}""" }
            File(dir, "manifest.json").writeText("""{"version":"0.7","files":[${entries.joinToString(",")}]}""")
            File(root, "current.txt").writeText("0.7 cpu")
            // bundle downloaded by app 0.6: no detector_rg files on the phone
            assertNull(BundleRepository(root, 10).load()!!.detector.outputs.regionLogits)
            assertFalse(BundleRepository(root, 10).missingExtras())
            val app07 = BundleRepository(root, 11)
            assertNotNull(app07.load())                                   // still runs, on the 3-output detector
            assertNull(app07.load()!!.detector.outputs.regionLogits)
            assertTrue(app07.missingExtras())
            rgFiles.forEach { (k, v) -> File(dir, k).writeText(v) }
            assertEquals("output_3", app07.load()!!.detector.outputs.regionLogits?.name)
            assertEquals("detector_rg.tflite", app07.load()!!.detector.model)
            assertFalse(app07.missingExtras())
            assertNull(BundleRepository(root, 10).load()!!.detector.outputs.regionLogits)
        } finally {
            root.deleteRecursively()
        }
    }
}
