package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.BundleJson
import io.github.cyberneticmedic.transferner.core.DetectorMeta
import io.github.cyberneticmedic.transferner.core.Latest
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

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
}
