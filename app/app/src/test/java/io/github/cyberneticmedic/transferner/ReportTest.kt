package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.core.Report
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import org.junit.Assert.assertEquals
import org.junit.Test

/** The device report sent with "Send" (0.6-0.7.1 recursed in its put helper and crashed the app). */
class ReportTest {
    @Test
    fun flatFieldsBecomeOneJsonObject() {
        val s = Report.json(listOf("app" to "0.7.2", "frames" to 100, "p50" to 58.5f, "gpu" to true, "missing" to null,
            "plan" to "Plan(speciesVariant=cpu, reason=\"x\")"))
        assertEquals("""{"app":"0.7.2","frames":100,"p50":58.5,"gpu":true,"plan":"Plan(speciesVariant=cpu, reason=\"x\")"}""", s)
        val o = Json.parseToJsonElement(s).jsonObject
        assertEquals(5, o.size)
    }

    @Test
    fun aLongReportStaysSmall() {
        val fields = (0 until 40).map { "k$it" to "value $it" }
        val s = Report.json(fields)
        assert(s.length < 1_000) { "report is ${s.length} chars" }
    }
}
