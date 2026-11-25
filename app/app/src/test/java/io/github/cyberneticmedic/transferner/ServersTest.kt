package io.github.cyberneticmedic.transferner

import com.sun.net.httpserver.HttpServer
import io.github.cyberneticmedic.transferner.data.Prefs
import io.github.cyberneticmedic.transferner.data.Servers
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test
import java.net.InetSocketAddress
import java.net.ServerSocket

/** Home network or tailnet: the saved server first, then the LAN default, then the tailnet; first that answers wins. */
class ServersTest {
    @Test
    fun candidatesKeepOrderAndNormalise() {
        val c = Servers.candidates("http://10.0.0.5:8765", listOf("http://box.example.ts.net:8765", " "))
        assertEquals(listOf("http://10.0.0.5:8765/", Prefs.DEFAULT_SERVER, "http://box.example.ts.net:8765/"), c)
        assertEquals(listOf(Prefs.DEFAULT_SERVER), Servers.candidates(Prefs.DEFAULT_SERVER, emptyList()))
        assertEquals("tailnet", Servers.label("http://box.example.ts.net:8765/"))
        assertEquals("home network", Servers.label(Prefs.DEFAULT_SERVER))
    }

    @Test
    fun firstReachableSkipsADeadServerButKeepsOrder() {
        val dead = ServerSocket(0).use { "http://127.0.0.1:${it.localPort}/" }       // closed port: refused
        val http = HttpServer.create(InetSocketAddress("127.0.0.1", 0), 0)
        http.createContext("/latest.json") { ex ->
            val b = "{}".toByteArray(); ex.sendResponseHeaders(200, b.size.toLong()); ex.responseBody.use { it.write(b) }
        }
        http.start()
        try {
            val live = "http://127.0.0.1:${http.address.port}/"
            assertEquals(live, Servers.firstReachable(listOf(dead, live), 1_000))
            assertEquals(live, Servers.firstReachable(listOf(live, dead), 1_000))
            assertNull(Servers.firstReachable(listOf(dead), 1_000))
        } finally {
            http.stop(0)
        }
    }
}
