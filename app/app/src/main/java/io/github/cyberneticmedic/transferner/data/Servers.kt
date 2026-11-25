package io.github.cyberneticmedic.transferner.data

import io.github.cyberneticmedic.transferner.BuildConfig
import java.net.HttpURLConnection
import java.net.URL
import java.util.concurrent.Executors
import java.util.concurrent.TimeUnit

/**
 * The bundle server is the training box. At home it is on the LAN; away, it is reachable over the tailnet when
 * Tailscale runs on the phone (user request 2026-11-24). The candidates, in order: the address saved in Setup,
 * the home-network default, then the tailnet addresses built in from app/local.properties
 * (`transferner.tailnet=`, comma-separated; gitignored, so the tailnet name stays out of the public repo).
 * All are probed at once and the first in that order that answers wins, so home stays on the LAN.
 */
object Servers {
    private fun norm(s: String) = s.trim().let { if (it.isEmpty() || it.endsWith("/")) it else "$it/" }

    val tailnet: List<String> = BuildConfig.TAILNET_SERVERS.split(',').map(::norm).filter { it.isNotEmpty() }

    fun candidates(saved: String, extra: List<String> = tailnet): List<String> =
        (listOf(saved, Prefs.DEFAULT_SERVER) + extra).map(::norm).filter { it.isNotEmpty() }.distinct()

    fun label(server: String): String = if (norm(server) in tailnet || ".ts.net" in server || server.contains("//100.")) "tailnet" else "home network"

    /** Blocking: the first candidate, in order, whose latest.json answers within [timeoutMs]; null if none does. */
    fun firstReachable(candidates: List<String>, timeoutMs: Int = 4_000): String? {
        if (candidates.isEmpty()) return null
        val pool = Executors.newFixedThreadPool(candidates.size)
        try {
            val probes = candidates.map { s -> pool.submit<Boolean> { probe(s, timeoutMs) } }
            for ((s, f) in candidates.zip(probes)) {
                if (runCatching { f.get(2L * timeoutMs + 1_000, TimeUnit.MILLISECONDS) }.getOrDefault(false)) return s
            }
            return null
        } finally {
            pool.shutdownNow()
        }
    }

    private fun probe(server: String, timeoutMs: Int): Boolean = runCatching {
        val c = URL(norm(server) + "latest.json").openConnection() as HttpURLConnection
        c.connectTimeout = timeoutMs; c.readTimeout = timeoutMs
        try {
            c.responseCode == 200
        } finally {
            c.disconnect()
        }
    }.getOrDefault(false)
}
