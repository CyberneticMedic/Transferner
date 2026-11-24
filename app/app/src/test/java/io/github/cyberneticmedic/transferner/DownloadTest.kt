package io.github.cyberneticmedic.transferner

import io.github.cyberneticmedic.transferner.data.BundleRepository
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File

/**
 * The app's bundle download (latest.json -> manifest -> every file, SHA-256 checked) against a running
 * scripts/serve_bundle.sh. Opt-in: ./gradlew testDebugUnitTest -Ptransferner.server=http://127.0.0.1:8765/
 */
class DownloadTest {
    @Test fun downloadsAndVerifiesBundle() {
        val server = System.getProperty("transferner.server").orEmpty()
        assumeTrue("set -Ptransferner.server to run", server.isNotBlank())
        val root = File(System.getProperty("java.io.tmpdir"), "transferner-bundle-test").apply { deleteRecursively(); mkdirs() }
        try {
            val repo = BundleRepository(root)
            var calls = 0
            val t0 = System.nanoTime()
            val v = repo.download(server) { calls++ }
            println("downloaded bundle $v in %.1f s (%d progress callbacks)".format((System.nanoTime() - t0) / 1e9, calls))
            val files = repo.load()
            assertNotNull(files)
            assertEquals(4411, files!!.catalog.species.size)
            assertEquals(files.head.n, files.catalog.species.size)
            // a second run keeps verified files instead of downloading again
            val t1 = System.nanoTime()
            repo.download(server) {}
            println("re-run (all files verified in place) %.1f s".format((System.nanoTime() - t1) / 1e9))
        } finally {
            root.deleteRecursively()
        }
    }
}
