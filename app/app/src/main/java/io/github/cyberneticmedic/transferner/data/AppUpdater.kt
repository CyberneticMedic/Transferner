package io.github.cyberneticmedic.transferner.data

import android.content.Context
import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.core.content.FileProvider
import androidx.core.content.pm.PackageInfoCompat
import io.github.cyberneticmedic.transferner.core.AppRelease
import io.github.cyberneticmedic.transferner.core.BundleJson
import java.io.File
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest

/**
 * In-app updates from the home server (D-042). scripts/publish_apk.py writes app.json next to the APK; the app
 * compares its versionCode, downloads a newer APK into cacheDir/updates/, checks its SHA-256 and hands it to the
 * system installer. Same package and signing key, so the update replaces the app in place and keeps its data
 * (settings, model bundle, log).
 */
class AppUpdater(private val ctx: Context) {
    val currentVersionCode: Long
        get() = PackageInfoCompat.getLongVersionCode(ctx.packageManager.getPackageInfo(ctx.packageName, 0))
    val currentVersionName: String
        get() = ctx.packageManager.getPackageInfo(ctx.packageName, 0).versionName ?: "?"

    private fun base(server: String) = if (server.endsWith("/")) server else "$server/"

    /** The server's release when it is newer than this app, else null. Blocking. */
    fun check(server: String): AppRelease? {
        val c = URL(base(server) + "app.json").openConnection() as HttpURLConnection
        c.connectTimeout = 8_000; c.readTimeout = 15_000
        if (c.responseCode == 404) return null
        if (c.responseCode != 200) throw IOException("HTTP ${c.responseCode} for app.json")
        val rel = BundleJson.decodeFromString<AppRelease>(c.inputStream.use { it.readBytes().decodeToString() })
        return rel.takeIf { it.newerThan(currentVersionCode) }
    }

    /** Download and verify the APK. Blocking; [onProgress] gets (done, total) bytes. */
    fun download(server: String, rel: AppRelease, onProgress: (Long, Long) -> Unit): File {
        val dir = File(ctx.cacheDir, "updates").apply { mkdirs() }
        dir.listFiles()?.filter { it.name != rel.apk }?.forEach { it.delete() }
        val dst = File(dir, rel.apk)
        if (dst.length() == rel.bytes && sha256(dst) == rel.sha256) return dst
        val part = File(dir, rel.apk + ".part")
        val md = MessageDigest.getInstance("SHA-256")
        val c = URL(base(server) + rel.apk).openConnection() as HttpURLConnection
        c.connectTimeout = 10_000; c.readTimeout = 60_000
        if (c.responseCode != 200) throw IOException("HTTP ${c.responseCode} for ${rel.apk}")
        var done = 0L
        c.inputStream.use { inp ->
            part.outputStream().use { out ->
                val buf = ByteArray(1 shl 16)
                var last = 0L
                while (true) {
                    val n = inp.read(buf)
                    if (n < 0) break
                    out.write(buf, 0, n); md.update(buf, 0, n); done += n
                    if (done - last > (1 shl 19)) { onProgress(done, rel.bytes); last = done }
                }
            }
        }
        val got = md.digest().joinToString("") { "%02x".format(it) }
        if (got != rel.sha256) { part.delete(); throw IOException("checksum mismatch for ${rel.apk}") }
        if (!part.renameTo(dst)) throw IOException("could not store ${rel.apk}")
        onProgress(rel.bytes, rel.bytes)
        return dst
    }

    /** Android asks the user once to let this app install updates ("Install unknown apps"). */
    fun canInstall() = ctx.packageManager.canRequestPackageInstalls()

    fun allowInstallsIntent(): Intent =
        Intent(Settings.ACTION_MANAGE_UNKNOWN_APP_SOURCES, Uri.parse("package:${ctx.packageName}"))

    fun installIntent(apk: File): Intent =
        Intent(Intent.ACTION_VIEW)
            .setDataAndType(FileProvider.getUriForFile(ctx, ctx.packageName + ".files", apk), "application/vnd.android.package-archive")
            .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION or Intent.FLAG_ACTIVITY_NEW_TASK)

    private fun sha256(f: File): String {
        val md = MessageDigest.getInstance("SHA-256")
        f.inputStream().use { inp -> val b = ByteArray(1 shl 16); while (true) { val n = inp.read(b); if (n < 0) break; md.update(b, 0, n) } }
        return md.digest().joinToString("") { "%02x".format(it) }
    }
}
