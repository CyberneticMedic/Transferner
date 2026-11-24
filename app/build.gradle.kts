// AGP 9 compiles Kotlin itself (built-in Kotlin); the kotlin-android entry only raises the Kotlin version
// from AGP's bundled 2.2.10 to match the Compose and serialization compiler plugins.
plugins {
    id("com.android.application") version "9.5.2" apply false
    id("org.jetbrains.kotlin.android") version "2.4.21" apply false
    id("org.jetbrains.kotlin.plugin.compose") version "2.4.21" apply false
    id("org.jetbrains.kotlin.plugin.serialization") version "2.4.21" apply false
}
