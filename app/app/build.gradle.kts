import java.util.Properties

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.plugin.compose")
    id("org.jetbrains.kotlin.plugin.serialization")
}

// Machine-local settings (gitignored): transferner.tailnet = the bundle server's tailnet addresses, comma-separated
val localProps = Properties().apply {
    rootProject.file("local.properties").takeIf { it.exists() }?.inputStream()?.use { load(it) }
}

android {
    namespace = "io.github.cyberneticmedic.transferner"
    compileSdk {
        version = release(37) { minorApiLevel = 2 }      // Compose BOM 2026.11 needs 37.1+
    }

    defaultConfig {
        applicationId = "io.github.cyberneticmedic.transferner"
        minSdk = 29
        targetSdk = 36
        versionCode = 13
        versionName = "0.7.2"
        ndk { abiFilters += listOf("arm64-v8a") }        // Pixel 7-9 (D-003); keeps the APK small
        buildConfigField("String", "TAILNET_SERVERS", "\"${localProps.getProperty("transferner.tailnet", "")}\"")
    }
    // Every sideloaded APK must carry the same signature, or Android refuses to update the installed app
    // (and its data) in place. The key is a copy of the debug keystore kept outside ~/.android.
    signingConfigs {
        create("sideload") {
            storeFile = file("/models/ml/android/signing/transferner-sideload.keystore")
            storePassword = "android"
            keyAlias = "androiddebugkey"
            keyPassword = "android"
        }
    }
    buildTypes {
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("sideload")   // prototype: sideloaded
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures { compose = true; buildConfig = true }
    packaging { jniLibs { useLegacyPackaging = false } }
    testOptions { unitTests.isReturnDefaultValues = true }
}

dependencies {
    val composeBom = platform("androidx.compose:compose-bom:2026.11.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.activity:activity-compose:1.13.0")
    implementation("androidx.core:core-ktx:1.19.1")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.11.1")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.11.1")
    implementation("androidx.exifinterface:exifinterface:1.4.2")
    implementation("androidx.camera:camera-core:1.7.0")
    implementation("androidx.camera:camera-camera2:1.7.0")
    implementation("androidx.camera:camera-lifecycle:1.7.0")
    implementation("androidx.camera:camera-view:1.7.0")
    implementation("com.google.ai.edge.litert:litert:2.4.0")
    implementation("com.google.ai.edge.litert:litert-gpu:2.4.0")
    implementation("org.jetbrains.kotlinx:kotlinx-serialization-json:1.13.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.11.0")
    testImplementation("junit:junit:4.13.2")
}

// Unit tests compare the Kotlin pre/post-processing with the Python pipeline's goldens (data, not in git).
tasks.withType<Test>().configureEach {
    systemProperty("transferner.data", System.getenv("TRANSFERNER_DATA") ?: "")
    systemProperty("transferner.server", (project.findProperty("transferner.server") as String?) ?: "")
    systemProperty("transferner.mirror", (project.findProperty("transferner.mirror") as String?) ?: "")
    systemProperty("transferner.mirrorServer", (project.findProperty("transferner.mirrorServer") as String?) ?: "")
    systemProperty("java.io.tmpdir", layout.buildDirectory.dir("tmp").get().asFile.path)
    testLogging { showStandardStreams = true }
}
